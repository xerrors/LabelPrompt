import torch
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import RobertaModel, MaskedLMOutput, RobertaPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaLMHead

from models.components import PrefixEncoder


class RobertaForPrefix(RobertaPreTrainedModel):
    ''' Copy from YHUDM/P-tuning-v2
        link: https://github.com/THUDM/P-tuning-v2/blob/850a67d15d5a9e19587cb84bc0f215685d32e02f/model/sequence_classification.py#L326
        and https://github.com/huggingface/transformers/blob/dee6f01636746dae6e73c3d258870b04d1b0832d/src/transformers/models/roberta/modeling_roberta.py#L1030
    '''

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--use_prompt", type=bool, default=True, help="Whether to use prompt in the dataset.")
        parser.add_argument("--init_answer_words", type=int, default=1, )
        parser.add_argument("--init_type_words", type=int, default=1, )
        parser.add_argument("--init_answer_words_by", type=str, default="whole_word", )
        parser.add_argument("--use_template_words", type=int, default=1, )

        parser.add_argument("--pre_seq_len", type=int, default=144)
        parser.add_argument("--prefix_projection", action="store_true", default=False)
        parser.add_argument("--prefix_hidden_size", type=int, default=512, )
        parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, )
        parser.add_argument("--tune_embeddings", action="store_true", default=False)
        return parser

    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config) -> None:
        super().__init__(config)

        if config.is_decoder:
            print(
                "If you want to use `RobertaForPrefix` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        
        try:
            config.pre_seq_len = config.args.pre_seq_len
            config.prefix_projection = config.args.prefix_projection
            config.prefix_hidden_size = config.args.prefix_hidden_size
            config.hidden_dropout_prob = config.args.hidden_dropout_prob
            config.tune_embeddings = config.args.tune_embeddings
        except:
            config.pre_seq_len = 144
            config.prefix_projection = False
            config.prefix_hidden_size = 512
            config.hidden_dropout_prob = 0.1
            config.tune_embeddings = False

        self.config = config
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        for param in self.roberta.parameters():
            param.requires_grad = False
        
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        self.init_weights()

        bert_param = 0
        for name, param in self.roberta.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param)) # 9860105

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def get_output_embeddings(self):
        return self.lm_head.decoder
    

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):   
    
        r"""
        https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/roberta/modeling_roberta.py#L1033
        
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = attention_mask.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

