import torch

class PrefixEncoder(torch.nn.Module):
    ''' Copy from THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py
        The torch.nn model to encode the prefix

        Input shape: (batch-size, prefix-length)

        Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config, with_init=False):
        super().__init__()
        self.pre_seq_len = config.pre_seq_len
        self.prefix_projection = config.prefix_projection
        self.prefix_hidden_size = config.prefix_hidden_size

        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(self.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, self.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(self.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(self.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values

    def reset_params(self, input_embeds):
        assert input_embeds.shape == self.embedding.weight.shape
        self.embedding.weight = torch.nn.Parameter(input_embeds)


class KnowConcat(torch.nn.Module):
    """在全连接层里面拼接 keys 和 values
    keys encoder.layer.[0-23].intermediate.dense
    values encoder.layer.[0-23].output.dense

    size: 2 * padding_layers * num_rel * 1024
    """
    def __init__(self, know_len, hidden_size=1024, padding_layers=6):
        super().__init__()
        self.embedding = torch.nn.Embedding(2 * padding_layers * know_len, hidden_size)

    def forward(self, know: torch.Tensor):
        return self.embedding(know)

    def reset_params(self, input_embeds):
        assert input_embeds.shape == self.embedding.weight.shape
        self.embedding.weight = torch.nn.Parameter(input_embeds)
