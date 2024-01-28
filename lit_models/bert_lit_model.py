import json
from operator import xor
import os
import time
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from functools import partial
import matplotlib.pyplot as plt
import wandb
import numpy as np
from utils import plt_mat_and_save_to_model_dir, plt_mats_and_save_to_model_dir
from lit_models.base import LR, BaseLitModel
from lit_models.util import f1_eval, compute_f1, acc, f1_score
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers.activations import ACT2FN, gelu

from lit_models.components import PrePromptEncoder, RelClassifier, LabelSmoothingCrossEntropy

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()



class BertLitModel(BaseLitModel):
    """ TAG: BertLitModel
        use AutoModelForMaskedLM, and select the output by another layer in the lit model
    """
    @staticmethod
    def add_to_argparse(parser):
        BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--w_ce", type=float, default=1.0, help="")
        parser.add_argument("--w_ke", type=float, default=0.0, help="")
        parser.add_argument("--w_type", type=float, default=0.0, help="")
        parser.add_argument("--w_cls", type=float, default=0.0, help="")
        parser.add_argument("--w_pre", type=float, default=0.0, help="")
        parser.add_argument("--w_ent", type=float, default=0.0, help="")
        parser.add_argument("--w_ent_two", type=float, default=0.0, help="")
        parser.add_argument("--w_stru", type=float, default=0.0, help="")
        parser.add_argument("--w_attn", type=float, default=0.0, help="")
        parser.add_argument("--use_ent_type", type=str, default="", help="")
        parser.add_argument("--init_rel_subtype", type=str, default="", help="")
        parser.add_argument("--similarity_func", type=str, default="l2", help="")
        parser.add_argument("--use_delay_w_ke", type=float, default=0.0, help="")
        parser.add_argument("--use_delay_w_type", type=float, default=0.0, help="")
        parser.add_argument("--use_type_projection", type=str, default="", help="")
        parser.add_argument("--use_prompt_projection", type=str, default="", help="")
        parser.add_argument("--cls_type", type=str, default="entity", help="")
        parser.add_argument("--use_pre_stage", type=str, default="", help="")
        parser.add_argument("--use_pre_prompt", type=str, default="", help="")
        parser.add_argument("--attn_loss_type", type=str, default="attn:0:mean:row", help="")
        parser.add_argument("--ce_type", type=str, default="additional", help="")

        parser.add_argument("--add_label_options", type=str, default="false", help="")
        parser.add_argument("--check_layer_output", type=str, default="default", help="")

        ## debug
        parser.add_argument("--padding_layers", type=float, default=0.25, help="")

        parser.add_argument("--len_first_stage_prompt", type=int, default=3)
        parser.add_argument("--t_gamma", type=float, default=0.3, help="")
        return parser

    def __init__(self, model, args, tokenizer, pre_model=None):
        super().__init__(model, args)
        self.args = args
        self.tokenizer = tokenizer
        self.pre_model = pre_model

        with open(f"{args.data_dir}/rel2id.json","r") as file:
            self.rel2id = json.load(file)
            self.id2rel = dict(zip(self.rel2id.values(), self.rel2id.keys()))

        self.Na_num = 0
        for k, v in self.rel2id.items():
            if k == "NA" or k == "no_relation" or k == "Other":
                self.Na_num = v
                break
        self.num_relation = len(self.rel2id)
        if args.w_ent or args.w_ent_two or args.w_type:
            self.num_sub = len(self.args.sub_types)
            self.num_obj = len(self.args.obj_types)
        self.mid_size = 512
        self.type_hidden_size = 256

        # linear layer should be defined here
        if self.args.use_emb_projection:
            self.projection_sub = torch.nn.Sequential(
                torch.nn.Linear(model.config.hidden_size, self.mid_size),
                torch.nn.ReLU(inplace=True)
            )
            self.projection_obj = torch.nn.Sequential(
                torch.nn.Linear(model.config.hidden_size, self.mid_size),
                torch.nn.ReLU(inplace=True)
            )
            self.projection_rel = torch.nn.Sequential(
                torch.nn.Linear(model.config.hidden_size, self.mid_size),
                torch.nn.ReLU(inplace=True)
            )

        if "linear" in self.args.ke_type.split(":")[1] or self.args.w_pre != 0:
            linear_input_size = self.mid_size if self.args.use_emb_projection else model.config.hidden_size
            self.linear_transe = nn.Linear(linear_input_size * 3, self.mid_size)

        if self.args.use_type_projection:
            self.type_projection = nn.Linear(model.config.hidden_size, self.type_hidden_size)

        if self.args.cls_type and self.args.w_cls != 0:
            self.rel_classifier = RelClassifier(model.config.hidden_size * 3, self.num_relation)

        if self.args.ce_type == "vocab":
            self.ce_classifier = RelClassifier(model.config.hidden_size, self.num_relation)

        if self.args.w_stru:
            self.stru_classifier = RelClassifier(model.config.hidden_size * 3, 2)  # pos neg

        # if self.args.use_prompt_projection:
        #     self.prompt_projection = torch.nn.Sequential(
        #         torch.nn.Linear(model.config.hidden_size, self.mid_size),
        #         torch.nn.ReLU(inplace=True),
        #         torch.nn.Linear(self.mid_size, model.config.hidden_size)
        #     )

        if self.args.use_pre_prompt:
            self.pre_encoder = PrePromptEncoder(self.args.len_first_stage_prompt, model.config.hidden_size)

        # # init loss function
        # self.loss_fn = multilabel_categorical_crossentropy if "dialogue" in args.data_dir else nn.CrossEntropyLoss()
        self.loss_fn = LabelSmoothingCrossEntropy() if args.label_smoothing else nn.CrossEntropyLoss()

        self.eval_fn = f1_eval if "dialogue" in args.data_dir else partial(f1_score, rel_num=self.num_relation, na_num=self.Na_num) # https://www.learnpython.org/en/Partial_functions

        self.best_f1 = 0

        # ! w_ke will be modified in trainging see: training_epoch_end
        self.w_ce = self.args.w_ce
        self.w_ke = 0 if args.use_delay_w_ke else args.w_ke
        self.w_type = 0 if args.use_delay_w_type else args.w_type
        self.w_cloze = args.w_cloze
        self.w_cls = args.w_cls
        self.w_pre = args.w_pre
        self.w_ent = args.w_ent
        self.w_stru = args.w_stru
        self.w_ent_two = args.w_ent_two
        self.w_attn = args.w_attn

        self._init_label_word()

    def _init_label_word(self, ):
        args = self.args
        dataset_name = args.data_dir.split("/")[1]
        with open(f"./dataset/{dataset_name}/rel2id.json", "r") as file:
            t = json.load(file)
            label_list = sorted(list(t.keys()), key=lambda x: t[x])

        if self.args.init_answer_words == 2:
            rel_label = None
            with open(f"./dataset/{dataset_name}/rel_label.json", "r") as file:
                rel_label = json.load(file)
            assert rel_label

        assert label_list

        label_word_list = []
        for label in label_list:

            if label == 'no_relation' or label == "NA" or label == "Other":
                label_word_id = self.tokenizer.encode('no relation', add_special_tokens=False)
                label_word_list.append(torch.tensor(label_word_id))

            else:
                if self.args.init_answer_words == 2 and rel_label.get(str(t[label])):
                    label = rel_label[str(t[label])]
                    label = " ".join(label).lower()
                elif self.args.init_answer_words == 1:
                    label = label.lower().replace("per","person").replace("org","organization")
                    label = label.replace(":"," ").replace("_"," ").replace("/"," ")
                label_word_id = self.tokenizer(label, add_special_tokens=False)['input_ids']
                # print(label, label_word_id)
                label_word_list.append(torch.tensor(label_word_id))

        label_word_idx = pad_sequence([x for x in label_word_list], batch_first=True, padding_value=0)

        # need sub and obj type
        if args.w_ent or args.w_ent_two or args.w_type:
            sub_word_list = []
            for sub in self.args.sub_types:
                sub = sub.replace(":"," ").replace("_"," ").replace("/"," ")
                sub_id = self.tokenizer(sub, add_special_tokens=False)['input_ids']
                sub_word_list.append(torch.tensor(sub_id))
            sub_word_idx = pad_sequence([x for x in sub_word_list], batch_first=True, padding_value=0)

            obj_word_list = []
            for obj in self.args.obj_types:
                obj = obj.replace(":"," ").replace("_"," ").replace("/"," ")
                obj_id = self.tokenizer(obj, add_special_tokens=False)['input_ids']
                obj_word_list.append(torch.tensor(obj_id))
            obj_word_idx = pad_sequence([x for x in obj_word_list], batch_first=True, padding_value=0)


        num_labels = len(label_word_idx)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Init Prompt Tokens
        with torch.no_grad():
            word_embeddings = self.model.get_input_embeddings() # A torch module mapping vocabulary to hidden states. Embedding(50300, 1024)

            if self.args.use_self_mlm:
                continous_label_word = [a[0] for a in self.tokenizer(label_list, add_special_tokens=False)['input_ids']]
            else:
                continous_label_word = [a[0] for a in self.tokenizer([f"[C{i}]" for i in range(1, num_labels+1)], add_special_tokens=False)['input_ids']]

            self.label_st_id = continous_label_word[0]

            # init label words
            if self.args.init_answer_words:
                if self.args.init_answer_words_by == "vocab_mean":
                    word_embeddings.weight[continous_label_word] = torch.mean(word_embeddings.weight, dim=0)
                elif self.args.init_answer_words_by == "one_token":
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = word_embeddings.weight[idx][-1]
                elif self.args.init_answer_words_by == "whole_word":
                    for i, idx in enumerate(label_word_idx): # idx: e.g. tensor([265, 138, 18727, 0, 0, 0])
                        idx = idx[idx!=0]
                        word_embeddings.weight[continous_label_word[i]] = torch.mean(word_embeddings.weight[idx], dim=0) # mean
                else:
                    raise ValueError(f"`args.init_answer_words_by`: {self.args.init_answer_words_by}")

            # init type words
            if self.args.init_type_words:
                if self.args.init_type_words == 1:
                    sub_meaning_word = [a[0] for a in self.tokenizer(["person", "organization", "location", "date", "country"], add_special_tokens=False)['input_ids']] # e.g. [5970, 17247, 41829, 10672, 12659]
                    obj_meaning_word = [a[0] for a in self.tokenizer(["person", "organization", "location", "date", "country"], add_special_tokens=False)['input_ids']] # e.g. [5970, 17247, 41829, 10672, 12659]

                elif self.args.init_type_words == 2:
                    meaning_word = torch.cat([torch.cat(sub_labels) for sub_labels in self.rel_subtype.values()])
                    sub_meaning_word = meaning_word
                    obj_meaning_word = meaning_word

                so_word = [a[0] for a in self.tokenizer(["[sub]","[obj]"], add_special_tokens=False)['input_ids']] # e.g. [50294, 50293]
                word_embeddings.weight[so_word[0]] = self.get_init_embeds(word_embeddings.weight[sub_meaning_word], type='so') # mean, sub = obj
                word_embeddings.weight[so_word[1]] = self.get_init_embeds(word_embeddings.weight[obj_meaning_word], type='so') # mean, sub = obj

                sp_mark = [a[0] for a in self.tokenizer(["[s]", "[/s]", "[o]", "[/o]"], add_special_tokens=False)['input_ids']]
                word_embeddings.weight[sp_mark[0]] = self.get_init_embeds(word_embeddings.weight[sub_meaning_word], type='sp')
                word_embeddings.weight[sp_mark[1]] = self.get_init_embeds(word_embeddings.weight[sub_meaning_word], type='sp')
                word_embeddings.weight[sp_mark[2]] = self.get_init_embeds(word_embeddings.weight[obj_meaning_word], type='sp')
                word_embeddings.weight[sp_mark[3]] = self.get_init_embeds(word_embeddings.weight[obj_meaning_word], type='sp')

                relation_word = [a[0] for a in self.tokenizer(["relation"], add_special_tokens=False)['input_ids']]
                mark_prefix_id, mark_suffix_id = [item[0] for item in self.tokenizer(["[m]","[/m]"], add_special_tokens=False)['input_ids']]
                word_embeddings.weight[mark_prefix_id] = self.get_init_embeds(word_embeddings.weight[relation_word], type='mr')
                word_embeddings.weight[mark_suffix_id] = self.get_init_embeds(word_embeddings.weight[relation_word], type='mr')

            if self.args.use_cloze:
                cloze_id = self.tokenizer(["[cloze]"], add_special_tokens=False)['input_ids'][0][0]
                word_embeddings.weight[cloze_id] = word_embeddings.weight[self.tokenizer.mask_token_id]

            if args.w_ent or args.w_ent_two or args.w_type:
                self.rel_subtype = {}
                sub_type_ids = [a[0] for a in self.tokenizer([f"[S{i}]" for i in range(len(self.args.sub_types))], add_special_tokens=False)['input_ids']]
                obj_type_ids = [a[0] for a in self.tokenizer([f"[O{i}]" for i in range(len(self.args.obj_types))], add_special_tokens=False)['input_ids']]
                for i, idx in enumerate(sub_word_idx):
                    word_embeddings.weight[sub_type_ids[i]] = torch.mean(word_embeddings.weight[idx], dim=0)

                for i, idx in enumerate(obj_word_idx):
                    word_embeddings.weight[obj_type_ids[i]] = torch.mean(word_embeddings.weight[idx], dim=0)

                self.sub_type_st_id = sub_type_ids[0]
                self.obj_type_st_id = obj_type_ids[0]

                for key, value in args.rel_ent_types.items():
                    sub_token = sub_type_ids[value[0]]
                    obj_token = obj_type_ids[value[1]]
                    self.rel_subtype[key] = [sub_token, obj_token]

            assert torch.equal(self.model.get_input_embeddings().weight, word_embeddings.weight)
            assert torch.equal(self.model.get_input_embeddings().weight, self.model.get_output_embeddings().weight)

            # ./models/roberta_for_knowledge.py line123
            if self.args.debug1 == "init_pre_encoder_params":
                past_keys = word_embeddings.weight[continous_label_word].clone()
                past_keys = past_keys.unsqueeze(0).repeat([24, 1, 1])
                past_values = word_embeddings.weight[continous_label_word].clone()
                past_values = past_values.unsqueeze(0).repeat([24, 1, 1])

                past_key_values = torch.cat([past_keys.unsqueeze(0), past_values.unsqueeze(0)], dim=0).permute(2,0,1,3)
                assert past_key_values.shape == (self.args.rel_num, 2, 24, self.model.config.hidden_size)
                past_key_values = past_key_values.reshape(self.args.rel_num, -1)
                self.model.prefix_encoder.reset_params(past_key_values)

            elif self.args.debug1 == "concat_know":
                know_keys = word_embeddings.weight[continous_label_word].clone()
                know_values = word_embeddings.weight[continous_label_word].clone()
                if self.args.padding_layers < 0:
                    reset_layers = self.model.roberta.encoder.layer[int(self.model.config.num_hidden_layers * self.args.padding_layers):]
                else:
                    reset_layers = self.model.roberta.encoder.layer[:int(self.model.config.num_hidden_layers * self.args.padding_layers)]

                for layer in reset_layers:
                    layer.intermediate.dense.weight = torch.nn.Parameter(torch.cat([layer.intermediate.dense.weight, know_keys], dim=0))
                    layer.intermediate.dense.bias = torch.nn.Parameter(torch.cat([layer.intermediate.dense.bias, torch.zeros(self.args.rel_num)], dim=0))
                    layer.output.dense.weight = torch.nn.Parameter(torch.cat([layer.output.dense.weight, know_values.T], dim=-1))

            if self.args.debug3 == "use_key_values_projection":
                nn.init.constant_(self.model.key_projection.weight, 1)
                nn.init.constant_(self.model.value_projection.weight, 1)
                self.model.key_projection.weight = nn.Parameter(torch.eye(self.model.config.hidden_size))
                self.model.value_projection.weight = nn.Parameter(torch.eye(self.model.config.hidden_size))
                self.model.key_projection.bias = nn.Parameter(torch.zeros(self.model.config.hidden_size))
                self.model.value_projection.bias = nn.Parameter(torch.zeros(self.model.config.hidden_size))


        self.word2label = continous_label_word # a continous list e.g. [50269, 50270, 50271, 50272, 50273, 50274, 50275, 50276, 50277, 50278, 50279, 50280, 50281, 50282, ...]
        self.model.word2label = continous_label_word # a continous list e.g. [50269, 50270, 50271, 50272, 50273, 50274, 50275, 50276, 50277, 50278, 50279, 50280, 50281, 50282, ...]

        # reset lmhead
        if self.args.debug4 == "reset_dense":
            self.model.lm_head.dense.weight = nn.Parameter(nn.Embedding(self.model.config.hidden_size, self.model.config.hidden_size).weight)

    def get_init_embeds(self, embeds, type=None):
        """ get the init embeddings of special tag by existed embeddings. default `mean` """

        mean_embeds = torch.mean(embeds, dim=0)
        return mean_embeds

    def forward(self, x):
        return self.model(x)

    def convert_ids_to_embeds(self, input_ids, model):
        model_embeddings = model.get_input_embeddings().weight
        input_embeddings = model_embeddings[input_ids]

        # if self.args.use_prompt_projection:
        #     prompt_embds = input_embeddings[:, 1:6]
        #     prompt_embds = self.prompt_projection(prompt_embds)
        #     input_embeddings[:, 1:6] = prompt_embds

        return input_embeddings

    def get_first_stage_output(self, ids, mask):
        len_first_stage_prompt = self.args.len_first_stage_prompt

        embeddings = self.convert_ids_to_embeds(ids, self.pre_model)

        if self.args.use_pre_prompt != "long_mask":
            embeddings = self.convert_ids_to_embeds(ids, self.pre_model)
            soft_prompt_idx = torch.arange(len_first_stage_prompt).repeat(mask.size(0), 1).cuda()
            soft_prompt = self.pre_encoder(soft_prompt_idx)
            embeddings[:, 1:len_first_stage_prompt+1] = soft_prompt

        hidden_state = self.pre_model(inputs_embeds=embeddings, attention_mask=mask, return_dict=True, output_hidden_states=True).hidden_states[-1]

        if self.args.use_pre_prompt == "default":
            mask_idx = (ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            output = hidden_state[torch.arange(len(ids)), mask_idx]
        elif self.args.use_pre_prompt == "cls":
            output = hidden_state[:, 0]
        elif self.args.use_pre_prompt == "long":
            output = hidden_state[:, :len_first_stage_prompt+1] # bsz, len_first_stage_prompt, hidden_size
        elif self.args.use_pre_prompt == "long_mask":
            bsz = hidden_state.shape[0]
            mask_idx = (ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1].view(bsz, len_first_stage_prompt) # must be right
            output = torch.stack([hidden_state[i, mask_idx[i]] for i in range(bsz)])
            output = torch.cat([hidden_state[:, 0].unsqueeze(1), output], dim=1)

        return output

    def get_second_embeddings(self, batch, return_ent_emb=False):
        input_ids, attention_mask, labels, so, sub_ids, sub_mask, obj_ids, obj_mask = batch

        input_embeddings = self.convert_ids_to_embeds(input_ids=input_ids, model=self.pre_model)
        sub_emb = self.get_first_stage_output(sub_ids, sub_mask)
        obj_emb = self.get_first_stage_output(obj_ids, obj_mask)

        if self.args.use_pre_prompt == "long" or self.args.use_pre_prompt == "long_mask":
            input_embeddings[:, 1:self.args.len_first_stage_prompt+2] = sub_emb
            input_embeddings[:, self.args.len_first_stage_prompt+2:self.args.len_first_stage_prompt*2+3] = obj_emb
        else:
            for i in range(input_ids.size(0)):
                input_embeddings[i, so[i][0]-1] = sub_emb[i]
                input_embeddings[i, so[i][2]-1] = obj_emb[i]

        if return_ent_emb:
            return input_embeddings, sub_emb, obj_emb
        else:
            return input_embeddings


    # Train
    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        # return train_step_ext(self, batch, batch_idx)
        if self.args.use_cloze:
            input_ids, attention_mask, labels, so, masked_ids, masked_lm_positions, masked_lm_labels = batch
            input_embeddings = self.convert_ids_to_embeds(input_ids=masked_ids, model=self.model)

        elif self.args.use_pre_prompt:
            assert self.args.use_pre_stage
            input_ids, attention_mask, labels, so, sub_ids, sub_mask, obj_ids, obj_mask = batch
            input_embeddings, sub_emb, obj_emb = self.get_second_embeddings(batch, return_ent_emb=True)

        else:
            input_ids, attention_mask, labels, so = batch # (bsz, 256), (bsz, 256), (bsz,), (bsz, 4)
            input_embeddings = self.convert_ids_to_embeds(input_ids=input_ids, model=self.model)

        if self.args.use_cloze or self.args.use_pre_prompt:
            result = self.model(inputs_embeds=input_embeddings, attention_mask=attention_mask,
                                return_dict=True, output_hidden_states=True, output_attentions=True)
        else:
            result = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                return_dict=True, output_hidden_states=True, output_attentions=True,
                                output_act_value=True)

        loss = None

        act_value = result.act_value
        # layer_digits = ()
        # for hid_stat in result.hidden_states:
        #     layer_digits += (self.model.lm_head(hid_stat))

        assert bool(self.args.w_cls > 0) ^ bool(self.args.w_ce > 0)  # must have one and only have one

        if self.args.w_ce:
            logits = self.get_logits(result, input_ids, self.args.ce_type, so=so)
            lmhead_loss = self.loss_fn(logits, labels)
            loss = loss + self.w_ce * lmhead_loss if loss else self.w_ce * lmhead_loss
            self.log("Train/lmhead_loss", lmhead_loss)

        elif self.args.w_cls:
            cls_logits = self.get_rels(result.hidden_states[-1], so)
            cls_loss = self.loss_fn(cls_logits, labels)
            loss = loss + self.w_cls * cls_loss if loss else self.w_cls * cls_loss
            self.log("Train/cls_loss", cls_loss)

        if self.args.w_stru:
            assert self.args.use_pre_stage
            stru_loss = self.first_stage_stru(sub_emb, obj_emb, labels)
            loss += self.w_stru * stru_loss
            self.log("Train/stru_loss", stru_loss)

        if self.args.w_pre:
            assert self.args.use_pre_stage
            pre_ke_loss = self.pre_ke(input_ids, result.hidden_states[-1], sub_emb, obj_emb, labels, so)
            loss += self.w_pre * pre_ke_loss
            self.log("Train/pre_ke_loss", pre_ke_loss)

        if self.args.w_ke:
            ke_loss = self.ke_loss(result.hidden_states[-1], labels, so, input_ids, attention_mask)
            loss += self.w_ke * ke_loss
            self.log("Train/ke_loss", ke_loss)

        if self.args.w_cloze:
            cloze_loss = self.cloze_loss(result.logits, masked_lm_positions, masked_lm_labels, input_ids)
            loss = loss + self.w_cloze * cloze_loss if loss else self.w_cloze * cloze_loss
            self.log("Train/cloze_loss", cloze_loss)

        if self.args.w_ent:
            ent_loss = self.ent_type_loss(labels, so, sub_emb, obj_emb)
            loss += self.w_ent * ent_loss
            self.log("Train/ent_loss", ent_loss)

        if self.args.w_type:
            type_loss = self.type_loss(result.hidden_states[-1], result.logits, labels, so, input_ids, attention_mask)
            loss += self.w_type * type_loss
            self.log("Train/type_loss", type_loss)

        if self.args.w_ent_two:
            ent2_loss = self.second_ent_type_loss(input_ids, labels, result, so)
            loss += self.w_ent_two * ent2_loss
            self.log("Train/ent2_loss", ent2_loss)

        if self.args.w_attn:
            attn_loss = self.attn_loss(input_ids, result.attentions, attention_mask, labels, act_value)
            loss += self.w_attn * attn_loss
            self.log("Train/attn_loss", attn_loss)

        if self.args.check_layer_output and random.random() < 0.01:
            label_embeds = self.get_label_embeddings()
            mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            sim_out = []
            sim_head_out = []
            for hs in result.hidden_states[1:]:
                hs = hs[torch.arange(self.batch_size), mask_idx]
                sim = torch.matmul(hs, label_embeds.T)
                sim_head = self.model.lm_head(hs)[:, self.label_st_id:self.label_st_id+self.num_relation]
                sim_out.append(sim)
                sim_head_out.append(sim_head)

            input_length = sum(attention_mask[0]).item() + self.num_relation
            atts = torch.stack([torch.sum(att, dim=0)[0][:input_length,:input_length] for att in result.attentions])
            sim_out = torch.stack(sim_out, dim=0)
            sim_head_out = torch.stack(sim_head_out, dim=0)
            cur_state = f"(epoch {self.current_epoch}, step {self.global_step})"
            att_info = "{name}({idx})".format(name=self.id2rel[labels[0].item()], idx=labels[0].item())
            plt_mats_and_save_to_model_dir(mats=sim_out, out_dir=self.args.model_dir, title=f"sim {cur_state}")
            plt_mats_and_save_to_model_dir(mats=sim_head_out, out_dir=self.args.model_dir, title=f"sim_head {cur_state}")
            plt_mats_and_save_to_model_dir(mats=sim_out.permute(1,0,2), Nr=2, Nc=2, out_dir=self.args.model_dir, title=f"sim-batch 4 {cur_state}")
            plt_mats_and_save_to_model_dir(mats=sim_head_out.permute(1,0,2), Nr=2, Nc=2, out_dir=self.args.model_dir, title=f"sim_head 4 {cur_state}")
            plt_mats_and_save_to_model_dir(mats=atts, out_dir=self.args.model_dir, title=f"attns of batch 0{cur_state} {att_info}")

            if self.args.debug4 == "layer_loss":
                for head_out in sim_head_out:
                    loss += self.loss_fn(head_out, labels)

        self.log("Train/sum_loss", loss)

        cur_lr = self.lr_schedulers().get_last_lr()[0]
        self.log("trainer/lr", cur_lr)

        return loss

    def get_label_embeddings(self):
        weights = self.model.get_output_embeddings().weight[self.word2label]
        return weights


    def attn_loss(self, input_ids, attns, attention_mask, labels, act_value):
        """ Calc attens between words and labels.

        with add_input_options in debug3

        1. layers: 1/6/0 means last/last_6/mean
        2. heads: first/mean
        3. calc_type: col/row/mean
        """
        bsz = len(input_ids)
        loss_type, layers, heads, calc_type = self.args.attn_loss_type.split(":")

        if self.args.debug4 == "cls":
            mask_idx = torch.tensor([0] * bsz).cuda()
        else:
            mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        scores = attns if loss_type == "attn" else act_value
        scores = sum(scores[-int(layers):]) / len(scores[-int(layers):])

        if loss_type == "attn":
            assert self.args.add_label_options != "false", "You have to set `add_label_options` with any value without `false`."
            scores = scores[:, 0] if heads == "first" else torch.mean(scores, dim=1)
            mask_scores = (scores if calc_type == "col" else scores.permute(0,2,1))[torch.arange(bsz), mask_idx]

            label_st = (input_ids == self.label_st_id).nonzero(as_tuple=True)[1]
            label_scores = pad_sequence([mask_scores[i, label_st[i]:label_st[i]+self.args.rel_num] for i in range(bsz)], batch_first=True)
        else:
            mask_scores = scores[torch.arange(bsz), mask_idx]
            label_scores = mask_scores[:, -self.num_relation:]

        loss = self.loss_fn(label_scores, labels)

        if random.random() < 0.01:
            input_length = sum(attention_mask[0]).item() + self.num_relation
            plt_mat_and_save_to_model_dir(scores[0][:input_length,:input_length], self.args.model_dir,
                "{name}({idx}, {value:.4f})".format(name=self.id2rel[labels[0].item()], idx=labels[0].item(), value=label_scores[0, labels[0].item()]))

        return loss

    def ent_type_loss(self, labels, so, sub_emb, obj_emb):
        bsz = len(so)
        assert self.args.use_ent_type
        weights = self.pre_model.get_output_embeddings().weight
            # current just use embeddings with any projection.
        prefix_type = torch.stack([torch.mean(weights[self.rel_subtype[labels[i].item()][0]], dim=0) for i in range(bsz)])
        suffix_type = torch.stack([torch.mean(weights[self.rel_subtype[labels[i].item()][1]], dim=0) for i in range(bsz)])

        if self.args.use_pre_prompt == "long" or self.args.use_pre_prompt == "long_mask":
            sub_emb = sub_emb[:, 0]
            obj_emb = obj_emb[:, 0]

        if self.args.use_ent_type == "lmhead":
            assert self.args.use_pre_stage == "share"

            sub_logits = self.model.lm_head(sub_emb)[:, self.sub_type_st_id:self.sub_type_st_id+self.num_sub]
            sub_types = torch.tensor([self.args.rel_ent_types[labels[i].item()][0] for i in range(len(labels))]).cuda()
            sub_loss = self.loss_fn(sub_logits, sub_types)

            obj_logits = self.model.lm_head(obj_emb)[:, self.obj_type_st_id:self.obj_type_st_id+self.num_obj]
            obj_types = torch.tensor([self.args.rel_ent_types[labels[i].item()][1] for i in range(len(labels))]).cuda()
            obj_loss = self.loss_fn(obj_logits, obj_types)

            loss = sub_loss + obj_loss
        # FIXME looooooose constraint
        else:
            if self.args.use_type_projection:
                prefix_type = self.type_projection(prefix_type)
                suffix_type = self.type_projection(suffix_type)
                mark_prefix = self.type_projection(sub_emb)
                mark_suffix = self.type_projection(obj_emb)

            # similarity func
            similarity_funcs = {
                "cosine": lambda a, b: 1 - torch.cosine_similarity(a, b, dim=-1),
                "l2": lambda a, b: torch.norm(a-b, p=2, dim=-1)
            }

            similarity_func = similarity_funcs[self.args.similarity_func]
            loss = similarity_func(mark_prefix, prefix_type).sum() + similarity_func(mark_suffix, suffix_type).sum()

        return loss / bsz

    def second_ent_type_loss(self, input_ids, labels, result, so):
        bsz = len(so)
        assert self.args.use_ent_type

        if self.args.use_ent_type == "lmhead":
            sub_tag = torch.stack([result.hidden_states[-1][i, so[i][0]-1] for i in range(bsz)]).cuda()
            obj_tag = torch.stack([result.hidden_states[-1][i, so[i][2]-1] for i in range(bsz)]).cuda()

            sub_logits = self.model.lm_head(sub_tag)[:, self.sub_type_st_id:self.sub_type_st_id+self.num_sub]
            obj_logits = self.model.lm_head(obj_tag)[:, self.obj_type_st_id:self.obj_type_st_id+self.num_obj]
        elif self.args.use_ent_type == "mask":
            mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            bs = input_ids.shape[0]
            assert mask_idx.shape[0] == bs, "only one mask in sequence!"

            mask_output = result.logits[torch.arange(bs), mask_idx]
            sub_logits = mask_output[:, self.sub_type_st_id:self.sub_type_st_id+self.num_sub]
            obj_logits = mask_output[:, self.obj_type_st_id:self.obj_type_st_id+self.num_obj]

        sub_types = torch.tensor([self.args.rel_ent_types[labels[i].item()][0] for i in range(len(labels))]).cuda()
        sub_loss = self.loss_fn(sub_logits, sub_types)
        obj_types = torch.tensor([self.args.rel_ent_types[labels[i].item()][1] for i in range(len(labels))]).cuda()
        obj_loss = self.loss_fn(obj_logits, obj_types)

        loss = sub_loss + obj_loss

        return loss

    def training_epoch_end(self, outputs) -> None:
        super().training_epoch_end(outputs)
        if self.args.use_ascend_w_ke:
            if self.current_epoch == 8:
                self.w_ke = self.w_ke * 10
            if self.current_epoch == 18:
                self.w_ke = self.w_ke * 10

        if self.current_epoch == 11:
            print("Lie Sha Shi Ke.")

        self.w_ke = 0 if self.current_epoch < self.args.max_epochs * self.args.use_delay_w_ke else self.args.w_ke
        self.w_type = 0 if self.current_epoch < self.args.max_epochs * self.args.use_delay_w_type else self.args.w_type

    # Validation
    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument

        if self.args.use_pre_prompt:
            input_ids, attention_mask, labels, so, sub_ids, sub_mask, obj_ids, obj_mask = batch
            input_embeddings, sub_emb, obj_emb = self.get_second_embeddings(batch, return_ent_emb=True)

        else:
            input_ids, attention_mask, labels, so = batch[:4] # (bsz, 256), (bsz, 256), (bsz,), (bsz, 4)
            input_embeddings = self.convert_ids_to_embeds(input_ids=input_ids, model=self.model)

        result = self.model(inputs_embeds=input_embeddings, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)

        del input_embeddings

        logits = self.get_logits(result, input_ids, so=so)

        if self.args.init_rel_subtype:
            type_loss = type_loss(self, result.hidden_states[-1], result.logits, labels, so, input_ids, attention_mask)
            self.log("EvalMore/type_loss", type_loss)

        loss = self.loss_fn(logits, labels)
        self.log("Eval/lmhead_loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}

    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        result = self.eval_fn(logits, labels)
        f1 = result['f1']
        self.log("Eval/f1", f1)
        self.log("EvalMore/recall", result['r'])
        self.log("EvalMore/precision", result['p'])

        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("best", self.best_f1, prog_bar=True, on_epoch=True)

    # Test
    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        if self.args.use_pre_prompt:
            input_ids, attention_mask, labels, so, sub_ids, sub_mask, obj_ids, obj_mask = batch
            input_embeddings = self.get_second_embeddings(batch)

        else:
            input_ids, attention_mask, labels, so = batch[:4] # (bsz, 256), (bsz, 256), (bsz,), (bsz, 4)
            input_embeddings = self.convert_ids_to_embeds(input_ids=input_ids, model=self.model)

        result = self.model(inputs_embeds=input_embeddings, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)

        del input_embeddings

        logits = self.get_logits(result, input_ids, so=so)

        if labels[0].item() != labels[-1].item():
            debug = 1

        if labels[0].item() == self.args.debug1:
            print("find")

        if not (torch.argmax(logits, dim=-1) == labels).all():
            idx = ((torch.argmax(logits, dim=-1) != labels) == True).nonzero().squeeze(-1)
            assert True

        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        result = self.eval_fn(logits, labels)
        f1 = result['f1']
        self.log("Charts/f1", f1)
        self.log("TestMore/recall", result['r'])
        self.log("TestMore/precision", result['p'])

        f1_by_relation = result["f1_by_relation"]
        recall_by_relation = result["recall_by_relation"]
        prec_by_relation = result["prec_by_relation"]
        gold_by_relation = result["gold_by_relation"]
        guess_by_relation = result["guess_by_relation"]
        correct_by_relation = result["correct_by_relation"]

        if self.args.wandb:

            f1_data = [[self.conv_id2rel(rel_id), count] for (rel_id, count) in enumerate(f1_by_relation)]
            r_data = [[self.conv_id2rel(rel_id), count] for (rel_id, count) in enumerate(recall_by_relation)]
            p_data = [[self.conv_id2rel(rel_id), count] for (rel_id, count) in enumerate(prec_by_relation)]

            count_data = [[
                            self.conv_id2rel(rel_id),
                            gold_by_relation[rel_id],
                            gold_by_relation[rel_id] / sum(gold_by_relation) * 100,
                            guess_by_relation[rel_id],
                            correct_by_relation[rel_id],
                            f1_by_relation[rel_id],
                            prec_by_relation[rel_id],
                            recall_by_relation[rel_id],
                        ] for rel_id in range(self.num_relation)]
            count_table = wandb.Table(data=count_data, columns=["rel names", "gold", "rate(%)", "guess", "correct", "f1", "precision", "recall"])
            wandb.log({"TestMore/count": count_table})

            f1_table = wandb.Table(data=f1_data, columns=["rel names", "counts"])
            r_table = wandb.Table(data=r_data, columns=["rel names", "counts"])
            p_table = wandb.Table(data=p_data, columns=["rel names", "counts"])

            wandb.log({"TestMore/f1_t": wandb.plot.bar(f1_table, "rel names", "counts", title="f1 by relations")})
            wandb.log({"TestMore/recall": wandb.plot.bar(r_table, "rel names", "counts", title="recall by relations")})
            wandb.log({"TestMore/precision": wandb.plot.bar(p_table, "rel names", "counts", title="precision by relations")})


    # Loss
    def get_logits(self, result, input_ids, ce_type="additional", so=None):
        if self.w_ce == 0:
            assert so is not None
            return self.get_rels(result.hidden_states[-1], so)

        mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        bs = input_ids.shape[0]
        assert mask_idx.shape[0] == bs # * "only one mask in sequence!"

        mask_output = result.logits[torch.arange(bs), mask_idx]

        if ce_type == "additional":
            x = mask_output[:,self.word2label]
        elif ce_type == "vocab":
            x = self.ce_classifier(mask_output)

        return x

    def get_rels(self, hidden_state, so):
        bsz = len(so)

        cls_tag = hidden_state[:, 0]  # [4, 1024]

        if self.args.cls_type == "tag":
            sub_ent = torch.stack([hidden_state[i, so[i][0]-1] for i in range(bsz)]).cuda()   # [sub] tag before subject in prompt template
            obj_ent = torch.stack([hidden_state[i, so[i][2]-1] for i in range(bsz)]).cuda()   # [obj] tag before subject in prompt template
        elif self.args.cls_type == "entity":
            sub_ent, obj_ent = self.get_entity_embeddings(hidden_state, so, bsz)

        input_state = torch.cat([sub_ent, cls_tag, obj_ent], dim=-1)  # [4, 1536]
        output_state = self.rel_classifier(input_state)  # [4, 19]

        return output_state # [bsz, rel_nums]

    # Optimizers
    def configure_optimizers(self):

        no_decay_param = ["bias", "LayerNorm.weight", "layer_norm.weight"]

        linear_lr = self.args.linear_lr if self.args.linear_lr else self.lr
        prompt_lr = self.args.prompt_lr if self.args.prompt_lr else linear_lr

        def get_params(name, lr, additional=[]):
            filter = lambda n, p, c: (name in n or not name) \
                and not any(nd in n for nd in additional) \
                and p.requires_grad \
                and (c ^ any(nd in n for nd in no_decay_param))

            if len([n for n, p in self.named_parameters() if filter(n, p, True)]) == 0:
                return []

            print("\nParameters (execpt model.roberta.encoder.layer) LR: {}\n{}".format(
                lr,
                "- " + "\n- ".join([f"{n} {list(p.size())}" for n, p in self.named_parameters() if filter(n, p, True) and "roberta.encoder.layer" not in n])))

            return [{
                "params": [p for n, p in self.named_parameters() if filter(n, p, True)],
                'lr': lr,
                "weight_decay": self.args.weight_decay
            }, {
                "params": [p for n, p in self.named_parameters() if filter(n, p, False)],
                'lr': lr,
                "weight_decay": 0
            }]

        optimizer_group_parameters = get_params("model", self.lr, ["prefix_encoder", "key_projection", "value_projection"])

        if self.args.use_prompt_projection:
            optimizer_group_parameters.extend(get_params("prompt_projection", prompt_lr))

        if self.args.use_pre_prompt:
            optimizer_group_parameters.extend(get_params("pre_encoder", prompt_lr))

        if self.args.model_class == "RobertaForKnow":
            optimizer_group_parameters.extend(get_params("prefix_encoder", prompt_lr))

        if self.args.debug3 == "use_key_values_projection":
            optimizer_group_parameters.extend(get_params("key_projection", prompt_lr))
            optimizer_group_parameters.extend(get_params("value_projection", prompt_lr))

        optimizer_group_parameters.extend(get_params("", linear_lr, ["model", "prompt_projection", "pre_encoder", "prefix_encoder", "key_projection", "value_projection"]))

        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps)
        # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }


    def type_loss(self, hidden_state, logits, labels, so, input_ids, attention_mask):
        bsz = hidden_state.shape[0]
        weights = self.model.get_output_embeddings().weight

        mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        # current just use embeddings with any projection.
        prefix_type = torch.stack([torch.mean(weights[self.rel_subtype[labels[i].item()][0]], dim=0) for i in range(bsz)])
        suffix_type = torch.stack([torch.mean(weights[self.rel_subtype[labels[i].item()][1]], dim=0) for i in range(bsz)])

        # default use mask tags
        if self.args.init_rel_subtype == "use_entity_tags":
            mark_prefix = torch.stack([hidden_state[i, so[i][0]] for i in range(bsz)]).cuda()     # [sub] tag before subject in prompt template
            mark_suffix = torch.stack([hidden_state[i, so[i][2]] for i in range(bsz)]).cuda()     # [obj] tag before subject in prompt template

        elif self.args.init_rel_subtype == "use_logits_top":
            prefix_top10 = torch.topk(self.model.lm_head(prefix_type), k=10, dim=-1)
            suffix_top10 = torch.topk(self.model.lm_head(suffix_type), k=10, dim=-1)

            mark_prefix_logits = logits[torch.arange(bsz), mask_idx-1]
            mark_suffix_logits = logits[torch.arange(bsz), mask_idx+2]
            mark_prefix = mark_prefix_logits[torch.arange(bsz), prefix_top10.indices.T].T
            mark_suffix = mark_suffix_logits[torch.arange(bsz), suffix_top10.indices.T].T

            prefix_type = prefix_top10.values
            suffix_type = suffix_top10.values

        else:
            mark_prefix = hidden_state[torch.arange(bsz), mask_idx-1]   # [m] tag
            mark_suffix = hidden_state[torch.arange(bsz), mask_idx+2]   # [/m] tag

        if self.args.use_type_projection and self.args.init_rel_subtype == "use_logits_top":
            raise NameError("can not use `use_type_projection` and `init_rel_subtype` at the same time.")

        if self.args.use_type_projection and self.args.init_rel_subtype != "use_logits_top":
            prefix_type = self.type_projection(prefix_type)
            suffix_type = self.type_projection(suffix_type)
            mark_prefix = self.type_projection(mark_prefix)
            mark_suffix = self.type_projection(mark_suffix)

        # similarity func
        similarity_funcs = {
            "cosine": lambda a, b: 1 - torch.cosine_similarity(a, b, dim=-1),
            "l2": lambda a, b: torch.norm(a-b, p=2, dim=-1)
        }

        similarity_func = similarity_funcs[self.args.similarity_func]
        loss = similarity_func(mark_prefix, prefix_type).sum() + similarity_func(mark_suffix, suffix_type).sum()

        return loss / bsz


    # Loss - Cloze Loss
    def cloze_loss(self, logits, masked_lm_positions, masked_lm_labels, input_ids):
        """
        Inputs:
            - logits                 => [batch_size, seq_len, vocab_size]
            - masked_lm_positions    => [batch_size, cloze_size]
            - masked_lm_labels       => [batch_size, cloze_size]
            - input_ids              => [batch_size, seq_len]

        Targets:
            [batch_size, seq_len, vocab_size] => [batch_size, cloze_size, vocab_size] => [batch_size, cloze_size]
        """
        bs, seq_len, vocab_size = logits.shape
        ont_hot_label = F.one_hot(masked_lm_labels, num_classes=vocab_size)  # => [batch_size, cloze_size, vocab_size]

        log_probs = F.log_softmax(logits, dim=-1)  # => [batch_size, seq_len, vocab_size]

        loss = 0
        for i in range(bs):
            temp = log_probs[i, masked_lm_positions[i], :]  * ont_hot_label[i]  # => [cloze_size, vocab_size]
            temp = torch.sum(temp, dim=-1).squeeze(-1)  # => [cloze_size]
            temp = torch.sum(temp, dim=-1).squeeze(-1)  # => scalar
            loss += temp / bs

        return -1 * loss

    # Loss - KE Loss
    def ke_loss(self, hidden_state, labels, so, input_ids, attention_mask):
        """ Implicit Structured Constraints """
        bsz = hidden_state.shape[0]

        assert len(self.args.ke_type.split(":")) == 2
        ke_type, calc_type = self.args.ke_type.split(":")

        if ke_type == "origin":
            pos_sub_embdeeings, pos_obj_embdeeings, neg_sub_embeddings, neg_obj_embeddings = self.get_entity_embeddings(hidden_state, so, bsz, pos=True, neg=True)
        elif ke_type == "tag":
            pos_sub_embdeeings = torch.stack([hidden_state[i, so[i][0]-1] for i in range(bsz)]).cuda()
            pos_obj_embdeeings = torch.stack([hidden_state[i, so[i][2]-1] for i in range(bsz)]).cuda()
            neg_sub_embeddings, neg_obj_embeddings = self.get_entity_embeddings(hidden_state, so, bsz, pos=False, neg=True)

        # trick , the relation ids is concated,

        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        mask_output = hidden_state[torch.arange(bsz), mask_idx]
        mask_relation_embedding = mask_output
        real_relation_embedding = self.model.get_output_embeddings().weight[labels+self.label_st_id]

        log_sigmoid = torch.nn.LogSigmoid()

        if self.args.use_emb_projection:
            pos_sub_embdeeings = self.projection_sub(pos_sub_embdeeings)
            pos_obj_embdeeings = self.projection_obj(pos_obj_embdeeings)
            neg_sub_embeddings = self.projection_sub(neg_sub_embeddings)
            neg_obj_embeddings = self.projection_obj(neg_obj_embeddings)
            mask_relation_embedding = self.projection_rel(mask_relation_embedding)
            real_relation_embedding = self.projection_rel(real_relation_embedding)

        if calc_type == "default":
            d_1 = torch.norm(pos_sub_embdeeings + mask_relation_embedding - pos_obj_embdeeings, p=2)
            d_2 = torch.norm(neg_sub_embeddings + real_relation_embedding - neg_obj_embeddings, p=2)

        elif calc_type == "mask":
            d_1 = torch.norm(pos_sub_embdeeings + mask_relation_embedding - pos_obj_embdeeings, p=2)
            d_2 = torch.norm(neg_sub_embeddings + mask_relation_embedding - neg_obj_embeddings, p=2)

        elif calc_type == "mask_linear":
            d_1 = torch.norm(self.linear_transe(torch.cat([pos_sub_embdeeings, mask_relation_embedding, pos_obj_embdeeings], dim=-1)), p=2)
            d_2 = torch.norm(self.linear_transe(torch.cat([neg_sub_embeddings, mask_relation_embedding, neg_obj_embeddings], dim=-1)), p=2)

        elif calc_type == "default_linear":
            d_1 = torch.norm(self.linear_transe(torch.cat([pos_sub_embdeeings, mask_relation_embedding, pos_obj_embdeeings], dim=-1)), p=2)
            d_2 = torch.norm(self.linear_transe(torch.cat([neg_sub_embeddings, real_relation_embedding, neg_obj_embeddings], dim=-1)), p=2)
        else:
            raise NameError("Unable to recogniz calc_type: '{}' or 'mask_reverse' can be used only if the corpus_type equals to 'semeval'".format(calc_type))

        loss = -1. * log_sigmoid(self.args.t_gamma - d_1) - log_sigmoid(d_2 - self.args.t_gamma)

        return loss

    def get_entity_embeddings(self, hidden_state, so, bsz, pos=True, neg=False):
        pos_sub_embdeeings = []
        pos_obj_embdeeings = []

        for i in range(bsz):
            sub_e = torch.mean(hidden_state[i, so[i][0]:so[i][1]], dim=0)  # include "space"
            obj_e = torch.mean(hidden_state[i, so[i][2]:so[i][3]], dim=0)  # include "space"
            pos_sub_embdeeings.append(sub_e)
            pos_obj_embdeeings.append(obj_e)

        pos_sub_embdeeings = torch.stack(pos_sub_embdeeings)
        pos_obj_embdeeings = torch.stack(pos_obj_embdeeings)

        neg_sub_embeddings = []
        neg_obj_embeddings = []
        for i in range(bsz):
            st_sub = random.randint(1, hidden_state[i].shape[0] - 6)
            st_obj = random.randint(1, hidden_state[i].shape[0] - 6)
            neg_sub_e = torch.mean(hidden_state[i, st_sub:st_sub + random.randint(1, 5)], dim=0)
            neg_obj_e = torch.mean(hidden_state[i, st_obj:st_obj + random.randint(1, 5)], dim=0)

            neg_sub_embeddings.append(neg_sub_e)
            neg_obj_embeddings.append(neg_obj_e)

        neg_sub_embeddings = torch.stack(neg_sub_embeddings)
        neg_obj_embeddings = torch.stack(neg_obj_embeddings)

        if pos and neg:
            return pos_sub_embdeeings, pos_obj_embdeeings, neg_sub_embeddings, neg_obj_embeddings
        elif pos:
            return pos_sub_embdeeings, pos_obj_embdeeings
        elif neg:
            return neg_sub_embeddings, neg_obj_embeddings

    def first_stage_stru(self, pos_sub_embdeeings, pos_obj_embdeeings, labels):

        bsz = len(labels)

        if self.args.use_pre_prompt == "long" or self.args.use_pre_prompt == "long_mask":
            pos_sub_embdeeings = pos_sub_embdeeings[:, 0]
            pos_obj_embdeeings = pos_obj_embdeeings[:, 0]

        fake_labels = labels.clone()
        while (fake_labels == labels).any():
            fake_labels = torch.randint(self.num_relation, [bsz], device="cuda")

        real_relation_embedding = self.model.get_output_embeddings().weight[labels+self.label_st_id]
        nega_relation_embedding = self.model.get_output_embeddings().weight[fake_labels+self.label_st_id]

        pos_out = self.stru_classifier(torch.cat([pos_sub_embdeeings, real_relation_embedding, pos_obj_embdeeings], dim=-1))
        neg_put = self.stru_classifier(torch.cat([pos_sub_embdeeings, nega_relation_embedding, pos_obj_embdeeings], dim=-1))

        loss = self.loss_fn(pos_out, torch.ones(bsz, device="cuda", dtype=torch.long)) + self.loss_fn(neg_put, torch.zeros(bsz, device="cuda", dtype=torch.long))
        return loss / bsz

    def pre_ke(self, input_ids, hidden_state, pos_sub_embdeeings, pos_obj_embdeeings, labels, so):
        # TODO: 这里是不是应该使用 s,o,r 与 s,o,r' 来着

        bsz = hidden_state.shape[0]

        if self.args.use_pre_prompt == "long" or self.args.use_pre_prompt == "long_mask":
            pos_sub_embdeeings = pos_sub_embdeeings[:, 0]
            pos_obj_embdeeings = pos_obj_embdeeings[:, 0]

        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        mask_output = hidden_state[torch.arange(bsz), mask_idx]
        mask_relation_embedding = mask_output

        pre_ke = "neg_rel" # for debug
        if pre_ke == "default":
            real_relation_embedding = self.model.get_output_embeddings().weight[labels+self.label_st_id]

            neg_sub_embeddings, neg_obj_embeddings = self.get_entity_embeddings(hidden_state, so, bsz, False, True)

            d_1 = torch.norm(self.linear_transe(torch.cat([pos_sub_embdeeings, mask_relation_embedding, pos_obj_embdeeings], dim=-1)), p=2) / bsz
            d_2 = torch.norm(self.linear_transe(torch.cat([neg_sub_embeddings, real_relation_embedding, neg_obj_embeddings], dim=-1)), p=2) / bsz

        elif pre_ke == "neg_rel":
            fake_labels = labels.clone()
            while (fake_labels == labels).any():
                fake_labels = torch.randint(self.num_relation, [bsz], device="cuda")

            nega_relation_embedding = self.model.get_output_embeddings().weight[fake_labels+self.label_st_id]
            d_1 = torch.norm(self.linear_transe(torch.cat([pos_sub_embdeeings, mask_relation_embedding, pos_obj_embdeeings], dim=-1)), p=2) / bsz
            d_2 = torch.norm(self.linear_transe(torch.cat([pos_sub_embdeeings, nega_relation_embedding, pos_obj_embdeeings], dim=-1)), p=2) / bsz

        log_sigmoid = torch.nn.LogSigmoid()
        loss = -1. * log_sigmoid(self.args.t_gamma - d_1) - log_sigmoid(d_2 - self.args.t_gamma)

        return loss

    # Utils
    def get_reverse_rel(self, rel_id):
        if "semeval" in self.args.data_dir:
            reverse_rel_dict = torch.tensor([0, 12, 17, 9, 11, 18, 16, 13, 14, 3, 15, 4, 1, 7, 8, 10, 6, 2, 5])
            return reverse_rel_dict[rel_id]
        else:
            raise NameError("Can not get reverse relations in '{}'".format(self.args.data_dir))

    def conv_id2rel(self, id_before_conv):
        if id_before_conv == 0:
            return self.id2rel[self.Na_num]
        elif id_before_conv <= self.Na_num:
            return self.id2rel[id_before_conv - 1]
        elif id_before_conv > self.Na_num:
            return self.id2rel[id_before_conv]
        else:
            raise NameError("Something WRONG Here")