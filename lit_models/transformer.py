from argparse import ArgumentParser
from json import decoder
from logging import debug
import pytorch_lightning as pl
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
# Hide lines below until Lab 5
import wandb
import numpy as np
# Hide lines above until Lab 5

from .base import BaseLitModel
from .util import f1_eval, compute_f1, acc, f1_score
from .bert_lit_model import BertLitModel, multilabel_categorical_crossentropy
from transformers.optimization import get_linear_schedule_with_warmup

from functools import partial

import random

def mask_hook(grad_input, st, ed):
    mask = torch.zeros((grad_input.shape[0], 1)).type_as(grad_input)
    mask[st: ed] += 1.0  # 只优化id为1～8的token
    # for the speaker unused token12
    mask[1:3] += 1.0
    return grad_input * mask


class TransformerLitModelTwoSteps(BertLitModel):
    def configure_optimizers(self):
        no_decay_param = ["bais", "LayerNorm.weight"]
        optimizer_group_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]
        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.args.lr_2, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }



class DialogueLitModel(BertLitModel):

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, token_type_ids , labels = batch
        result = self.model(input_ids, attention_mask, token_type_ids, return_dict=True, output_hidden_states=True)
        logits = result.logits
        logits = self.pvp(logits, input_ids)
        loss = self.loss_fn(logits, labels)
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, token_type_ids , labels = batch
        logits = self.model(input_ids, attention_mask, token_type_ids, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        loss = self.loss_fn(logits, labels)
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}

    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("best", self.best_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, token_type_ids , labels = batch
        logits = self.model(input_ids, attention_mask, token_type_ids, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Charts/f1", f1)



    @staticmethod
    def add_to_argparse(parser):
        BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--w_ke", type=float, default=0.0, help="")
        return parser

    def pvp(self, logits, input_ids):
        # convert the [batch_size, seq_len, vocab_size] => [batch_size, num_labels]
        #! hard coded
        _, mask_idx = (input_ids == 103).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        assert mask_idx.shape[0] == bs, "only one mask in sequence!"
        final_output = mask_output[:,self.word2label]

        return final_output

def decode(tokenizer, output_ids):
    return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in output_ids]

class GPTLitModel(BaseLitModel):
    def __init__(self, model, args , data_config):
        super().__init__(model, args)
        # self.num_training_steps = data_config["num_training_steps"]
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = multilabel_categorical_crossentropy
        self.best_f1 = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, cls_idx , labels = batch
        logits = self.model(input_ids, attention_mask=attention_mask, mc_token_ids=cls_idx)
        if not isinstance(logits, torch.Tensor):
            logits = logits.mc_logits

        loss = self.loss_fn(logits, labels)
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, cls_idx , labels = batch
        logits = self.model(input_ids, attention_mask=attention_mask, mc_token_ids=cls_idx)
        if not isinstance(logits, torch.Tensor):
            logits = logits.mc_logits
        loss = self.loss_fn(logits, labels)
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}

    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        # f1 = compute_f1(logits, labels)["f1"]
        f1 = f1_score(logits, labels)
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("best", self.best_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argumenT
        input_ids, attention_mask, cls_idx , labels = batch
        logits = self.model(input_ids, attention_mask=attention_mask, mc_token_ids=cls_idx)
        if not isinstance(logits, torch.Tensor):
            logits = logits.mc_logits
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = f1_score(logits, labels)
        # f1 = acc(logits, labels)
        self.log("Charts/f1", f1)

from models.trie import get_trie
class BartRELitModel(BaseLitModel):
    def __init__(self, model, args, tokenizer=None):
        super().__init__(model, args)
        self.best_f1 = 0
        self.first = True

        with open(f"{args.data_dir}/rel2id.json","r") as file:
            rel2id = json.load(file)

        Na_num = 0
        for k, v in rel2id.items():
            if k == "NA" or k == "no_relation" or k == "Other":
                Na_num = v
                break
        num_relation = len(rel2id)
        # init loss function
        self.loss_fn = multilabel_categorical_crossentropy if "dialogue" in args.data_dir else nn.CrossEntropyLoss()
        # ignore the no_relation class to compute the f1 score
        self.eval_fn = f1_eval if "dialogue" in args.data_dir else partial(f1_score, rel_num=num_relation, na_num=Na_num)

        self.tokenizer = tokenizer
        self.trie, self.rel2id = get_trie(args, tokenizer=tokenizer)

        self.decode = partial(decode, tokenizer=self.tokenizer)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        real_label  = batch.pop("label")
        loss = self.model(**batch).loss
        self.log("Train/loss", loss)
        return loss



    def validation_step(self, batch, batch_idx):
        real_label = batch.pop("label")
        labels = batch.pop("labels")
        batch.pop("decoder_input_ids")
        topk = 1
        outputs = self.model.generate(**batch,
            prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist()),
            num_beams=topk, num_return_sequences=topk,
            output_scores=True,
            min_length=0,
            max_length=32,
        ).cpu()
        # calculate the rank in the decoder output

        pad_id = self.tokenizer.pad_token_id
        outputs = self.decode(output_ids=outputs)
        labels = self.decode(output_ids=labels)

        preds = torch.tensor([self.rel2id[o] for o in outputs])
        true = real_label


        return {"eval_logits": preds.detach().cpu().numpy(), "eval_labels": true.detach().cpu().numpy()}



    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Eval/f1", f1)
        if f1 > self.best_f1 and not self.first:
            self.best_f1 = f1
        self.first = False
        self.log("best", self.best_f1, prog_bar=True, on_epoch=True)


    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        real_label = batch.pop("label")
        labels = batch.pop("labels")
        batch.pop("decoder_input_ids")
        topk = 1
        outputs = self.model.generate(**batch,
            prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist()),
            num_beams=topk, num_return_sequences=topk,
            output_scores=True,
            min_length=0,
            max_length=32,
        ).cpu()
        # calculate the rank in the decoder output

        pad_id = self.tokenizer.pad_token_id
        outputs = self.decode(output_ids=outputs)
        labels = self.decode(output_ids=labels)

        preds = torch.tensor([self.rel2id[o] for o in outputs])
        true = real_label


        return {"test_logits": preds.detach().cpu().numpy(), "test_labels": true.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Charts/f1", f1)


    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        optimizer_group_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]

        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }
