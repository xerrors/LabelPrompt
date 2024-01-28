from data.funcs import sample_few_shot_data
from .base_data_module import BaseDataModule
from .processor import get_dataset, processors
from transformers import AutoTokenizer

from dataclasses import dataclass
from torch.utils.data import DataLoader

import random
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers.file_utils import PaddingStrategy
from transformers.models.bert import BertTokenizer, BertTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
            sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
            maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
            different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features



class DIALOGUE(BaseDataModule):
    def __init__(self, args, model=None) -> None:
        super().__init__(args)
        self.processor = processors[self.args.task_name](self.args.data_dir, self.args.use_prompt)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)

        self.num_labels = len(self.processor.get_labels())

        class_list = [f"[C{i}]" for i in range(1, self.num_labels+1)]
        unused_list = [f"[unused{i}]" for i in range(1,50)]
        speaker_list = [f"[speaker{i}]" for i in range(1,50)]
        so_list = ["[sub]", "[obj]"]
        sp_mark = ["[s]", "[/s]", "[o]", "[/o]", "[m]", "[/m]", "[cloze]"]

        additional_special_tokens = class_list + unused_list + speaker_list + so_list + sp_mark

        if args.prefix_length > 0:
            additional_special_tokens.extend([f"[{i}]" for i in range(args.prefix_length)])

        self.tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})


    def setup(self, stage=None):
        self.data_train = get_dataset("train", self.args, self.tokenizer, self.processor)
        self.data_val = get_dataset("dev", self.args, self.tokenizer, self.processor)
        self.data_test = get_dataset("test", self.args, self.tokenizer, self.processor)

    def prepare_data(self):
        pass


    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--task_name", type=str, default="normal", help="[normal, reloss, ptune]")
        parser.add_argument("--model_name_or_path", type=str, default="/home/zwj/nlp/Bert/bert-base-uncased", help="Number of examples to operate on per forward step.")
        parser.add_argument("--max_seq_length", type=int, default=512, help="Number of examples to operate on per forward step.")
        parser.add_argument("--ptune_k", type=int, default=7, help="number of unused tokens in prompt")
        return parser

# TAG: WIKI80 add special tokens
class WIKI80(BaseDataModule):
    def __init__(self, args, model=None) -> None:
        super().__init__(args)
        self.args = args

        # check if it was few-shot.
        if self.args.few_shot:
            self.args.data_dir = sample_few_shot_data(self.args.data_dir, self.args.seed, self.args.few_shot)

        self.processor = processors[self.args.task_name](self.args.data_dir, self.args.use_prompt)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)

        use_gpt = "gpt" in args.model_name_or_path

        self.num_labels = len(self.processor.get_labels())

        so_list = ["[sub]", "[obj]"]
        prompt_tokens = [f"[T{i}]" for i in range(1, args.prefix_length if args.prefix_length > 0 else 6)]  # GPT need T1-T6 here
        entity_list = ["[object_start]", "[object_end]", "[subject_start]", "[subject_end]"]
        sp_mark = ["[s]", "[/s]", "[o]", "[/o]", "[m]", "[/m]", "[cloze]", "(e1,e2)", "(e2,e1)"]
        additional_special_tokens = so_list + prompt_tokens + entity_list + sp_mark

        if args.w_ent or args.w_ent_two or args.w_type:
            sub_type_mark = [f"[S{i}]" for i in range(len(args.sub_types))]
            obj_type_mark = [f"[O{i}]" for i in range(len(args.obj_types))]
            additional_special_tokens = additional_special_tokens + sub_type_mark + obj_type_mark

        if not args.use_self_mlm:
            additional_special_tokens += [f"[C{i}]" for i in range(1, self.num_labels+1)]

        self.tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})

        if use_gpt:
            self.tokenizer.add_special_tokens({'cls_token': "[CLS]"})
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})


    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.data_train = get_dataset("train", self.args, self.tokenizer, self.processor)
            self.data_val = get_dataset("dev", self.args, self.tokenizer, self.processor)
        if stage == "test" or stage is None or stage == "predict":
            self.data_test = get_dataset("test", self.args, self.tokenizer, self.processor)

    def prepare_data(self):
        pass

    def get_tokenizer(self):
        return self.tokenizer

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--task_name", type=str, default="normal", help="[normal, reloss, ptune]")
        parser.add_argument("--model_name_or_path", type=str, default="/home/xx/bert-base-uncased", help="Number of examples to operate on per forward step.")
        parser.add_argument("--max_seq_length", type=int, default=512, help="Number of examples to operate on per forward step.")
        parser.add_argument("--ptune_k", type=int, default=7, help="number of unused tokens in prompt")
        return parser

class SST2(BaseDataModule):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.processor = processors[self.args.task_name](self.args.data_dir, self.args.use_prompt)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)

        labels = self.processor.get_labels()
        self.num_labels = len(labels)

        class_list = [f"[C{i}]" for i in range(1, self.num_labels+1)]

        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': class_list})

        if args.CT_CL:
            prompt_tokens = [f"[T{i}]" for i in range(1,6)]
            self.tokenizer.add_special_tokens({'additional_special_tokens': prompt_tokens})


        self.data_train = get_dataset("train", self.args, self.tokenizer, self.processor)
        self.num_training_steps = len(self.data_train) // self.batch_size // self.args.accumulate_grad_batches * self.args.max_epochs



    def setup(self, stage=None):
        self.data_val = get_dataset("dev", self.args, self.tokenizer, self.processor)
        self.data_test = get_dataset("test", self.args, self.tokenizer, self.processor)


    def prepare_data(self):
        pass

    def get_tokenizer(self):
        return self.tokenizer

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--task_name", type=str, default="normal", help="[normal, reloss, ptune]")
        parser.add_argument("--model_name_or_path", type=str, default="/home/xx/bert-base-uncased", help="Number of examples to operate on per forward step.")
        parser.add_argument("--max_seq_length", type=int, default=512, help="Number of examples to operate on per forward step.")
        parser.add_argument("--ptune_k", type=int, default=7, help="number of unused tokens in prompt")
        return parser


class BartREDataset(BaseDataModule):
    def __init__(self, args, model=None) -> None:
        super().__init__(args)
        self.processor = processors[self.args.task_name](self.args.data_dir, self.args.use_prompt)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)

        rel2id = self.processor.get_labels()
        self.num_labels = len(rel2id)
        entity_list = ["[object_start]", "[object_end]", "[subject_start]", "[subject_end]"]
        class_list = [f"[C{i}]" for i in range(1, self.num_labels+1)]

        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': entity_list})
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': class_list})
        so_list = ["[sub]", "[obj]"]
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': so_list})

        prompt_tokens = [f"[T{i}]" for i in range(1,6)]
        self.tokenizer.add_special_tokens({'additional_special_tokens': prompt_tokens})
        if "t5" in self.args.model_name_or_path:
            self.tokenizer.add_special_tokens({'mask_token': "<mask>"})

        self.collate_fn = DataCollatorForSeq2Seq(self.tokenizer,
            model=model,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_to_multiple_of=8 if self.args.fp16 else None,
            padding="longest",
            max_length=self.args.max_seq_length
        )





    def setup(self, stage=None):
        self.data_train = get_dataset("train", self.args, self.tokenizer, self.processor)
        self.data_val = get_dataset("dev", self.args, self.tokenizer, self.processor)
        self.data_test = get_dataset("test", self.args, self.tokenizer, self.processor)


    def prepare_data(self):
        pass

    def get_tokenizer(self):
        return self.tokenizer

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--task_name", type=str, default="normal", help="[normal, reloss, ptune]")
        parser.add_argument("--model_name_or_path", type=str, default="/home/xx/bert-base-uncased", help="Number of examples to operate on per forward step.")
        parser.add_argument("--max_seq_length", type=int, default=512, help="Number of examples to operate on per forward step.")
        parser.add_argument("--ptune_k", type=int, default=7, help="number of unused tokens in prompt")
        return parser



    def train_dataloader(self):
        dataloader =  DataLoader(self.data_train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

        return dataloader

    def val_dataloader(self):
        return DataLoader(self.data_val, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.data_test, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)


class SST2(BaseDataModule):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.processor = processors[self.args.task_name](self.args.data_dir, self.args.use_prompt)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)

        labels = self.processor.get_labels()
        self.num_labels = len(labels)

        class_list = [f"[C{i}]" for i in range(1, self.num_labels+1)]

        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': class_list})

        if args.CT_CL:
            prompt_tokens = [f"[T{i}]" for i in range(1,6)]
            self.tokenizer.add_special_tokens({'additional_special_tokens': prompt_tokens})


        self.data_train = get_dataset("train", self.args, self.tokenizer, self.processor)
        self.num_training_steps = len(self.data_train) // self.batch_size // self.args.accumulate_grad_batches * self.args.max_epochs



    def setup(self, stage=None):
        self.data_val = get_dataset("dev", self.args, self.tokenizer, self.processor)
        self.data_test = get_dataset("test", self.args, self.tokenizer, self.processor)


    def prepare_data(self):
        pass

    def get_tokenizer(self):
        return self.tokenizer

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--task_name", type=str, default="normal", help="[normal, reloss, ptune]")
        parser.add_argument("--model_name_or_path", type=str, default="/home/xx/bert-base-uncased", help="Number of examples to operate on per forward step.")
        parser.add_argument("--max_seq_length", type=int, default=512, help="Number of examples to operate on per forward step.")
        parser.add_argument("--ptune_k", type=int, default=7, help="number of unused tokens in prompt")
        return parser