import torch

from transformers import  BartForConditionalGeneration, T5ForConditionalGeneration # RobertaForMaskedLM
from models.roberta.modeling_roberta import RobertaForMaskedLM, RobertaModel
"""
    # TAG: RoBERTa

    [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta)

    Abstract :

    Language model pretraining has led to significant performance gains
    but careful comparison between different approaches is challenging.
    Training is computationally expensive, often done on private datasets
    of different sizes, and, as we will show, hyperparameter choices have
    significant impact on the final results. We present a replication
    study of BERT pretraining (Devlin et al., 2019) that carefully measures
    the impact of many key hyperparameters and training data size. We find
    that BERT was significantly undertrained, and can match or exceed the
    performance of every model published after it. Our best model achieves
    state-of-the-art results on GLUE, RACE and SQuAD. These results highlight
    the importance of previously overlooked design choices, and raise
    questions about the source of recently reported improvements. We release
    our models and code.
"""


from models.roberta_for_prefix import RobertaForPrefix
from models.roberta_for_knowledge import RobertaForKnow

class RobertaForPrompt(RobertaForMaskedLM):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--use_prompt", type=bool, default=True, help="Whether to use prompt in the dataset.")
        parser.add_argument("--init_answer_words", type=int, default=1, )
        parser.add_argument("--init_type_words", type=int, default=1, )
        parser.add_argument("--init_answer_words_by", type=str, default="whole_word", )
        parser.add_argument("--use_template_words", type=int, default=1, )
        return parser

class BartRE(BartForConditionalGeneration):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--use_prompt", type=bool, default=True, help="Whether to use prompt in the dataset.")
        return parser

class T5RE(T5ForConditionalGeneration):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--use_prompt", type=bool, default=True, help="Whether to use prompt in the dataset.")
        return parser

