
import argparse
from utils import _import_class
import pytorch_lightning as pl

def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--gpu", type=str, default="not specified")
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--litmodel_class", type=str, default="TransformerLitModel")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data_class", type=str, default="WIKI80")
    parser.add_argument("--lr_2", type=float, default=3e-5)
    parser.add_argument("--model_class", type=str, default="bert.BertForSequenceClassification")
    parser.add_argument("--two_steps", default=False, action="store_true")
    parser.add_argument("--test_from_checkpoint", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--output_dirpath", type=str, default="output")

    parser.add_argument("--special_mark", type=str, default="debug")
    parser.add_argument("--prefix_length", type=int, default=40, )
    parser.add_argument("--use_cloze", type=int, default=0, help="")
    parser.add_argument("--w_cloze", type=float, default=0, help="")
    parser.add_argument("--ke_type", type=str, default="origin:default", help="")
    parser.add_argument("--use_emb_projection", default=False, action="store_true")
    parser.add_argument("--use_ascend_w_ke", type=str, default="")
    parser.add_argument("--label_smoothing", type=str, default="")

    parser.add_argument("--use_self_mlm", default=False, action="store_true")
    parser.add_argument("--auto_lr", default=False, action="store_true")

    parser.add_argument("--few_shot", type=float, default=0.0)

    parser.add_argument("--debug1", type=str, default="")
    parser.add_argument("--debug2", type=str, default="")
    parser.add_argument("--debug3", type=str, default="")
    parser.add_argument("--debug4", type=str, default="")

    parser.add_argument("--help", "-h", action="help")
    parser.add_argument("--note", type=str, default="None")
    parser.add_argument("--tag", type=str, default="simple")

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    if temp_args.test_from_checkpoint:
        return parser

    data_class = _import_class(f"data.{temp_args.data_class}")
    model_class = _import_class(f"models.{temp_args.model_class}")
    litmodel_class = _import_class(f"lit_models.{temp_args.litmodel_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    litmodel_class.add_to_argparse(lit_model_group)
    return parser