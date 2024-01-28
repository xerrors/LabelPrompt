"""Experiment-running framework."""

import os
import torch
import random
import numpy as np
import pytorch_lightning as pl
from args import _setup_parser
from utils import _import_class, get_gpu_by_user_input, handle_args_by_local_env, save_args
from transformers import AutoConfig
from pytorch_lightning.plugins import DDPPlugin
os.environ["TOKENIZERS_PARALLELISM"] = "false"


device = "cuda"

# _get_relation_embedding have been removed from here

def seed_every_thing(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed)


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    args = handle_args_by_local_env(args)

    seed_every_thing(args.seed)

    # 1. get data, model, lit_model
    data_class = _import_class(f"data.{args.data_class}")
    model_class = _import_class(f"models.{args.model_class}") # e.g. RobertaForPrompt
    litmodel_class = _import_class(f"lit_models.{args.litmodel_class}")

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    data = data_class(args, model=None)
    data.tokenizer.save_pretrained('test')
    data_config = data.get_data_config()

    config.args = args
    config.new_vocab_size = len(data.tokenizer)

    model = model_class.from_pretrained(args.model_name_or_path, config=config) # PLM
    model.resize_token_embeddings(config.new_vocab_size) # resize


    if args.use_pre_stage:
        if args.use_pre_stage == "share":
            pre_model = model
        elif args.use_pre_stage == "frozen":
            pre_config = AutoConfig.from_pretrained(args.model_name_or_path)
            pre_model = model_class.from_pretrained(args.model_name_or_path, config=pre_config)
            pre_model.resize_token_embeddings(config.new_vocab_size) # resize
            for param in pre_model.parameters():
                param.requires_grad = False
        elif args.use_pre_stage == "tune":
            pre_config = AutoConfig.from_pretrained(args.model_name_or_path)
            pre_model = model_class.from_pretrained(args.model_name_or_path, config=pre_config)
            pre_model.resize_token_embeddings(config.new_vocab_size) # resize

        lit_model = litmodel_class(args=args, model=model, tokenizer=data.tokenizer, pre_model=pre_model)

    else:
        lit_model = litmodel_class(args=args, model=model, tokenizer=data.tokenizer)

    # 2. get trainer
    logger = pl.loggers.TensorBoardLogger("training/logs")
    if args.wandb:
        if args.few_shot == 0:
            project_name = f"{args.dataset_name}_{args.tag}"
        elif args.few_shot < 1:
            project_name = f"{args.dataset_name}_R_{args.tag}"
        else:
            project_name = f"{args.dataset_name}_K_{args.tag}"
        logger = pl.loggers.WandbLogger(project=project_name, name=f"{args.special_mark}")
        logger.log_hyperparams(vars(args))

    early_callback = pl.callbacks.EarlyStopping(monitor="Eval/f1", mode="max", patience=5,check_on_train_epoch_end=False)
    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="Eval/f1", mode="max",
        filename='f1={Eval/f1:.4f}-epoch={epoch}',
        dirpath=args.model_dir,
        auto_insert_metric_name=False,
        save_weights_only=True
    )
    callbacks = [early_callback, model_checkpoint]

    # args.weights_summary = "full"  # Print full summary of the model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpu_count = torch.cuda.device_count()
    accelerator = "ddp" if gpu_count > 1 else None

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, default_root_dir="training/logs",
        gpus=gpu_count, accelerator=accelerator,
        plugins=DDPPlugin(find_unused_parameters=False) if gpu_count > 1 else None,
        log_every_n_steps=10,
        # val_check_interval=0.5,
        auto_lr_find=args.auto_lr,
    )


    # 3. Test or Train ?
    if args.test_from_checkpoint or args.ckpt_path:
        litmodel_class = _import_class(f"lit_models.{args.litmodel_class}")
        lit_model = litmodel_class.load_from_checkpoint(
            args.path if args.test_from_checkpoint else args.ckpt_path, args=args, model=model,
            tokenizer=data.tokenizer,
            pre_model=pre_model if args.use_pre_stage else None)
        trainer.test(lit_model, datamodule=data)

    elif args.auto_lr:
        trainer.tune(lit_model, datamodule=data) # with auto_lr_find=True
    else:
        trainer.fit(lit_model, datamodule=data)
        trainer.test(lit_model, datamodule=data)

        args.path = model_checkpoint.best_model_path
        print(f"best model save path {args.path}")

        save_args(args)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
