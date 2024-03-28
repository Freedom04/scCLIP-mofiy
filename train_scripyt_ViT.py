import sys
sys.path.append(".")
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
import os
import argparse
import torch
from pathlib import Path

from datasets.data import MultiOmicsModule
from config import Config
from models.model import scCLIPModel
from models.vit import ViTConfig
from models.clip import CLIPModel


# HOME = Path.home()
print("Start", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="clip")
    parser.add_argument("--seed", default=42, type=int)
    checkpoint = "results/AD.h5mu/1_False_30000_0.00015_v1.0/lightning_logs/v1.0/checkpoints/epoch161step30000.ckpt"
    parser.add_argument("--checkpoint", type=str, default=checkpoint)
    # DataModule
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="data/multiome/human_brain_3k.h5mu")
    parser.add_argument("--backed", action="store_true", default=False)
    parser.add_argument("--split", default=0.9)
    parser.add_argument("--n_top_genes", type=int, default=None)
    parser.add_argument("--binary", action=argparse.BooleanOptionalAction, default=True)
    # parser.add_argument("--binary", action="store_true", default=True)
    parser.add_argument("--dist", type=str, default=None)
    parser.add_argument("--mask", type=float, default=None)
    parser.add_argument("--peak_dist", type=int, default=10_000)
    parser.add_argument(
        "--experiment", action="store_true", default=True
    )
    parser.add_argument("--mod", type=str, default="multiome")
    parser.add_argument("--atac", type=str, default=None)
    parser.add_argument("--rna", type=str, default=None)

    # Module
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--ffn_dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument( 
        "--use_imputed", action="store_true", default=True
    )
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument(
        "--requires_grad", action="store_true", default=True
    )
    parser.add_argument(
        "--normalize", action="store_true", default=True
    )
    parser.add_argument("--version", type=str, default="v1.0")

    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--fast_dev_run", action="store_true", default=False)

    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--no_scalex", action="store_true", default=False)
    parser.add_argument(
        "--use_val", action="store_true", default=True
    )
    parser.add_argument("--test_data", type=str, default="ADRD")
    parser.add_argument("--cell_type", type=str, default="cell_type")
    parser.add_argument(
        "--use_seq", action="store_true", default=False
    )
    parser.add_argument("--logit_scale", type=float, default=1)
    parser.add_argument("--num_patches", type=int, default=128)
    parser.add_argument(
        "--early_stop", action="store_true", default=False
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    if args.checkpoint is None:
        dm = MultiOmicsModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        args.peaks = dm.dataset.mdata.mod["atac"].var
        args.genes = dm.dataset.mdata.mod["rna"].var

        model_config = {
            "hidden_size": 256,
            "num_hidden_layers": 6,
            "num_attention_heads": 8,
            "intermediate_size": 512,
            "decoder_num_attention_heads": 8,
            "decoder_hidden_size": 256,
            "decoder_num_hidden_layers": 6,
            "decoder_intermediate_size": 512,
        }
        
        model_config_large = {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "decoder_num_attention_heads": 16,
            "decoder_hidden_size": 1024,
            "decoder_num_hidden_layers": 16,
            "decoder_intermediate_size": 4096,
        }

        atac_config = ViTConfig(
            **{
                "modality": "atac",
                "num_patches": args.num_patches,
                "feature_size": dm.dataset.mdata.mod["atac"].shape[1],
                "attention_probs_dropout_prob": args.dropout,
                "hidden_dropout_prob": args.dropout,
                **model_config,
            }
        )

        rna_config = ViTConfig(
            **{
                "modality": "rna",
                "num_patches": args.num_patches,
                "attention_probs_dropout_prob": args.dropout,
                "hidden_dropout_prob": args.dropout,
                "feature_size": dm.dataset.mdata.mod["rna"].shape[1],
                **model_config,
            }
        )

        model = CLIPModel(
            config=args,
            atac_config=atac_config,
            rna_config=rna_config
        )

        # out_dir
        if args.experiment:
            args.default_root_dir = f"results/{os.path.basename(args.data_dir)}/{args.logit_scale}_{args.requires_grad}_{args.max_steps}_{args.lr}_{args.version}"
        else:
            args.default_root_dir = (
                f"results/{os.path.basename(args.data_dir)}/{args.logit_scale}_{args.max_steps}"
            )
        # os.makedirs(args.default_root_dir, exist_ok=True)
        print("default_root_dir:", args.default_root_dir, flush=True)

        # trainer
        logger = TensorBoardLogger(
            save_dir=args.default_root_dir, default_hp_metric=False, version="v1.0"
        )

        trainer = Trainer(
            # callbacks=callbacks,
            accelerator="gpu",
            devices=[1, 2],
            strategy="dp",
            gradient_clip_val=5,
            num_sanity_val_steps=0,
            logger=logger,
            max_steps=args.max_steps,
            fast_dev_run=args.fast_dev_run, 
            # log_every_n_steps=1,
        )

        # fit
        trainer.fit(model, dm)

    else:
        model = CLIPModel.load_from_checkpoint(args.checkpoint)
        print("normalize", args.normalize, flush=True)
        model.config.normalize = args.normalize
        args.default_root_dir = args.checkpoint.split("lightning_logs/")[0]
        
        dm = MultiOmicsModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            # n_top_peaks=model.config.peaks,
            # n_top_genes=model.config.genes.index,
            # binary=model.config.binary,
        )

    if not args.fast_dev_run:
        out_dir = os.path.join(args.default_root_dir, os.path.basename(args.data_dir))
        os.makedirs(out_dir, exist_ok=True)

        # if args.mod == "multiome":
        #     if args.data_dir == model.config.data_dir:
        #         dataloader = dm.dataset
        #     else:
        #         dataloader = dm.dataset

        model.get_batch_features(dm, out_dir=out_dir)