import sys
sys.path.append(".")
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
import os
import argparse
import torch
from pathlib import Path

from data import MultiOmicsModule
from config import Config
from models.model import scCLIPModel

# HOME = Path.home()
print("Start", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="clip")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--checkpoint", type=str, default=None)
    # DataModule
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="../scCLIP/data/multiome/human_brain_3k.h5mu")
    parser.add_argument("--backed", action="store_true", default=False)
    parser.add_argument("--split", default=0.9)
    parser.add_argument("--n_top_genes", type=int, default=None)
    # parser.add_argument("--binary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--binary", action="store_true", default=True)
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
        "--requires_grad", action="store_true", default=False
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

    # parser = Trainer.add_argparse_args(parser)

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

        model_config = Config(
            n_peaks=dm.dataset.mdata.mod['atac'].shape[1],
            n_genes=dm.dataset.mdata.mod['rna'].shape[1],
            
        )
        # atac_config = ViTConfig(
        #     **{
        #         "modality": "atac",
        #         "num_patches": args.num_patches,
        #         "feature_size": dm.dataset.mdata.mod["atac"].shape[1],
        #         "attention_probs_dropout_prob": args.dropout,
        #         "hidden_dropout_prob": args.dropout,
        #         **model_config,
        #     }
        # )

        # rna_config = ViTConfig(
        #     **{
        #         "modality": "rna",
        #         "num_patches": args.num_patches,
        #         "attention_probs_dropout_prob": args.dropout,
        #         "hidden_dropout_prob": args.dropout,
        #         "feature_size": dm.dataset.mdata.mod["rna"].shape[1],
        #         **model_config,
        #     }
        # )

        model = scCLIPModel(
            config=model_config
        )

        # out_dir
        if args.experiment:
            args.default_root_dir = f"results/{args.data_dir}/{args.logit_scale}_{args.requires_grad}_{args.max_steps}_{args.lr}_{args.version}"
        else:
            args.default_root_dir = (
                f"results/{args.data_dir}/{args.logit_scale}_{args.max_steps}"
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
            devices=[0],
            gradient_clip_val=5,
            num_sanity_val_steps=0,
            logger=logger,
            max_steps=args.max_steps,
            fast_dev_run=args.fast_dev_run,
            log_every_n_steps=50,
        )

        # fit
        trainer.fit(model, dm)

    # else:
    #     model = CLIPModel.load_from_checkpoint(args.checkpoint)
    #     print("normalize", args.normalize, flush=True)
    #     model.config.normalize = args.normalize
    #     args.default_root_dir = args.checkpoint.split("lightning_logs/")[0]

    #     dm = MixDataModule(
    #         data_dir=args.data_dir,
    #         modality=args.mod,
    #         batch_size=args.batch_size,
    #         n_top_peaks=model.config.peaks,
    #         n_top_genes=model.config.genes.index,
    #         binary=model.config.binary,
    #         use_seq=model.config.use_seq,
    #     )

    # if not args.fast_dev_run:
    #     out_dir = os.path.join(args.default_root_dir, args.data_dir)
    #     os.makedirs(out_dir, exist_ok=True)

    #     if args.mod == "multiome":
    #         if args.data_dir == model.config.data_dir:
    #             dataloader = dm.dataloader() #val_dataloader()
    #         else:
    #             dataloader = dm.dataloader()
    #     if args.rna:
    #         rna_dm = MixDataModule(
    #             data_dir=args.data_dir,
    #             modality="rna",
    #             batch_size=args.batch_size,
    #             n_top_peaks=model.config.peaks,
    #             n_top_genes=model.config.genes.index,
    #             binary=model.config.binary,
    #             use_seq=model.config.use_seq,
    #         )
    #     else:
    #         rna_dm = None
    #     if args.atac:
    #         atac_dm = MixDataModule(
    #             data_dir=args.data_dir,
    #             modality="atac",
    #             batch_size=args.batch_size,
    #             n_top_peaks=model.config.peaks,
    #             n_top_genes=model.config.genes.index,
    #             binary=model.config.binary,
    #             use_seq=model.config.use_seq,
    #         )
    #     else:
    #         atac_dm = None

    #     model.get_batch_features(dataloader, atac_dm, rna_dm, out_dir=out_dir)