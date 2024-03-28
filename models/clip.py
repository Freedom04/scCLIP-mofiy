import sys
sys.path.append("../")
import torch
import torch.nn as nn
import lightning as L
from torch import optim
from torch.utils.data import Subset
from anndata import AnnData, concat
from mudata import MuData
from tqdm import tqdm
import scanpy as sc
import numpy as np

from models.vit import ViTModel
from utils.metrics import matching_metrics
from utils.logger import create_logger
from utils.plot import plot_umap, plot_paired_umap


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits, torch.arange(len(logits), device=logits.device)
    )


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    atac_loss = contrastive_loss(similarity.T)
    return (caption_loss + atac_loss) / 2.0


class CLIPLoss(nn.Module):
    def __init__(self, logit_scale=2.6592, requires_grad=False):
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.ones([]) * logit_scale, requires_grad=requires_grad
        )

    def forward(self, atac_embeds, rna_embeds):
        # normalized features
        # atac_embeds = atac_embeds / atac_embeds.norm(dim=-1, keepdim=True)
        # rna_embeds = rna_embeds / rna_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_atac = torch.matmul(atac_embeds, rna_embeds.t()) * logit_scale
        logits_per_rna = logits_per_atac.T

        loss = clip_loss(logits_per_atac)

        return loss, logits_per_atac  # , logits_per_rna


class CLIPModel(L.LightningModule):
    def __init__(
        self,
        config,
        atac_config,
        rna_config,
    ):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.atac_model = ViTModel(atac_config)
        self.rna_model = ViTModel(rna_config)

        # config.hidden_size = self.atac_model.config.hidden_size
        self.atac_projection = nn.Linear(
            self.atac_model.config.hidden_size, config.projection_dim, bias=False
        )
        
        self.rna_projection = nn.Linear(
            self.rna_model.config.hidden_size, config.projection_dim, bias=False
        )

        self.criterion = CLIPLoss(
            self.config.logit_scale, requires_grad=self.config.requires_grad
        )

        print(f"atac_num_patches: {self.atac_model.embeddings.num_patches}", flush=True)
        print(f"rna_num_patches: {self.rna_model.embeddings.num_patches}", flush=True)

    def forward(
        self,
        atac_values=None,
        rna_values=None,
    ):
        atac_embeds = self._get_atac_features(atac_values)
        rna_embeds = self._get_rna_features(rna_values)

        return atac_embeds, rna_embeds

    def _step(self, batch, batch_idx, mode):  # TO DO: add reconstruction
        atac_embeds, rna_embeds = self(batch["atac"], batch["rna"])
        loss, similarity = self.criterion(atac_embeds, rna_embeds)

        acc, matchscore, foscttm = matching_metrics(similarity)
        log_dict = {
            f"acc/{mode}": acc,
            f"matchscore/{mode}": matchscore,
            f"foscttm/{mode}": foscttm,
            f"loss/{mode}": loss,
        }

        # logit_scale learnable
        if self.config.requires_grad:
            log_dict.update({"logit_scale": self.criterion.logit_scale})

        if mode == "predict":
            return atac_embeds, rna_embeds, log_dict

        # log_dict.update({f'loss/{mode}': loss})
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def _get_rna_features(self, rna_values=None):
        rna_outputs = self.rna_model(rna_values)

        rna_features = rna_outputs[1]
        rna_features = self.rna_projection(rna_features)

        if self.config.normalize:
            rna_features = rna_features / rna_features.norm(dim=-1, keepdim=True)

        return rna_features

    def _get_atac_features(self, atac_values=None):
        atac_outputs = self.atac_model(atac_values)

        atac_features = atac_outputs[1]  # pooled_output
        atac_features = self.atac_projection(atac_features)

        if self.config.normalize:
            atac_features = atac_features / atac_features.norm(dim=-1, keepdim=True)

        return atac_features
    
    def _get_batch_features(
        self, dataloader, cell_type="cell_type",modality="rna", out_dir=".", device="cuda"
    ):
        if isinstance(dataloader.dataset, Subset):
            obs = dataloader.dataset.dataset.mdata.obs.iloc[dataloader.dataset.indices]
        else:
            obs = dataloader.dataset.mdata.obs

        self.to(device)
        fn = self._get_rna_features if modality == "rna" else self._get_atac_features
        adata = torch.empty(0, self.config.projection_dim)
        batch_labels = np.empty((0, 1))

        for batch_id, batch in enumerate(tqdm(dataloader.predict_dataloader(), desc=modality)):
            embeds = fn(batch[modality].to(device)).detach().cpu()
            batch_size = embeds.size()[0]
            adata = torch.cat((adata, embeds), dim=0)
            
            batch_ids = np.tile(np.array([batch_id]), batch_size)
            batch_ids = np.expand_dims(batch_ids, axis=0).reshape(-1, 1)
            batch_labels = np.concatenate((batch_labels, batch_ids), axis=0)

        obs['batch'] = batch_labels
        adata = AnnData(adata.numpy(), obs=obs)
        
        adata = AnnData(adata, obs=obs)
        
        sc.settings.figdir = out_dir
        plot_umap(adata, color=cell_type, metric="cosine", save=f"_{modality}.png")
        
        adata.write(f"{out_dir}/{modality}.h5ad")
        
        return adata

    def get_batch_features(
        self, dataloader, cell_type="cell_type", out_dir="."
    ):
        log = create_logger("", fh=out_dir + "/log.txt")
        if not self.config.normalize:
            out_dir = f"{out_dir}_no_norm"
        
        # get rna embeddings and atac embeddings
        if dataloader is not None:
            rna_embeds = self._get_batch_features(
                dataloader, modality="rna", out_dir=out_dir
            )
            atac_embeds = self._get_batch_features(
                dataloader, modality="atac", out_dir=out_dir
            )
            loss, similarity = self.criterion(atac_embeds.X, rna_embeds.X)
            acc, match_score, foscttm = matching_metrics(similarity)
            # acc, match_score, foscttm = matching_metrics(
            #     x=atac_embeds.obsm["X_umap"], y=rna_embeds.obsm["X_umap"]
            # )
            if log is not None:
                log.info(
                    f"atac-rna\nacc: {acc:.4f}\nmatching score: {match_score:.4f}\nfoscttm: {foscttm:.4f}"
                )
            else:
                print(
                    f"atac-rna\nacc: {acc:.4f}\nmatching score: {match_score:.4f}\nfoscttm: {foscttm:.4f}",
                    flush=True,
                )
            
            multi_omics_embeds = MuData({'rna': rna_embeds, 'atac': atac_embeds})
            multi_omics_embeds.write(f"{out_dir}/multiomics_embeds.h5mu")

            concat_embeds = concat(
                [atac_embeds, rna_embeds],
                label="modality",
                keys=["atac", "rna"],
                index_unique="#",
            )
            
            sc.settings.figdir = out_dir
            if dataloader is not None:
                plot_umap(
                    concat_embeds,
                    color=[cell_type, "modality"],
                    metric="cosine",
                    save="_concat.png",
                )
                plot_paired_umap(concat_embeds, color=celltype, save=os.path.join(out_dir, 'umap_concat.png'))
            else:
                plot_umap(
                    concat_embeds, color=cell_type, metric="cosine", save="_concat.png"
                )

            concat_embeds.write(f"{out_dir}/concat.h5ad")
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "predict")
    
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )

        return optimizer