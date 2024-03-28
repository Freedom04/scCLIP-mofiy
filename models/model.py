import sys
sys.path.append(".")
import torch
import torch.nn as nn
import lightning as L
from torch import optim

from models.module import TransformerModel, CLIPLoss
from utils.metrics import matching_metrics

"""
config = {
    "normalize": ...,
    "n_peaks": ...,
    "n_genes": ...,
    "logit_scale": ...,
    "requires_grad": ...,
    "hidden_size": ...,
    "projection_dim":...,
    "lr": ...,
    "weight_decay": ...,
    "num_patches": ...,
    "num_heads": ...,
    "ffn_dim": ...,
    "dropout": ...,
    "num_layers": ...,

}

"""
class scCLIPModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.atac_encoder = TransformerModel(config, config.n_peaks)
        self.rna_encoder = TransformerModel(config, config.n_genes)
        self.criterion = CLIPLoss(logit_scale=config.logit_scale, requires_grad=config.requires_grad)

        self.atac_proj = nn.Linear(config.hidden_size, config.projection_dim)
        self.rna_proj = nn.Linear(config.hidden_size, config.projection_dim)

        self.save_hyperparameters()

    def forward(
        self,
        atac=None,      # [batch_size, atac_peak_size]
        rna=None,       # [batch_size, rna_gene_size]
    ):
        # print(atac.size())
        # print(rna.size())
        if atac is not None:
            atac_embeds = self.atac_encoder(atac)[:, 0]
            atac_embeds = self.atac_proj(atac_embeds)
            if self.config.normalize:
                atac_embeds = atac_embeds / atac_embeds.norm(dim=-1, keepdim=True)
        else:
            atac_embeds = None

        if rna is not None:
            # self.rna_encoder(rna).size() is [batch_size, sequence_length, embedding_size]
            rna_embeds = self.rna_encoder(rna)[:, 0]    # [batch_size, embedding_size] 
            rna_embeds = self.rna_proj(rna_embeds)      # [batch_size, projection_dim]
            if self.config.normalize:
                rna_embeds = rna_embeds / rna_embeds.norm(dim=-1, keepdim=True)
        else:
            rna_embeds = None

        return atac_embeds, rna_embeds

    def _step(self, batch, batch_idx, mode):
        atac_embeds, rna_embeds = self(batch["atac"], batch["rna"])
        loss, similarity = self.criterion(atac_embeds, rna_embeds)

        acc, matchscore, foscttm = matching_metrics(similarity)
        log_dict = {
            f"acc/{mode}": acc,
            f"foscttm/{mode}": foscttm,
            f"matchscore/{mode}": matchscore,
            f"loss/{mode}": loss,
        }

        if mode == "predict":
            return atac_embeds, rna_embeds, log_dict

        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
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