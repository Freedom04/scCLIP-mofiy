
class Config:
    def __init__(self,
                n_peaks,
                n_genes,
                normalize: bool = True,
                logit_scale=2.6592,
                requires_grad: bool = False,
                hidden_size: int = 512 ,
                projection_dim: int = 128,
                lr=1.5e-4,
                weight_decay=0.05,
                num_patches: int = 128,
                num_heads: int = 8,
                ffn_dim: int = 2048,
                dropout=0.1,
                num_layers: int = 6,
                hidden_dropout_prob=0.1,
                ) -> None:
        self.n_peaks = n_peaks
        self.n_genes = n_genes
        self.normalize = normalize
        self.logit_scale = logit_scale
        self.requires_grad = requires_grad
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_patches = num_patches
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.hidden_dropout_prob = hidden_dropout_prob
