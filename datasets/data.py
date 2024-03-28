import os
import muon as mu
import scanpy as sc
import numpy as np
import scipy
from anndata import AnnData
from typing import Union
from torch.utils.data import Dataset, random_split, Subset, DataLoader
from sklearn.preprocessing import maxabs_scale, LabelEncoder
import lightning as L


class MultiOmicsDataset(Dataset):
    def __init__(
        self,
        data_dir: str = None,
        backed: bool = False,
        n_top_genes: int = None,
        n_top_peaks: int = None,
        split: Union[float, str, list] = 0.9,
        mask: float = None,
        binary: bool = True, # whether scale each feature to the [-1, 1] range without breaking the sparsity.
        cell_type: str = "cell_type"    
            ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.backed = backed
        self.n_top_genes = n_top_genes
        self.n_top_peaks = n_top_peaks
        self.mask = mask
        self.binary = binary
        self.cell_type = cell_type
        
        self.read()
        self._preprocess_multiome()

        self.train_dataset, self.val_dataset = self._split(split)
        
        if self.cell_type in self.mdata.obs.columns:
            self.le = LabelEncoder()
            self.cell_types = self.le.fit_transform(self.mdata.obs[self.cell_type])
        else:
            self.cell_types = None        
        
        print(
            "RNA shape",
            self.mdata.mod["rna"].shape,
            "atac shape",
            self.mdata.mod["atac"].shape,
            flush=True,
        )
    
    def read(self):
        """
        read multiomics datasets
        """
        assert os.path.isfile(self.data_dir), f"MultiOme file {self.data_dir} not found"
        self.mdata = mu.read(self.data_dir, backed=self.backed)
        
        if 'atac' in self.mdata.mod.keys():
            self.mdata.mod['atac'].var_names_make_unique()
        if 'rna' in self.mdata.mod.keys():
            self.mdata.mod['rna'].var_names_make_unique()

    def random_split(self, train_ratio=0.9):
        """
        Randomly split a dataset into non-overlapping new datasets of given lengths.
        """
        train_size = int(train_ratio * len(self))
        val_size = len(self) - train_size
        return random_split(self, [train_size, val_size])
    
    def _split(self, split):
        """
        split the dataset into train-dataset and validate-dataset
        """
        if isinstance(split, float):
            return self.random_split(split)
        
        elif isinstance(split, str):  # split only one cell type
            if split.endswith("_r"):
                split = split.strip("_r")
                reverse = True
            else:
                reverse = False
            assert (
                split in self.mdata.obs["cell_type"].unique()
            ), f"Cell type {split} not found"
            if reverse:
                train_idx = np.where(self.mdata.obs["cell_type"] != split)[0]
                val_idx = np.where(self.mdata.obs["cell_type"] == split)[0]
            else:
                train_idx = np.where(self.mdata.obs["cell_type"] == split)[0]
                val_idx = np.where(self.mdata.obs["cell_type"] != split)[0]

            return Subset(self, train_idx), Subset(self, val_idx)

        elif isinstance(split, list):  # split multiple cell types # TO DO
            train_idx = self.mdata.obs[self.mdata.obs["cell_type"].isin(split)].index
            val_idx = self.mdata.obs[~self.mdata.obs["cell_type"].isin(split)].index
            
            return Subset(self, train_idx), Subset(self, val_idx)

    def reindex_genes(self, adata, genes):
        """
        reorder the `adata` dataset according to the `genes`
        """
        idx = [i for i, g in enumerate(genes) if g in adata.var_names]
        print("There are {} gene in selected genes".format(len(idx)))
        if len(idx) == len(genes):
            adata = adata[:, genes].copy()
        else:
            new_X = scipy.sparse.lil_matrix((adata.shape[0], len(genes)))
            new_X[:, idx] = adata[:, genes[idx]].X
            adata = AnnData(new_X.tocsr(), obs=adata.obs, var={"var_names": genes})
        
        return adata
     
    def _preprocess_rna(self):
        """
        preprocess scRNA-seq dataset
        """
        rna = self.mdata.mod["rna"]
        rna = rna[
            :,
            [
                gene
                for gene in rna.var_names
                if not str(gene).startswith(tuple(["ERCC", "MT-", "mt-", "mt"]))
            ],
        ].copy()
        # sc.pp.filter_cells(rna, min_genes=200) #, max_counts=15000, max_genes=5000)
        # sc.pp.filter_genes(rna, min_cells=3)
        sc.pp.normalize_total(rna, target_sum=1e4)
        sc.pp.log1p(rna)

        if isinstance(self.n_top_genes, int):
            if self.n_top_genes > 0:
                sc.pp.highly_variable_genes(
                    rna, n_top_genes=self.n_top_genes, inplace=False, subset=True
                ) 
        elif self.n_top_genes is not None:
            if len(self.n_top_genes) != len(rna.var_names):
                rna = self.reindex_genes(rna, self.n_top_genes)

        if self.binary:
            rna.X = maxabs_scale(rna.X)
        
        self.mdata.mod["rna"] = rna

    def _preprocess_atac(self):
        """
        preprocess scATAC-seq dataset
        """
        atac = self.mdata.mod["atac"]
        atac.X[atac.X > 0] = 1

        if isinstance(self.n_top_peaks, int):
            sc.pp.highly_variable_genes(
                atac,
                n_top_genes=self.n_top_peaks,
                batch_key="batch",
                inplace=False,
                subset=True,
            )
        elif self.n_top_peaks is not None:
            if len(self.n_top_peaks) != len(atac.var_names):
                raise ValueError('n_top_peaks must be None or a list of length {}'.format(len(atac.var_names)))
        # elif self.linked:
        #     print(
        #         "Linking {} peaks to {} genes".format(
        #             atac.shape[1], self.mdata.mod["rna"].shape[1]
        #         ),
        #         flush=True,
        #     )
        #     gene_peak_links = self._get_gene_peak_links(dist=self.linked)
        #     peak_index = np.unique(gene_peak_links[1])
        #     gene_index = np.unique(gene_peak_links[0])
        #     atac = atac[:, peak_index].copy()
        #     self.mdata.mod["rna"] = self.mdata.mod["rna"][:, gene_index].copy()

        self.mdata.mod["atac"] = atac

    def _preprocess_multiome(self):
        """
        preprocess multiomics datasets
        """
        self._preprocess_rna()
        self._preprocess_atac()
        # Subset observations (samples or cells) in-place taking observations present only in all modalities.
        mu.pp.intersect_obs(self.mdata)
    
    def _transform_rna(self, x):
        x = x / x.sum() * 1e4
        x = np.log1p(x)
        return x
    
    def _transform_atac(self, x):
        x[x > 1] = 1
        return x
    
    def _transform_multiome(self, batch):
        return {
            "atac": self._transform_atac(batch["atac"]),
            "rna": self._transform_rna(batch["rna"]),
        }
    
    def get_rna(self, index):
        x = self.mdata["rna"].X[index].toarray().squeeze()
        
        if self.mask is not None:
            index = np.where(x > 0)[0]
            index = np.random.choice(
                index, size=int(len(index) * self.mask), replace=False
            )
            x[index] = 0
        return x
    
    def get_atac(self, index):
        x = self.mdata["atac"].X[index].toarray().squeeze()
        
        if self.mask is not None:
            index = np.where(x > 0)[0]
            index = np.random.choice(
                index, size=int(len(index) * self.mask), replace=False
            )
            x[index] = 0
        return x
    
    def get_multiome(self, index):
        return {
            "atac": self.get_atac(index), 
            "rna": self.get_rna(index),
        }
    
    def __getitem__(self, index):
        x = self.get_multiome(index)
        
        if self.backed:
            x = self._transform_multiome(x)
        if self.cell_types is not None:
            x.update({"cell_type": self.cell_types[index]})
        return x
    
    def __len__(self):
        return self.mdata.mod["atac"].shape[0]
    

class MultiOmicsModule(L.LightningDataModule):
    def __init__(self, 
                 data_dir: str = None,
                 batch_size: int = 256,
                 num_workers: int = 4,
                 pin_memory: bool = False,
                 ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset = MultiOmicsDataset(self.data_dir)

    # def prepare_data(self):
    #     self.dataset = MultiOmicsDataset(self.data_dir)
        
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset = self.dataset.train_dataset
            self.val_dataset = self.dataset.val_dataset

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = self.dataset

        if stage == "predict":
            self.predict_dataset = self.dataset

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return loader

    def predict_dataloader(self, dataset=None, shuffle=False):
        loader = DataLoader(
            dataset or self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return loader
    
    

if __name__ == "__main__":
    data_path = "../data/multiome/human_brain_3k.h5mu"

    multiDataset = MultiOmicsDataset(data_dir=data_path)
    multiomicsDataModule = MultiOmicsModule(data_dir=data_path)
