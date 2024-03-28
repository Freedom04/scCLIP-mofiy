import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os
import scanpy as sc
# import colorcet as cc
import pynndescent
import numba
import muon as mu


@numba.njit(fastmath=True)
def correct_alternative_cosine(ds):
    result = np.empty_like(ds)
    for i in range(ds.shape[0]):
        result[i] = 1.0 - np.power(2.0, ds[i])
    return result


pynn_dist_fns_fda = pynndescent.distances.fast_distance_alternatives
pynn_dist_fns_fda["cosine"]["correction"] = correct_alternative_cosine
pynn_dist_fns_fda["dot"]["correction"] = correct_alternative_cosine

def plot_umap(
        adata,
        color=None,
        save=None,
        n_neighbors=30,
        min_dist=0.5,
        metric="euclidean",  #'correlation',
        use_rep="X",
        **pl_kwargs,
        ):
    sc.settings.set_figure_params(
        dpi=200, facecolor="white", figsize=(4, 4), frameon=True
    )
    sc.pp.neighbors(
        adata, metric=metric, use_rep=use_rep
    )  # , n_neighbors=n_neighbors),
    sc.tl.leiden(
        adata, resolution=0.6
    )
    sc.tl.umap(adata)  # , min_dist=min_dist)

    if "show" in pl_kwargs and not pl_kwargs["show"]:
        axis = sc.pl.umap(
            adata, color=color, save=save, wspace=0.4, ncols=4, **pl_kwargs
        )
        return axis[0].figure if isinstance(axis, list) else axis.figure
    else:
        sc.pl.umap(
            adata, color=color, save=save, wspace=0.65, ncols=4, show=False, **pl_kwargs
        )


def plot_mdata(
        mdata,
        color=None,
        save=None,
        n_neighbors=30,
        min_dist=0.5,
        metric="euclidean",  #'correlation',
        use_rep="X",
        **pl_kwargs,
        ):
    sc.settings.set_figure_params(
        dpi=200, facecolor="white", figsize=(4, 4), frameon=True
    )
    mu.pp.neighbors(
        mdata, metric=metric
    )
    mu.tl.leiden(mdata)
    mu.tl.umap(mdata)
    mu.pl.umap(
        mdata, color=color, save=save, wspace=0.65, ncols=4, show=False, **pl_kwargs
    )

def plot_paired_umap(
    adata,
    color=["cell_type", "modality"],
    save=None,
    n_neighbors=30,
    min_dist=0.5,
    metric="euclidean",  #'correlation',
    use_rep="X",
    # **tl_kwargs,
    **pl_kwargs,
):
    sc.settings.set_figure_params(
        dpi=80, facecolor="white", figsize=(4, 4), frameon=True
    )
    sc.pp.neighbors(
        adata, metric=metric, use_rep=use_rep
    )  # , n_neighbors=n_neighbors),
    sc.tl.leiden(adata)
    sc.tl.umap(adata)  # , min_dist=min_dist)

    ncols = 2
    nrows = 1
    figsize = 4
    wspace = 0.5
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * figsize + figsize * wspace * (ncols - 1), nrows * figsize),
    )
    plt.subplots_adjust(wspace=wspace)

    sc.pl.umap(adata, color="cell_type", ax=axs[0], show=False, legend_loc="on data")
    sc.pl.umap(adata, color="modality", ax=axs[1], show=False)
    concat = adata.obsm["X_umap"]
    plt.plot(
        concat[:, 0].reshape(2, -1),
        concat[:, 1].reshape(2, -1),
        color="gray",
        linestyle="dashed",
        linewidth=0.5,
    )
    plt.tight_layout()

    if save:
        plt.savefig(save)
    else:
        return plt.gcf()