import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os


DATA_DIR = "/Users/anqiwu/Desktop/iicd26-celltracking-ot-exploration/data/exp1"

N_REPS = 10
PERTURBATION_FRAC = 0.05   # drift noise per division, as a fraction of each PC's std

# Load MOSTA data
adata = sc.read_h5ad("/Users/anqiwu/Downloads/E9.5_E1S1.MOSTA.h5ad")

# Basic information
print(adata)

# Select top 2000 highly variable genes
adata_hvg = adata.copy()

sc.pp.highly_variable_genes(
    adata_hvg,
    n_top_genes=2000,
    flavor="seurat_v3",
    layer="count"   # use raw counts for HVG selection
)

hvg_genes = adata_hvg.var_names[
    adata_hvg.var["highly_variable"]
]

print(f"Selected {len(hvg_genes)} HVGs")

# normalize/log transform if needed
sc.pp.normalize_total(adata_hvg)
sc.pp.log1p(adata_hvg)

# PCA embedding
N_PCS = 30

sc.pp.pca(
    adata_hvg,
    n_comps=N_PCS
)

# cells × PCs
X = adata_hvg.obsm["X_pca"]

print(X.shape)

pc_columns = [f"PC{i+1}" for i in range(N_PCS)]

# per-PC std across all MOSTA cells, used to scale drift noise below
pc_std = X.std(axis=0)

dir = "/Users/anqiwu/Desktop/iicd26-celltracking-ot-exploration/"
# output folder
os.makedirs(dir + "data/exp1_with_expr", exist_ok=True)

# process each replicate

for rep in range(N_REPS):

    print(f"Processing rep{rep:02d}")

    rep_dir = f"{DATA_DIR}/rep{rep:02d}"

    lineage_path = f"{rep_dir}/lineage.csv"

    lineage = pd.read_csv(lineage_path)


    # assign founder expression from MOSTA data

    founders = lineage[
        lineage["parent_id"] == -1
    ]["cell_id"].values


    n_founders = len(founders)

    # sample MOSTA cells
    sampled = np.random.choice(
        adata.n_obs,
        size=n_founders,
        replace=False
    )


    expression = {}


    for cell_id, idx in zip(founders, sampled):

        expression[cell_id] = X[idx].copy()



    # propagte expression to descendants

    # process cells by birth order
    lineage_sorted = lineage.sort_values(
        "birth_time"
    )


    for _, row in lineage_sorted.iterrows():

        cell_id = row["cell_id"]
        parent = row["parent_id"]


        # already assigned founder
        if parent == -1:
            continue


        parent_expr = expression[parent]


        # add stochastic drift, scaled per-PC by that PC's std across MOSTA
        noise = np.random.normal(
            loc=0,
            scale=PERTURBATION_FRAC * pc_std,
            size=parent_expr.shape
        )


        child_expr = parent_expr + noise


        expression[cell_id] = child_expr



    # save PC embedding matrix

    expr_matrix = np.vstack(
        [
            expression[cell_id]
            for cell_id in lineage["cell_id"]
        ]
    )


    expr_df = pd.DataFrame(
        expr_matrix,
        columns=pc_columns
    )

    expr_df.insert(
        0,
        "cell_id",
        lineage["cell_id"]
    )


    expr_df.to_csv(
        f"{rep_dir}/pca_embedding.csv",
        index=False
    )
