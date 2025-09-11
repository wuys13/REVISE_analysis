import os
import numpy as np
import pandas as pd
import squidpy as sq
import scanpy as sc
import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import ttest_ind

from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score


import warnings
warnings.filterwarnings("ignore")

def process_adata_autocorr(adata_raw, scale = False, filter = True, min_genes = 10, min_cells = 3):
    
    adata = adata_raw.copy()
    if filter:
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", subset=True)
    # adata = adata[:, adata.var.highly_variable].copy()
    
    if scale:
        sc.pp.scale(adata, max_value=10)
    # sc.tl.pca(adata, svd_solver='arpack')
    # sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)  # 计算邻居图
    
    return adata
def computer_gene_autocorr(adata_gene, mode = "moran", scale = False, filter = False):
    
    if mode == "moran":
        add = "I"
    elif mode == "geary":
        add = "C"
    uns_name = mode + add

    adata_gene = process_adata_autocorr(adata_gene, scale = scale, filter = filter)
        
    sq.gr.spatial_neighbors(adata_gene)
    sq.gr.spatial_autocorr( # before scale can be = True # adata.uns['moranI']
        adata_gene,
        # connectivity_key="spatial_distances",
        mode=mode,
        genes=adata_gene.var_names,
        n_perms=1,
        n_jobs=1,
    )
    df = adata_gene.uns[uns_name]
    df = df[add]
    
    return df

def get_top_n_rows_by_column(all_df, top_n = 10):
    # 初始化一个空的DataFrame，用于存储最终结果
    result_df = pd.DataFrame()
    
    # 遍历DataFrame的每一列
    for column in all_df.columns:
        # 对当前列按值降序排列，并取前top_n行的索引
        top_n_indices = all_df[column].nlargest(top_n).index
        
        # 根据索引提取对应的行，并将这些行添加到结果DataFrame中
        result_df = pd.concat([result_df, all_df.loc[top_n_indices]])
    
    # 去重，因为同一行可能在多个列的top_n中被选中
    result_df = result_df.drop_duplicates()
    
    return result_df

def save_gene_autocorr_result(all_df, mode, save_dir = None, sort_values = "All", top_n = 10):
    all_df.fillna(0, inplace=True) 
    # all_df.to_csv(f"{save_dir}/{mode}_autocorr.csv")
    if sort_values is not None:
        all_df.sort_values(by = "All", ascending = False, inplace = True)

    
    select_df = get_top_n_rows_by_column(all_df, top_n)
    # 创建热图
    plt.figure(figsize=(8, 6))  # 设置图形大小
    sns.clustermap(select_df, annot=False, cmap="coolwarm",
                            row_cluster = False,
                            # standard_scale=1,
                            )

    plt.title(f"{mode}_all_cell_type", fontsize=16)

    if save_dir is not None:
        plt.savefig(f"{save_dir}/{mode}_heatmap.pdf", dpi=300)
        all_df.to_csv(f"{save_dir}/{mode}_autocorr.csv")
    else:
        plt.show()

    return all_df, select_df

def spatial_gene_autocorr(adata_sp, cell_type_col = "Level1", save_dir = None):
    """
    Calculate spatial autocorrelation for each cell type.
    """
    if cell_type_col is not None:
        cts = list(adata_sp.obs[cell_type_col].unique())
    else:
        cts = []
    cts.append("All")
    cts.sort()
    print(cts)

    all_moranI_df = pd.DataFrame()
    all_gearyC_df = pd.DataFrame()
    for select_ct in tqdm(cts, desc="Spatial autocorrelation"):
        if select_ct == "All":
            adata_gene = adata_sp.copy()
        else:
            adata_gene = adata_sp[adata_sp.obs[cell_type_col] == select_ct].copy()
        print(adata_gene.shape)
        if adata_gene.shape[0] <= 50:
            continue
        
        moranI_df = computer_gene_autocorr(adata_gene, mode = "moran", scale = False, filter = False)
        # print(moranI_df)
        overlap_genes = list(set(moranI_df.index) & set(adata_sp.var_names))
        moranI_df = moranI_df.loc[overlap_genes]
        moranI_df.name = select_ct
        # print(moranI_df)

        gearyC_df = computer_gene_autocorr(adata_gene, mode = "geary", scale = False, filter = False)
        overlap_genes = list(set(gearyC_df.index) & set(adata_sp.var_names))
        gearyC_df = gearyC_df.loc[overlap_genes]
        gearyC_df.name = select_ct

        print(select_ct, np.nanpercentile(moranI_df, [0, 25, 50, 75, 100]))
        all_moranI_df = pd.concat([all_moranI_df, moranI_df], axis=1)
        all_gearyC_df = pd.concat([all_gearyC_df, gearyC_df], axis=1)

    all_moranI_df, select_moranI_df = save_gene_autocorr_result(all_moranI_df, mode="moran", save_dir = save_dir)
    all_gearyC_df, select_gearyC_df = save_gene_autocorr_result(all_gearyC_df, mode="geary", save_dir = save_dir)

    return all_moranI_df, all_gearyC_df, select_moranI_df, select_gearyC_df



def plot_compare_spatial_autocorr(df_sp_svc, df_original, save_dir, mode):
    """
    Generate a boxplot comparing two dataframes across cell types with significance testing.
    
    Parameters:
    df_sp_svc (pd.DataFrame): Dataframe with genes as rows, cell types as columns.
    df_original (pd.DataFrame): Dataframe with genes as rows, cell types as columns.
    output_file (str): Path to save the boxplot image.
    """
    # Ensure dataframes have the same structure
    if not df_sp_svc.columns.equals(df_original.columns):
        raise ValueError("Dataframes must have the same columns (cell types).")
    
    # Prepare data for boxplot
    cell_types = df_sp_svc.columns
    data = []
    for cell_type in cell_types:
        # Extract values for each cell type
        sp_svc_values = df_sp_svc[cell_type].dropna()
        original_values = df_original[cell_type].dropna()
        # Add to data list with labels
        for val in sp_svc_values:
            data.append({'Cell Type': cell_type, 'Value': val, 'Source': 'SP_SVC'})
        for val in original_values:
            data.append({'Cell Type': cell_type, 'Value': val, 'Source': 'Original'})
    
    # Convert to dataframe
    plot_df = pd.DataFrame(data)
    
    # Set up the plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Cell Type', y='Value', hue='Source', data=plot_df, palette=['#1f77b4', '#ff7f0e'])
    
    # Add dashed lines between cell types
    for i in range(len(cell_types) - 1):
        plt.axvline(x=i + 0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Rotate x-axis labels
    plt.xticks(rotation=30, ha='right')
    
    # Add significance annotations
    for i, cell_type in enumerate(cell_types):
        sp_svc_vals = plot_df[(plot_df['Cell Type'] == cell_type) & (plot_df['Source'] == 'SP_SVC')]['Value']
        orig_vals = plot_df[(plot_df['Cell Type'] == cell_type) & (plot_df['Source'] == 'Original')]['Value']
        if len(sp_svc_vals) > 0 and len(orig_vals) > 0:
            t_stat, p_val = ttest_ind(sp_svc_vals, orig_vals, nan_policy='omit')
            # Add stars for significance
            if p_val < 0.05:
                sig = '*' if p_val >= 0.01 else '**' if p_val >= 0.001 else '***'
                max_y = max(sp_svc_vals.max(), orig_vals.max())
                plt.text(i, max_y + 0.05 * (plot_df['Value'].max() - plot_df['Value'].min()), 
                        sig, ha='center', va='bottom', fontsize=12)
    
    # Adjust layout and save
    plt.title(f'Spatial Autocorrelation ({mode}) Comparison by Cell Type')
    plt.xlabel('Cell Type')
    plt.ylabel('Spatial Autocorrelation')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/Compare_{mode}.pdf")

def compute_diff_spatial_autocorr(df_sp_svc, df_original):
    """
    Compute normalized difference between two dataframes, return long-format sorted dataframe.
    
    Parameters:
    df_sp_svc (pd.DataFrame): Dataframe with genes as rows, cell types as columns.
    df_original (pd.DataFrame): Dataframe with genes as rows, cell types as columns.
    
    Returns:
    pd.DataFrame: Long-format dataframe with columns 'Gene', 'Cell_Type', 'Normalized_Difference',
                  sorted by absolute Normalized_Difference in descending order.
    """
    # Ensure dataframes have the same structure
    if not (df_sp_svc.index.equals(df_original.index) and df_sp_svc.columns.equals(df_original.columns)):
        raise ValueError("Dataframes must have the same index (genes) and columns (cell types).")
    
    # Compute difference
    diff = df_sp_svc - df_original
    
    # Compute sum of absolute values
    abs_sum = np.abs(df_sp_svc) + np.abs(df_original)
    # abs_sum = 1
    
    # Compute normalized difference (handle division by zero)
    normalized_diff = diff / abs_sum
    normalized_diff = normalized_diff.fillna(0)  # Replace NaN (from 0/0) with 0
    
    # Convert to long format
    result = normalized_diff.stack().reset_index()
    result.columns = ['Gene', 'Cell_Type', 'Normalized_Difference']
    
    # Sort by absolute normalized difference
    result['Abs_Difference'] = result['Normalized_Difference'].abs()
    result = result.sort_values('Abs_Difference', ascending=False).drop('Abs_Difference', axis=1)
    
    return result


def clustering_metrics(adata, pred_label_key, true_label_key):
    pred_labels = adata.obs[pred_label_key].values
    true_labels = adata.obs[true_label_key].values
    # 转为数字
    true_labels = pd.Categorical(true_labels).codes
    pred_labels = pd.Categorical(pred_labels).codes
    
    # print(len(np.unique(pred_labels)), len(np.unique(true_labels)))
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    # print(f"ARI: {ari:.4f}, NMI: {nmi:.4f}", len(np.unique(pred_labels)), len(np.unique(true_labels)))
    return ari, nmi






import scanpy as sc
import numpy as np
def get_sampeled_adata(adata, sample_size = 10000, seed = 42):
    
    np.random.seed(seed)
    n_cells = adata.shape[0]
    sample_size = min(sample_size, n_cells)
    print(f"Sampling {sample_size} cells from {n_cells} total cells.")
    sampled_indices = np.random.choice(n_cells, size=sample_size, replace=False)
    adata_sampled = adata[sampled_indices].copy()

    return adata_sampled


def get_sub_adata(adata, cell_type_col = "Level1", cell_type = "Fibroblast"):
    """
    获取指定细胞类型的子集数据
    """
    adata_subset = adata[adata.obs[cell_type_col] == cell_type].copy()
    print(f"Subset shape: {adata_subset.shape}")
    
    return adata_subset

def process_adata_cluster(adata_raw, scale = False, filter = True, min_genes = 50, min_cells = 3, select_highly_variable = True):
    
    adata = adata_raw.copy()
    if filter:
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)
    # filter MT genes
    adata = adata[:, ~adata.var_names.str.startswith('MT-')].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if adata.shape[1] > 2000 and select_highly_variable:
        print(f"Highly variable genes filtering for {adata.shape[1]} genes.")
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", subset=True)
        adata = adata[:, adata.var.highly_variable].copy()
    
    if scale:
        sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)  # 计算邻居图
    
    return adata

def get_adata_cluster(adata, method = "ledien", resolutions = [0.3, 0.5, 0.8]):
    
    for res in resolutions:
        
        if method == "ledien":
            sc.tl.leiden(adata, resolution=res, key_added=f"leiden_res_{res}")
            n_clusters = len(adata.obs[f"leiden_res_{res}"].cat.categories)
            print(f"Number of clusters for leiden resolution {res}: {n_clusters}")
        elif method == "louvain":
            sc.tl.louvain(adata, resolution=res, key_added=f"louvain_res_{res}")
            n_clusters = len(adata.obs[f"louvain_res_{res}"].cat.categories)
            print(f"Number of clusters for louvain resolution {res}: {n_clusters}")
        else:
            print(f"Unknown method: {method}")
        
    return adata


def run_cluster_plot(adata_sp_cluster, save_dir, resolutions = [0.3, 0.5, 0.8], cell_type_col = "Level1" ):
    os.makedirs(save_dir, exist_ok=True)
    
    adata_sp_cluster = process_adata_cluster(adata_sp_cluster, scale = False, filter = True)
    adata_sp_cluster = get_adata_cluster(adata_sp_cluster, resolutions = resolutions)
    
    all_metric_df = pd.DataFrame()
    for resolution in resolutions:
        ari, nmi = clustering_metrics(adata_sp_cluster, f"leiden_res_{resolution}", cell_type_col)
        cluster_num = adata_sp_cluster.obs[f"leiden_res_{resolution}"].nunique()
        metric_df = pd.DataFrame(
            {
                "resolution": [resolution],
                "ARI": [ari],
                "NMI": [nmi],
                "cluster_num": [cluster_num]
            })
        all_metric_df = pd.concat([all_metric_df, metric_df], axis=0)
    all_metric_df.to_csv(f"{save_dir}/metric.csv", index=False)
    print(all_metric_df)


    sc.tl.tsne(adata_sp_cluster, n_pcs=30, random_state=0)
    plt.figure(figsize=(10, 10))
    sc.pl.tsne(adata_sp_cluster, 
               color=[f"leiden_res_{resolutions[0]}", f"leiden_res_{resolutions[1]}",
                      f"leiden_res_{resolutions[2]}", cell_type_col], ncols=2,
               show = False
               )
    plt.savefig(f"{save_dir}/tsne.png")

    sc.tl.umap(adata_sp_cluster)
    plt.figure(figsize=(10, 10))
    sc.pl.umap(adata_sp_cluster, 
               color=[f"leiden_res_{resolutions[0]}", f"leiden_res_{resolutions[1]}",
                      f"leiden_res_{resolutions[2]}", cell_type_col], ncols=2,
               show = False
               )
    plt.savefig(f"{save_dir}/umap.png")

    plt.figure(figsize=(10, 10))
    sc.pl.scatter(adata_sp_cluster, size=30,
            color=[f"leiden_res_{resolutions[1]}", cell_type_col], 
            x = "x", y = "y", 
            show = False
        )
    plt.savefig(f"{save_dir}/spatial.png")