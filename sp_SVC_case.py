import os

import scanpy as sc
import pandas as pd
import numpy as np

from tqdm import tqdm
from spatial_metric import spatial_gene_autocorr, plot_compare_spatial_autocorr, get_sampeled_adata, run_cluster_plot


resolutions = [0.3, 0.5, 0.8]
resolutions = [0.1, 0.3, 0.5]
patient_ids = ["P1CRC", "P2CRC", "P5CRC"]
patient_ids = ["P1CRC"]

patient_ids = ["P2CRC", "P5CRC", "P3NAT","P5NAT"]
patient_ids = ["P5NAT","P3NAT"]
patient_ids = ["P1CRC", "P2CRC", "P5CRC", "P3NAT","P5NAT"]

data_type = "HD"

cell_type_col = "Level1"

sample_size = 30000

raw_data_path = "/home/wys/Sim2Real-ST/REVISE_data_process/raw_data"
svc_data_path = "../REVISE/results/HD"
save_dir = "output/sp_SVC_case"


for patient_id in tqdm(patient_ids, desc="Processing patients"):
            
    save_path = f"{save_dir}/{patient_id}_{data_type}"
    os.makedirs(save_path, exist_ok=True)


    print(f"Processing original data for {patient_id}_{data_type}")
    adata_sp = sc.read(f"{raw_data_path}/{patient_id}_{data_type}.h5ad")
    adata_sp = adata_sp[adata_sp.obs[cell_type_col] != "Unknown"].copy()

    if sample_size is not None:
        adata_sp = get_sampeled_adata(adata_sp, sample_size=sample_size, seed=0)
        print(adata_sp.obs[cell_type_col].value_counts())

    run_cluster_plot(adata_sp, f"{save_path}/original", 
                        resolutions = resolutions, 
                        cell_type_col = "Level1",
                        plot_single_flag=True) 
    # moranI_sp, gearyC_sp, _, _ = spatial_gene_autocorr(adata_sp, cell_type_col = "Level1", save_dir = f"{save_path}/original")
    

    print(f"Processing SVC data for {patient_id}_{data_type}")
    adata_sp_svc = sc.read(f"{svc_data_path}/{patient_id}/{patient_id}_{data_type}_pot_REVISE.h5ad")
    adata_sp_svc.X = adata_sp_svc.layers["ot_smooth"]
    # adata_sp_svc.X.data = np.round(adata_sp_svc.X.data, decimals=0)
    # adata_sp_svc.X.data = np.floor(adata_sp_svc.X.data)

    adata_sp_svc = adata_sp_svc[adata_sp_svc.obs[cell_type_col]!= "Unknown"].copy()
    
    if sample_size is not None:
        adata_sp_svc = get_sampeled_adata(adata_sp_svc, sample_size=sample_size, seed=0)
        print(adata_sp.obs[cell_type_col].value_counts())

    run_cluster_plot(adata_sp_svc, f"{save_path}/sp_SVC", 
                        resolutions = resolutions,
                        cell_type_col = "Level1",
                        plot_single_flag=True)
    # moranI_sp_svc, gearyC_sp_svc, _, _ = spatial_gene_autocorr(adata_sp_svc, cell_type_col = "Level1", 
    #                                                             save_dir = f"{save_path}/sp_SVC")
    
    # plot_compare_spatial_autocorr(moranI_sp_svc, moranI_sp, save_dir = save_path, mode = "moran")
    # plot_compare_spatial_autocorr(gearyC_sp_svc, gearyC_sp, save_dir = save_path, mode = "geary")