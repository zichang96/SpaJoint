import scanpy as sc 
from SpaJoint.spaJoint import spaJoint
import pandas as pd
import numpy as np

dataset = "example"
ref_adata = sc.read("./" + dataset + "/scRNA.h5ad")
inf_adata = sc.read("./" + dataset + "/ST.h5ad")
save_dir = "./" + dataset
print(ref_adata.shape)
print(inf_adata.shape)

inf_adata.obsm['spatial'] = np.array(inf_adata.obs[['x','y']])
ref_adata.obs["CellType"] = ref_adata.obs["cell_type"].copy()
ref_adata.obs["CellType"] = ref_adata.obs["CellType"].astype("category")
ref_adata.obs["celltype"]=ref_adata.obs["CellType"].astype("category").cat.codes 

spajoint = spaJoint(ref_ds=ref_adata,inf_ds=inf_adata,save_dir=save_dir,epochs=50,batch_size=256)
spajoint.PrepareDataLoader_stage()
spajoint.BuildNet_stage()
spajoint.TrainingProcessStage()

celltype_density = pd.DataFrame(spajoint.inf_ds.obsm["cell_pred"])
cell_type_dict = dict(zip(ref_adata.obs['celltype'],ref_adata.obs['CellType']))
celltype_density.rename(columns=cell_type_dict,inplace=True)
celltype_density.to_csv(save_dir +'/spajoint_result.txt')
