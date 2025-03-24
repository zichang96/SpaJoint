import os 
from SpaJoint.logger import create_logger   
import torch
import os
import pandas as pd 
import scanpy as sc 
import pandas as pd 
from SpaJoint.utils import ReferenceDataSet,InferenceDataSet_stage
from SpaJoint.utils import preprocess,filter_with_overlap_gene, np_unranked_unique,def_cycle,save_checkpoint
from SpaJoint.model import Net_cell,Net_encoder
from torch.autograd import Variable
from SpaJoint.loss import L1regularization, CellLoss, EncodingLoss
import torch.optim as optim
import random
from scipy.linalg import norm
from scipy.special import softmax
from tqdm import tqdm
import numpy as np
from time import time 
import squidpy as sq

## create spaJoint
class spaJoint:
    def __init__(self,verbose=True,save_dir="./output/",ref_ds=None,inf_ds=None,input_size=None,num_class=None,batch_size=256,n_neighbor=6,lr=0.001,
                 epochs=100,embedding_size=64,momentum=0.9,with_crossentorpy=True,seed=1,checkpoint="",use_cuda=False,threads=1):
        """                               
        create spaJoint Model object
        Argument:
        ------------------------------------------------------------------
        - verbose: 'str',optional, Default,'True', write additional information to log file when verbose=True
        - save_dir: folder to save result and log information
        ------------------------------------------------------------------
        """     
        self.verbose=verbose
        self.save_dir=save_dir 

        self.ref_ds = ref_ds 
        self.inf_ds = inf_ds
        self.input_size=input_size 
        self.number_of_class = num_class
        self.batch_size = batch_size
        self.n_neighbor = n_neighbor
        self.lr = lr
        self.epochs = epochs
        self.embedding_size = embedding_size
        self.momentum = momentum
        self.with_crossentorpy = with_crossentorpy
        self.seed = seed
        self.checkpoint = checkpoint
 
        self.use_cuda = use_cuda
        self.threads = threads

        if not self.use_cuda:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')

            # hardware constraint for speed test
        torch.set_num_threads(1)

        os.environ['OMP_NUM_THREADS'] = '1'
        torch.manual_seed(self.seed)

        assert ref_ds is not None
        assert inf_ds is not None
        
        preprocess(ref_ds)
        preprocess(inf_ds)
        inf_ds, ref_ds = filter_with_overlap_gene(inf_ds, ref_ds)
        sq.gr.spatial_neighbors(inf_ds, n_rings=2,coord_type="generic", n_neighs=n_neighbor) 
        # sq.gr.spatial_neighbors(inf_ds, n_rings=2,coord_type="grid", n_neighs=n_neighbor) 
        inf_ds.obsm['graph_neigh'] = np.array(inf_ds.obsp['spatial_connectivities'].todense()).copy() + np.eye(inf_ds.obsp['spatial_connectivities'].shape[0])
        self.dim = int(max(np.sum(inf_ds.obsm['graph_neigh'],axis=1)))
        self.inf_ds, self.ref_ds = inf_ds, ref_ds 

        if(self.input_size is None):
            self.input_size = ref_ds.shape[1]
        if(self.number_of_class is None):
            self.number_of_class = len(ref_ds.obs["celltype"].value_counts())

        if not os.path.exists(self.save_dir): 
            os.makedirs(self.save_dir+"/")
        
        self.log = create_logger('',fh=self.save_dir+"/"+'log.txt')# create log file
        if(self.verbose):
            self.log.info("Create log file....") # write log information
            self.log.info("Create scJoint Object Done....") 

        self.training_iters =int(self.inf_ds.shape[0]/self.batch_size)
    
    def PrepareDataLoader_stage(self,ref_batch_size=256,inf_batch_size=256):
        num_workers = self.threads - 1
        if num_workers < 0:
            num_workers = 0
        print('num_workers:', num_workers)
        kwargs = {'num_workers': num_workers, 'pin_memory': False} # 0: one thread, 1: two threads ...
        random.seed(1)        

        self.ref_trainset = ReferenceDataSet(train=True,ref_data = self.ref_ds.X,labels = self.ref_ds.obs["celltype"].values.astype("float"))
        self.ref_trainloader = torch.utils.data.DataLoader(self.ref_trainset, batch_size=
                                ref_batch_size, shuffle=True,**kwargs)
        #self.iter_ref_trainloader = def_cycle(self.ref_trainloader)

        self.ref_testset = ReferenceDataSet(train=False, ref_data = self.ref_ds.X,labels = self.ref_ds.obs["celltype"].values)
        self.ref_testloader = torch.utils.data.DataLoader(self.ref_testset, batch_size = ref_batch_size, shuffle=False,**kwargs)      

        self.inf_trainset = InferenceDataSet_stage(train=True,inf_data = self.inf_ds.X,inf_neigh=self.inf_ds.obsm['graph_neigh'],dim=self.dim)
        self.inf_trainloader = torch.utils.data.DataLoader(self.inf_trainset, batch_size=
                                inf_batch_size, shuffle=True,**kwargs)
        #self.iter_inf_trainloader = def_cycle(self.inf_trainloader)
 
        self.inf_testset = InferenceDataSet_stage(train=False, inf_data = self.inf_ds.X,inf_neigh=self.inf_ds.obsm['graph_neigh'],dim=self.dim)
        self.inf_testloader = torch.utils.data.DataLoader(self.inf_testset, batch_size = inf_batch_size, shuffle=False,**kwargs)      
    
    def BuildNet_stage(self):
        self.model_encoder = Net_encoder(self.input_size).to(self.device)
        self.model_cell = Net_cell(self.number_of_class).to(self.device)

        # initialize criterion (loss)
        self.criterion_cell = CellLoss()
        self.criterion_encoding = EncodingLoss(dim=64, use_gpu = self.use_cuda)
        self.l1_regular = L1regularization()
        
        # initialize optimizer (sgd/momemtum/weight decay)
        self.optimizer_encoder = optim.SGD(self.model_encoder.parameters(), lr=self.lr, momentum=self.momentum,
                                           weight_decay=0)
        self.optimizer_cell = optim.SGD(self.model_cell.parameters(), lr=self.lr, momentum=self.momentum,
                                        weight_decay=0)
        

    def TrainingProcessStage(self,ckpt_path=None):

        start_time = time()
        self.loss = pd.DataFrame({'rna_reduction_loss':[0],'st_reduction_loss':[0],'sim_loss':[0],'loc_loss':[0],'encoding_loss':[0],'cell_loss':[0]})
        if(ckpt_path is None):
            for epoch in range(self.epochs):
                print('Epoch:', epoch)
                self.iter_ref_trainloader = def_cycle(self.ref_trainloader)
                self.iter_inf_trainloader = def_cycle(self.inf_trainloader)
                self.lossbatch = []
                for batch_idx in range(self.training_iters):
                    ref_data, ref_label = next(self.iter_ref_trainloader)
                    ref_data, ref_label = Variable(ref_data).to(self.device),Variable(ref_label).to(self.device)
                    ref_embedding = self.model_encoder(ref_data)
                    ref_cell_prediction = self.model_cell(ref_embedding)

                    inf_data_all, inf_neigh_all, ind = next(self.iter_inf_trainloader)
                    in_data = inf_data_all[:,0,:]
                    in_neigh = inf_neigh_all[:,0,:]
                    id = ind[:,0]
                    for i in range(1,self.dim):
                        in_data = torch.cat((in_data,inf_data_all[:,i,:]),dim=0)
                        in_neigh = torch.cat((in_neigh,inf_neigh_all[:,i,:]),dim=0)
                        id = torch.cat((id,ind[:,i]))
                    id = np.ravel(id)
                    ids = np_unranked_unique(id)
                    idx = []
                    for i in ids:
                        if i in id:
                            idx.append(list(id).index(i))
                        else:
                            idx.append(-1)
                    indices = np.where(ids == 9999)
                    idx = np.delete(idx, indices)
                    ids = np.delete(ids, indices)
                    idx = np.unique(idx)
                    inf_data = in_data[idx,:]
                    inf_neigh = in_neigh[idx,:]
                    inf_neigh = inf_neigh[:,ids]
                    inf_data, inf_neigh = Variable(inf_data).to(self.device), Variable(inf_neigh).to(self.device)

                    inf_embedding = self.model_encoder(inf_data)
                    inf_cell_prediction = self.model_cell(inf_embedding)

                    cell_loss = self.criterion_cell(ref_cell_prediction, ref_label)
                    rna_reduction_loss,st_reduction_loss,sim_loss,loc_loss,encoding_loss = self.criterion_encoding([inf_embedding], [ref_embedding],inf_neigh)
                    regularization_loss_encoder = self.l1_regular(self.model_encoder)
                    #self.loc_loss = loc_loss.tolist(）

                    # update encoding weights
                    self.optimizer_encoder.zero_grad()  
                    regularization_loss_encoder.backward(retain_graph=True)         
                    #cell_loss.backward(retain_graph=True)
                    encoding_loss.backward(retain_graph=True)            
                    #self.optimizer_encoder.step()
                    
                    regularization_loss_cell = self.l1_regular(self.model_cell)
                    # update cell weights
                    self.optimizer_cell.zero_grad()
                    cell_loss.backward(retain_graph=True)
                    regularization_loss_cell.backward(retain_graph=True) 
                    self.optimizer_encoder.step()
                    self.optimizer_cell.step()

                    print("encoder_loss = {},cell_loss={}.".format(encoding_loss.item(),cell_loss.item()))
                rna_reduction_loss = rna_reduction_loss.tolist()
                st_reduction_loss = st_reduction_loss.tolist()
                sim_loss = sim_loss.tolist()
                loc_loss = loc_loss.tolist()
                encoding_loss = encoding_loss.tolist()
                cell_loss = cell_loss.tolist()
                new_row = pd.DataFrame({'rna_reduction_loss':[rna_reduction_loss],'st_reduction_loss':[st_reduction_loss],'sim_loss':[sim_loss],'loc_loss':[loc_loss],'encoding_loss':[encoding_loss],'cell_loss':[cell_loss]})
                self.loss = pd.concat([self.loss,new_row])
            
                # save checkpoint
                save_checkpoint({
                    'epoch': epoch,
                    'model_cell_state_dict': self.model_cell.state_dict(),
                    'model_encoding_state_dict': self.model_encoder.state_dict(),
                    'optimizer': self.optimizer_cell.state_dict()            
                },save_dir= self.save_dir,filename = "stage1_checkpoint.pth.tar")
                
        else:
            if os.path.isfile(ckpt_path):
                print("=> loading checkpoint '{}'".format(ckpt_path))
                checkpoint = torch.load(ckpt_path)                
                self.model_encoder.load_state_dict(checkpoint['model_encoding_state_dict'])
                self.model_cell.load_state_dict(checkpoint['model_cell_state_dict'])


        end_time = time()
        print("trainging stage cost {}s".format(end_time-start_time))
    
        # write embeddings for traning stage1
        ref_emb,ref_cell_pred = self.predict_ref_stage(self.ref_testloader)
        self.ref_ds.obsm["emb"] =ref_emb
        self.ref_ds.obsm["cell_pred"]=ref_cell_pred

        inf_emb,inf_cell_pred = self.predict_inf_stage(self.inf_testloader)
        self.inf_ds.obsm["emb"] =inf_emb
        self.inf_ds.obsm["cell_pred"]=inf_cell_pred


    def predict_inf_stage(self,dataloader):
        self.model_encoder.eval()
        self.model_cell.eval()
        device=torch.device("cpu")    
        data_iterator = tqdm(dataloader, leave=False, unit="batch")
        self.model_encoder=self.model_encoder.to(device)
        self.model_cell = self.model_cell.to(device)

        inf_emb =[]
        inf_cell_pred=[]
        with torch.no_grad():
            for (inf_data,inf_neigh) in data_iterator:
                #inf_data = inf_data[:,1,:]
                inf_embedding = self.model_encoder(inf_data)
                inf_cell_prediction = self.model_cell(inf_embedding)
                            
                inf_embedding = inf_embedding.data.cpu().numpy()
                inf_cell_prediction = inf_cell_prediction.data.cpu().numpy()
                
                # normalization & softmax
                inf_embedding = inf_embedding / norm(inf_embedding, axis=1, keepdims=True)
                inf_cell_prediction = softmax(inf_cell_prediction, axis=1)
                inf_emb.append(inf_embedding)
                inf_cell_pred.append(inf_cell_prediction)
            inf_emb=np.concatenate(inf_emb)
            inf_cell_pred = np.concatenate(inf_cell_pred)
        return inf_emb,inf_cell_pred
    
    def predict_ref_stage(self,dataloader):
        self.model_encoder.eval()
        self.model_cell.eval()
        device=torch.device("cpu")    
        data_iterator = tqdm(dataloader, leave=False, unit="batch")
        self.model_encoder=self.model_encoder.to(device)
        self.model_cell = self.model_cell.to(device)

        ref_emb =[]
        ref_cell_pred=[]
        with torch.no_grad():
            for (ref_data, ref_label) in data_iterator:
                ref_data,ref_label = ref_data.to(device),ref_label.to(device)
                ref_embedding = self.model_encoder(ref_data)
                ref_cell_prediction = self.model_cell(ref_embedding)
                            
                ref_embedding = ref_embedding.data.cpu().numpy()
                ref_cell_prediction = ref_cell_prediction.data.cpu().numpy()
                
                # normalization & softmax
                ref_embedding = ref_embedding / norm(ref_embedding, axis=1, keepdims=True)
                ref_cell_prediction = softmax(ref_cell_prediction, axis=1)
                ref_emb.append(ref_embedding)
                ref_cell_pred.append(ref_cell_prediction)
            ref_emb=np.concatenate(ref_emb)
            ref_cell_pred = np.concatenate(ref_cell_pred)
        return ref_emb,ref_cell_pred

if __name__=='__main__':

    ref_adata = sc.read("./output/ref_adata.h5ad")
    inf_adata = sc.read("./output/inf_adata.h5ad")

    map_dict = dict(zip(ref_adata.obs["CellType"], ref_adata.obs["CellType"].cat.codes))
    ref_celltype_set= set(ref_adata.obs["CellType"].value_counts().index)

    scjoint = spaJoint(ref_ds= ref_adata,inf_ds= inf_adata)
    scjoint.PrepareDataLoader_stage()
    scjoint.BuildNet_stage()
    scjoint.TrainingProcessStage()
    
    celltype_density = pd.DataFrame(scjoint.inf_ds.obsm["cell_pred"])
    cell_type_dict = dict(zip(ref_adata.obs['celltype'],ref_adata.obs['CellType']))
    celltype_density.rename(columns=cell_type_dict,inplace=True)
    celltype_density.to_csv('./spajoint_result.txt')
    ## 

    # print("test done .....")