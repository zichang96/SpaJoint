import sys
import time
import os
import scanpy as sc
import torch
import torch
import torch.utils.data as data
import numpy as np
import os
import os.path
import random

last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    """Progress Bar for display
    """
    def _format_time(seconds):
        days = int(seconds / 3600/24)
        seconds = seconds - days*3600*24
        hours = int(seconds / 3600)
        seconds = seconds - hours*3600
        minutes = int(seconds / 60)
        seconds = seconds - minutes*60
        secondsf = int(seconds)
        seconds = seconds - secondsf
        millis = int(seconds*1000)

        f = ''
        i = 1
        if days > 0:
            f += str(days) + 'D'
            i += 1
        if hours > 0 and i <= 2:
            f += str(hours) + 'h'
            i += 1
        if minutes > 0 and i <= 2:
            f += str(minutes) + 'm'
            i += 1
        if secondsf > 0 and i <= 2:
            f += str(secondsf) + 's'
            i += 1
        if millis > 0 and i <= 2:
            f += str(millis) + 'ms'
            i += 1
        if f == '':
            f = '0ms'
        return f

    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
    TOTAL_BAR_LENGTH = 30.
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()    # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('    Step: %s' % _format_time(step_time))
    L.append(' | Tot: %s' % _format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


# refer to https://github.com/xternalz/WideResNet-pytorch
def save_checkpoint(state,save_dir=None, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = save_dir+"/ckpt/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)


def def_cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class ReferenceDataSet(data.Dataset):
    def __init__(self, train = True, ref_data =None, labels = None):
        self.train = train        
        self.ref_data,self.labels = ref_data,labels
        self.input_size = self.ref_data.shape[1]
        self.sample_num = self.ref_data.shape[0]
        
    def __getitem__(self, index):
        if self.train: # random select sample
            # get sc data            
            rand_idx = random.randint(0, self.sample_num - 1)
            in_data = np.array(self.ref_data[rand_idx].todense()) #.todense()
            #in_data = in_data.reshape((1, self.input_size))
            in_label = self.labels[rand_idx]
            return in_data, in_label

        else: # select sample by sequence
            in_data = np.array(self.ref_data[index].todense())    #.todense()       
            #in_data = in_data.reshape((1, self.input_size))
            in_label = self.labels[index]
 
            return in_data, in_label

    def __len__(self):
        return self.sample_num
    

class InferenceDataSet_stage(data.Dataset):
    def __init__(self, train = True, inf_data = None,inf_neigh = None, dim=0):
        self.train = train
        self.inf_data = inf_data
        self.inf_neigh = inf_neigh
        self.dim = dim
        self.input_size = self.inf_data.shape[1]
        self.sample_num = self.inf_data.shape[0]
    
                    
    def __getitem__(self, index):
        if self.train:
            # get atac data
            rand_idx = random.randint(0, self.sample_num - 1) #每次选一个
            in_neigh = np.array(self.inf_neigh[rand_idx]).squeeze()
            idx = np.array(np.nonzero(in_neigh)).reshape(-1)
            indices = np.where(idx==rand_idx)
            idx = np.delete(idx,indices)
            ind = np.append(np.array(rand_idx),idx)
            in_data = np.array(self.inf_data[ind].todense()) #.todense()
            #in_data = in_data.reshape((1, self.input_size))
            in_neigh = np.array(self.inf_neigh[ind]).squeeze()#控制向量长度，选完之后再裁剪
            length = self.dim-in_data.shape[0]
            in_data = np.pad(in_data,((0,length),(0,0)),'constant',constant_values = (0,0)) 
            if in_neigh.ndim == 1:
                in_neigh = np.pad(in_neigh[np.newaxis,:],((0,length),(0,0)),'constant',constant_values = (0,0))
            else:
                in_neigh = np.pad(in_neigh,((0,length),(0,0)),'constant',constant_values = (0,0))
            ind = np.pad(ind,(0,length),'constant',constant_values=(0,9999))
            
            return in_data,in_neigh,ind

        else:
            in_data = np.array(self.inf_data[index].todense())#.todense()
            #in_data = in_data.reshape((1, self.input_size))
            in_neigh = np.array(self.inf_neigh[index]).squeeze()
                                      
            return in_data, in_neigh

    def __len__(self):
        return self.sample_num
    
##### set random seed for reproduce result ##### 
def seed_torch(seed=4321):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.badatahmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    
def filter_with_overlap_gene(adata, adata_sc):
    #remove all-zero-valued genes
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.filter_genes(adata_sc, min_cells=1)
    
    if 'highly_variable' not in adata.var.keys():
       raise ValueError("'highly_variable' are not existed in adata!")
    else:    
       adata = adata[:, adata.var['highly_variable']]
       
    if 'highly_variable' not in adata_sc.var.keys():
       raise ValueError("'highly_variable' are not existed in adata_sc!")
    else:    
       adata_sc = adata_sc[:, adata_sc.var['highly_variable']]
    #Refine `marker_genes` so that they are shared by both adatas
    genes = list(set(adata.var.index) & set(adata_sc.var.index))
    genes.sort()
    print('Number of overlap genes:', len(genes))

    adata.uns["overlap_genes"] = genes
    adata_sc.uns["overlap_genes"] = genes
    
    adata = adata[:, genes].copy()
    adata_sc = adata_sc[:, genes].copy()
    
    return adata, adata_sc

def preprocess(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata) #check data (gene counts)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)
    return adata
    
def np_unranked_unique(nparray):
    n_unique = len(np.unique(nparray))
    ranked_unique = np.zeros([n_unique])
    i = 0
    for x in nparray:
        if x not in ranked_unique:
            ranked_unique[i] = x
            i += 1
    return ranked_unique

