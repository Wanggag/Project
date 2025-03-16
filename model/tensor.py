
import torch

cuda =torch.cuda.is_available()
if cuda:
    device='cuda'
else:
    device='cpu'

def get_Xs(trans_data):
    T_hat = trans_data.shape[-1]
    Xs = [ trans_data[..., t] for t in range(T_hat)]
    return Xs 

def get_core(x):
    x = torch.cat([torch.unsqueeze(i,0) for i in x], dim=0)
    x = x.permute(0,2,1).to(device)
    return x

def loss_US(x):
    return torch.sum(torch.pow(x, 2))

def MDT_inverst(tucker_feature):
    sampe_num,time_step,fea_dim = tucker_feature.shape
    mdt_inv = [tucker_feature[0,:,:]]
    mdt_inv = mdt_inv + [tucker_feature[i:i+1,-1,:] for i in range(sampe_num)] 
    return torch.cat(mdt_inv, dim=0)