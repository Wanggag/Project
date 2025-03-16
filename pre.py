import time

import numpy as np
import scipy.io as sci
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from fvcore.nn import FlopCountAnalysis, parameter_count, parameter_count_table
from torch.autograd import Function

from BHT_ARIMA import BHTTUCKER3, BHTTUCKER3_update
from model.AE import AutoEncoder
from model.DANN import dann
from model.LSTMModel import LSTM3, LSTMmodel
from model.tensor import MDT_inverst, get_core, get_Xs, loss_US
from util.dataNorm import datanorm


#%%数据加载
t =8

data_path = "G:/data/Bearings/"
ResultData_path = "G:/code_mission2/XJTU_UNSW1/result(xjtu_unsw)/pre/"


X1=sci.loadmat(data_path+'HHT/XJTU_HHT/BJP_XJTUB1_1.mat')['BJP'][1393-100-t:1476]   
X2=sci.loadmat(data_path+'HHT/XJTU_HHT/BJP_XJTUB1_2.mat')['BJP'][1748-100-t:1932]
X3=sci.loadmat(data_path+'HHT/XJTU_HHT/BJP_XJTUB1_3.mat')['BJP'][1758-100-t:1896]
X5=sci.loadmat(data_path+'HHT/XJTU_HHT/BJP_XJTUB1_5.mat')['BJP'][553-100-t:624]
#X4=sci.loadmat('G:/UNSW/UNSW-HHT/HHT_UNSW1.mat')['BJP'][505-100-t:960]

X6=sci.loadmat('G:/UNSW/UNSW-HHT/HHT_UNSW2.mat')['BJP'][1532-100-t:2004]
 
R1 = sci.loadmat(data_path+'RMS/XJTU_Data/RMS/XJTU1_RMS.mat')['XRMSB1_1'][1393-100:1476]
R2 = sci.loadmat(data_path+'RMS/XJTU_Data/RMS/XJTU1_RMS.mat')['XRMSB1_2'][1748-100:1932]
R3 = sci.loadmat(data_path+'RMS/XJTU_Data/RMS/XJTU1_RMS.mat')['XRMSB1_3'][1758-100:1896]
R5 = sci.loadmat(data_path+'RMS/XJTU_Data/RMS/XJTU1_RMS.mat')['XRMSB1_5'][553-100:624]
#R4 = sci.loadmat('G:/UNSW/UNSW1.mat')['rmsH'][505-100:960]


S_index_e = [100,100,100]
V_index_e = [100]
T_index_e = [100]

S_index = [83,184,71]
V_index = [138]
# S_index = [459,1136,71]
# T_index = [41]
# S_index = [51,51,88]
T_index = [472]

dataS = [X1,X2,X5]
dataV = [X3]
dataT = [X6]
RMS_S = [R1,R2,R5,R3]

RMSS_set=[]
def create_RMS(dataX):  
    return torch.tensor(dataX).float()

for i in range(len(RMS_S)):
    RMSS_set.append(create_RMS(RMS_S[i]))

dataTrainS_list_all = []  
dataTrainT_list_all = []  
dataTrainV_list_all = []  

for i in range(len(dataS)):
    dataTrainS_list_all.append(create_RMS(dataS[i]))  
for i in range(len(dataT)):
    dataTrainT_list_all.append(create_RMS(dataT[i]))
for i in range(len(dataV)):
    dataTrainV_list_all.append(create_RMS(dataV[i]))
#构建数据成LSTM形式
def create_data_sample(dataX, time_step):   
    X = []
    for i in range(dataX.shape[0]-time_step):
        X.append(dataX[i:i+time_step])  
    return X

#构建RUL标签
def create_data_label(dataX, time_step): 
    y = np.linspace(1,0,len(dataX)-time_step)
    return (torch.tensor(y).float())

dataTrainS_list_label = []                    
dataTrainT_list_label = [] 
dataTrainV_list_label = [] 
for i in range(len(dataS)):
    dataTrainS_list_label.append(create_data_label(dataS[i][S_index_e[i]:],t))  
for i in range(len(dataT)):
    dataTrainT_list_label.append(create_data_label(dataT[i][T_index_e[i]:],t))
data_Test = dataTrainT_list_label[0]
for i in range(len(dataV)):
    dataTrainV_list_label.append(create_data_label(dataV[i][V_index_e[i]:],t))

def mmd(x,y):         
    return torch.sum(torch.pow((torch.mean(x,dim = 0) - torch.mean(y,dim = 0)),2))

def _get_Xs(trans_data):
    T_hat = trans_data.shape[-1]
    Xs = [ trans_data[..., t] for t in range(T_hat)]
    return Xs 

def get_core(x):
    x = torch.cat([torch.unsqueeze(i,0) for i in x], dim=0)
    x = x.permute(0,2,1).to(device)
    return x

def loss_US(x):
    return torch.sum(torch.pow(x, 2))

def flops_params_fps(model1, input_shape=(1079,8,50)):
    with torch.no_grad():
        model1 = model1.cuda().eval()
        input = torch.randn(size=input_shape, dtype=torch.float32).cuda()
        flops = FlopCountAnalysis(model1, input)
        params = parameter_count(model1)
        print(model1.__class__.__name__)
        # print('img/s:{:.2f}'.format(1 / mean_time))
        print('flops:{:.2f}G params:{:.2f}M'.format(flops.total() / 1e9, params[''] / 1e6))

def flops_params_fps(model1, input_shape=(1079,8,50)):
    with torch.no_grad():
        model1 = model1.cuda().eval()
        input = torch.randn(size=input_shape, dtype=torch.float32).cuda()
        flops = FlopCountAnalysis(model1, input)
        params = parameter_count(model1)
        print(model1.__class__.__name__)
        # print('img/s:{:.2f}'.format(1 / mean_time))
        print('flops:{:.2f}G params:{:.2f}M'.format(flops.total() / 1e9, params[''] / 1e6))
def flops_params_fps2(model1, input_shape=(1511,2558)):
    with torch.no_grad():
        model1 = model1.cuda().eval()
        input = torch.randn(size=input_shape, dtype=torch.float32).cuda()
        flops = FlopCountAnalysis(model1, input)
        params = parameter_count(model1)
        print(model1.__class__.__name__)
        # print('img/s:{:.2f}'.format(1 / mean_time))
        print('flops_ae:{:.2f}G params_ae:{:.2f}M'.format(flops.total() / 1e9, params[''] / 1e6))
def flops_params_fps1(model1, input_shape=(1479,8,25)):
    with torch.no_grad():
        model1 = model1.cuda().eval()
        input = torch.randn(size=input_shape, dtype=torch.float32).cuda()
        flops = FlopCountAnalysis(model1, input)
        params = parameter_count(model1)
        print(model1.__class__.__name__)
        # print('img/s:{:.2f}'.format(1 / mean_time))
        print('flops_ae:{:.2f}G params_ae:{:.2f}M'.format(flops.total() / 1e9, params[''] / 1e6))


#%% 模型参数
Rs = [25, 8] # tucker decomposition ranks
k =  10 # iterations
tol = 0.001 # stop criterion
Us_mode = 4 # orthogonality mode
input_size1 = 25
input_size = 50
hidden_size = 100 #该维度指的是lstm的特征输出维度，进入fc+relu
num_layers = 3
dropout = 0.2 
output_size =1

cuda =torch.cuda.is_available()
if cuda:
    device='cuda'
else:
    device='cpu'

autoencoder = AutoEncoder(2558,[1024,512,128,50]).to(device)    
model_domain = dann(input_size1, hidden_size, num_layers,dropout).to(device)
model1 = LSTMmodel(input_size,hidden_size,output_size,num_layers,dropout).to(device)
model2 = LSTM3().to(device)

optimizer = optim.Adam([{'params':autoencoder.parameters(),'lr':0.001,'weight_decay':0.000001},
                        {'params':model_domain.parameters(),'lr':0.001,'weight_decay':0.000001},
                        {'params':model1.parameters(),'lr':0.001,'weight_decay':0.000001},
                        {'params':model2.parameters(),'lr':0.001,'weight_decay':0.000001},])
scheduler1 = optim.lr_scheduler.StepLR(optimizer,step_size=25,gamma=0.95)

criterion1 = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss()

#%%  模型的测试部分！！！！！！！！
target_loss = []
out_list1=[]
out_list_raw = []
data_block = 34
def test():
    model1.to(device).eval()
    out_list2 = []
    out_list3 = []
    out_list4 = []
    for m in range(1,15): 
        model1.to(device).eval()
        test_loss = 0
        with torch.no_grad():
            data1, target = dataTrainT_list_all[0],data_Test[:m*data_block]
            target = target.to(device)
            data1 = data1.to(device)
            optimizer.zero_grad()

            enc_t, dec_t = autoencoder(data1)
            with torch.no_grad():
                enc_t1=datanorm(enc_t.cpu().detach().numpy(),enc_t.cpu().detach().numpy())
                enc_t1=torch.tensor(enc_t1).float().to(device) 
            fea_t = create_data_sample(enc_t,t)
            fea_test=torch.stack(fea_t, dim=0)               
            _,ts,_ = model1(fea_test[100:100+m*data_block])   #LSTM
            test_output=model2(ts)
            out_list2.append(test_output.cpu().detach().numpy())
            # 修改
            if(m==14):
                for i in range(len(out_list2)):
                    if i == len(out_list2) - 1:
                        out_list3.append(out_list2[i][i * data_block:])
                    else:
                        out_list3.append(out_list2[i][i * data_block:(i+1)*data_block])  
                for i in range (len(out_list3)):
                    for j in range(len(out_list3[i])):
                        out_list4.append(out_list3[i][j])
                # out_list4 += out_list3[0]+out_list3[1]+out_list3[2]+out_list3[3]+out_list3[4]  
                out_list1.append(out_list4)                      
                out_list_raw.append(out_list2[4]) 

            test_loss += float(criterion1(test_output.view(-1,1), target.view(-1,1)))
            target_loss.append(test_loss)  ##预测损失   
            print("测试的损失为：{}".format(test_loss))
 
#%%  模型的验证部分！！！！！！！！
valid_list = [1]
valid_list1 = [1]
valid_loss = []
Us_list_valid=[]
h_3v=[1]
def valid():
    model1.to(device).eval()
    valid_test_loss = 0
    with torch.no_grad():

        
        data2, valid = dataTrainV_list_all[0],dataTrainV_list_label[0]
        valid = valid.to(device)
        data2 = data2.to(device)
        optimizer.zero_grad()
        enc_v, dec_v = autoencoder(data2)

        h_3v[0]= enc_v 
        with torch.no_grad():
            enc_v=datanorm(enc_v.cpu().detach().numpy(),enc_v.cpu().detach().numpy())
            enc_v=torch.tensor(enc_v).float().to(device) 
        fea_v = create_data_sample(enc_v,t)
        fea_valid=torch.stack(fea_v, dim=0)        #张量分解        
        _,vs,_ = model1(fea_valid[100:])
        valid_output=model2(vs)
        # out_list[0]= output   #预测结果
        valid_list1[0]= valid_output   #预测结果
        valid_test_loss += float(criterion1(valid_output.view(-1,1), valid.view(-1,1)))
        valid_loss.append(valid_test_loss)  ##预测损失   
    print("验证的损失为：{}".format(valid_test_loss))

#%%模型的训练部分
start_time = time.time()
allepoch=150   #800
train_out_list = []
train_out_list1 = []
loss_list = []
Us_list=[[],[],[]]
for epoch in range(allepoch):
    i = 0
    autoencoder.to(device).train()
    model_domain.to(device).train()
    model1.to(device).train()
    model2.to(device).train()
    hhhh=[]
    hhh=[] 
    hh=[]
    hh_s=[]
    hh_t=[]
    hh_v=[]

    # h_1存放编码后的特征     
    h_1=[]          
    h_2=[]
    h_3 = []
    for batch_idx, (data_source, data_target) in enumerate(zip(dataTrainS_list_all,dataTrainT_list_all*len(dataTrainS_list_all))):
        p = float(epoch) / allepoch
        alpha = 2. / (1. + np.exp(-10. * p)) - 1
        #提取特征   
        data_source = data_source.to(device)
        optimizer.zero_grad()
        encs, decs = autoencoder(data_source)
        h_1.append(encs)
        loss_sae = criterion1(decs, data_source)  

        data_valid = dataTrainV_list_all[0][:108].to(device)
        optimizer.zero_grad()
        encv, decv = autoencoder(data_valid)
        h_3.append(encv)
        loss_vae = criterion1(decv, data_valid)

        data_target = dataTrainT_list_all[0][:108].to(device)
        optimizer.zero_grad()
        enct, dect = autoencoder(data_target)
        h_2.append(enct)
        loss_tae = criterion1(dect, data_target) 
        lossae = loss_sae+loss_vae+loss_tae
        Xs = create_data_sample(encs,t)
        Xv = create_data_sample(encv,t)
        Xt = create_data_sample(enct,t)
        Xs=torch.stack(Xs, dim=0)          #拼接、增加新的维度
        Xv=torch.stack(Xv, dim=0)        
        Xt=torch.stack(Xt, dim=0)        

        #源域回归器预测的训练
        s_img, s_label = Xs[-S_index[i]:],dataTrainS_list_label[i]
        s_img= s_img.to(device)      
        s_label = s_label.to(device).view(-1,1)          
        optimizer.zero_grad()
        _,fs,_ = model1(s_img)
        aa=model2(fs)
        train_out_list1.append(aa)
        err_s_label1 = criterion1(aa.view(-1,1) , s_label)


        Xsv = torch.cat((Xs, Xv), 0) 
        #张量分解        
        if ((epoch==0) or (epoch%50==0)):  
            
            model_tucker = BHTTUCKER3(Xsv.T,Xt.T,Rs, k, tol, verbose=0, Us_mode=Us_mode)
            Us,coresv, coret,ori_Xs,ori_Xt,_  = model_tucker.run()            
            Us1 = [i.cpu().detach().numpy() for i in Us]      
            Us2 = [torch.tensor(i).to(device) for i in Us1]
            Us_list[i] = Us2    
        else:                
            model_tucker2 = BHTTUCKER3_update(Xsv.T,Xt.T,Us_list[i])
            coresv, coret,ori_Xs,ori_Xt  = model_tucker2.run() 
                
        coresv = get_core(coresv)
        coret = get_core(coret)
        # X_S = tl.tenalg.multi_mode_dot(coress, Us2)
        # X_T = tl.tenalg.multi_mode_dot(coress, Us2)
        
        torch.save(Us_list, ResultData_path + 'Us_list.pt')
        loss_U1 = loss_US(Us_list[i][0])
        loss_U2 = loss_US(Us_list[i][1])

        X_S1 = coresv[:len(Xs)][:,-1,:]
        X_V1 = coresv[len(Xs):][:,-1,:]
        X_T1 = coret[:,-1,:]
        hh_s.append(X_S1)
        hh_t.append(X_T1)
        hh_v.append(X_V1)

        #源域数据和目标域早期故障数据对抗
        combined_image = torch.cat((coresv,coret), 0) 
        combined_image = combined_image.float().to(device)
        optimizer.zero_grad()
        _,fea_domain,domain_pred =  model_domain(combined_image,alpha)
        domain_source_labels = torch.zeros(coresv.shape[0]).type(torch.LongTensor)
        domain_target_labels = torch.ones(coret.shape[0]).type(torch.LongTensor)
        domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0)
        domain_loss = criterion2(domain_pred, domain_combined_label.to(device))
        #MMD损失
        hn1 = fea_domain[:len(Xs)].to(device)
        hhhh.append(hn1)
        hn3 = fea_domain[len(Xs):len(Xs)+len(Xv)].to(device) 
        hh.append(hn3)
        hn2 = fea_domain[len(Xs)+len(Xv):].to(device) 
        hhh.append(hn2)
        # lossmmd_all = mmd(hn1,hn2)+mmd(hn1,hn3)+mmd(hn2,hn3)

        # with torch.no_grad():
        #     X_S1 = X_S1.cpu().detach().numpy()
        #     pca=PCA(n_components=1)
        #     pca.fit(X_S1)
        #     fea_s1=pca.transform(X_S1)
        #     fea_s1 = torch.Tensor(fea_s1).float().to(device)
        fea_s1 = torch.mean(X_S1,dim=1).view(-1,1)
        s_RMS = RMSS_set[i].to(device)

        kl_s = mmd(s_RMS, fea_s1)


        #收集总误差
        err_all = 10*lossae+ 1*err_s_label1 +1*kl_s+5*domain_loss+0.001*loss_U1+0.001*loss_U2
        err_all.backward()
        # nn.utils.clip_grad_norm(model.parameters(), max_norm = 20, norm_type=2)
        optimizer.step()
                
        loss_list.append(err_all.cpu().item())
        print("epoch:{}, lossae:{},train_all_loss:{}, err_s1:{},d_loss:{},loss_U1:{},loss_U2:{}".format(epoch, 
                            lossae.item(),err_all.item(),err_s_label1.item(),domain_loss.item(),loss_U1.item(),loss_U2.item()))

        # flops = FlopCountAnalysis(model1, (s_img,))
        # print("FLOPs: ", flops.total())
        # flops = FlopCountAnalysis(model_domain, ((combined_image,alpha),))
        # print("FLOPs_domain: ", flops.total())
        # flops = FlopCountAnalysis(autoencoder, (combined_image1,))
        # print("FLOPs_ae: ", flops.total())

        # print('flops:{:.2f}G params:{:.2f}M'.format(flops.total() / 1e9, params[''] / 1e6))
        # print('flops:{:.2f}G'.format(flops.total() / 1e9))
        i = i+1

        # print(FlopCountAnalysis(autoencoder, combined_image1).by_module())
        # flops_params_fps(model1, s_img)
    test()
    valid()
    
    test_time = time.time() - start_time
    print("test_time:",test_time)
    #%% 结果保存
        # resultAll = out_list[0].cpu().numpy()   #测试结果，目标域无标签数据的预测结果
        # resultAll1 = out_list1[0].cpu().numpy()   #测试结果，目标域无标签数据的预测结果
    
    
    resultAll1 =  np.array(out_list1)
    resultAll1_raw = np.array(out_list_raw)

    if epoch == 149:
        tmp = resultAll1[-1:, :, 0]
        pre_out = tmp.T 
        sci.savemat(ResultData_path + 'pre_out.mat', {'pre_out': pre_out})
    # if epoch == 149:
    #     tmp = resultAll1_raw[-1:, :, 0]
    #     pre_out_raw = tmp.T #取最后一行作为预训练的伪标签
    #     sci.savemat(ResultData_path + 'pre_out_raw.mat', {'pre_out_raw': pre_out_raw})

    resultAll2 = valid_list1[0].cpu().numpy()   #测试结果，目标域无标签数据的预测结果
    # train_res = [i.cpu().detach() for i in train_out_list[-len(dataS):]] #源域七个轴承的训练结果
    # train_res = torch.cat(tuple(train_res),0).numpy()
    train_res1 = [i.cpu().detach() for i in train_out_list1[-len(dataS):]] #源域七个轴承的训练结果
    train_res1 = torch.cat(tuple(train_res1),0).numpy()
    h_3v1 = h_3v[0].cpu().numpy()
    fea1 = hhhh[0].cpu().detach().numpy()
    fea2 = hhhh[1].cpu().detach().numpy()
    fea3 = hhhh[2].cpu().detach().numpy()
    
    fea5 = hh[0].cpu().detach().numpy()
    fea6 = hhh[0].cpu().detach().numpy()

    fea11 = hh_s[0].cpu().detach().numpy()
    fea21 = hh_s[1].cpu().detach().numpy()
    fea31 = hh_s[2].cpu().detach().numpy()
    
    fea51 = hh_v[0].cpu().detach().numpy()
    fea61 = hh_t[0].cpu().detach().numpy()

    fea111 = h_1[0].cpu().detach().numpy()
    fea211 = h_1[1].cpu().detach().numpy()
    fea311 = h_1[2].cpu().detach().numpy()
    
    fea511 = h_3[0].cpu().detach().numpy()    
    fea611 = h_2[0].cpu().detach().numpy()
    if (epoch+1) % 50 == 0:
        sci.savemat(ResultData_path + "No_{}".format(epoch+1)+"B21_pre1.mat",{'out':resultAll1,
                                'fea1':fea1,'fea2':fea2,'fea3':fea3,'fea5':fea5,'fea6':fea6,'outTrain1':train_res1,
                                'valid2':resultAll2,
                                'h_3v1':h_3v1,
                                'fea11':fea11,'fea21':fea21,'fea31':fea31,'fea51':fea51,
                                'fea111':fea111,'fea211':fea211,'fea311':fea311,'fea511':fea511,'fea611':fea611,
                                })
        torch.save(autoencoder.state_dict(), ResultData_path + 'autoencoder21_{}.pth'.format((epoch+1)))
        torch.save(model_domain.state_dict(), ResultData_path + 'model_domain21_{}.pth'.format((epoch+1)))
        torch.save(model1.state_dict(), ResultData_path + 'model1_21_{}.pth'.format((epoch+1)))
        torch.save(model2.state_dict(), ResultData_path + 'model2_21_{}.pth'.format((epoch+1)))

 
# print(parameter_count_table(model1))
# print(parameter_count_table(model_domain))
# print(parameter_count_table(autoencoder))
# flops_params_fps(model1)
# # flops_params_fps1(model_domain)
# flops_params_fps2(autoencoder)

# torch.save(model.state_dict(),'model_par_cnn21.pth')
# for parameters in model.parameters():
#     print(parameters)
# model_dict = model.state_dict()
# for k,v in model_dict.items():
#     print(k)        
# model_dict1 = model_extract.state_dict()
# for k,v in model_dict1.items():
#     print(k)         
            
        
   