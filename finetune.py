import time
import numpy as np
import scipy.io as sci
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
import os
from dtw import dtw
from scipy.stats import linregress

from BHT_ARIMA import BHTTUCKER3, BHTTUCKER3_test, BHTTUCKER3_update
from model.AE import AutoEncoder
from model.DANN import dann
from model.LSTMModel import LSTM3, LSTMmodel
from model.tensor import MDT_inverst, get_core, get_Xs, loss_US
from util.dataNorm import datanorm


#%%数据加载
t = 8
data_block = 34

data_path = "G:/data/Bearings/"
ResultData_path = "G:/code_mission2/XJTU_UNSW1/result(xjtu_unsw)/"
SimuDatBaseBJPQ_path = 'G:/data/Simulaation/HHT/'
SimuDatBaseRMSQ_path = 'G:/data/Simulation/RMS/'

SimuDatBaseBJPQ = []
SimuDatBaseRMSQ = []
for filename in os.listdir(SimuDatBaseBJPQ_path):
    file_path = os.path.join(SimuDatBaseBJPQ_path,filename)
    SimuDatBaseBJPQ.append(sci.loadmat(file_path)['BJP'])

for filename in os.listdir(SimuDatBaseRMSQ_path):
    file_path = os.path.join(SimuDatBaseRMSQ_path,filename)
    SimuDatBaseRMSQ.append(sci.loadmat(file_path)['normolized_data'])


X1=sci.loadmat(data_path+'HHT/XJTU_HHT/BJP_XJTUB1_1.mat')['BJP'][1393-t:1476]   
X2=sci.loadmat(data_path+'HHT/XJTU_HHT/BJP_XJTUB1_2.mat')['BJP'][1748-t:1932]
X3=sci.loadmat(data_path+'HHT/XJTU_HHT/BJP_XJTUB1_3.mat')['BJP'][1758-t:1896]
X5=sci.loadmat(data_path+'HHT/XJTU_HHT/BJP_XJTUB1_5.mat')['BJP'][553-t:624]
#X4=sci.loadmat('G:/UNSW/UNSW-HHT/HHT_UNSW1.mat')['BJP'][505-t:960]

R1 = sci.loadmat(data_path+'RMS/XJTU_Data/RMS/XJTU1_RMS.mat')['XRMSB1_1'][1393:1476]
R2 = sci.loadmat(data_path+'RMS/XJTU_Data/RMS/XJTU1_RMS.mat')['XRMSB1_2'][1748:1932]
R3 = sci.loadmat(data_path+'RMS/XJTU_Data/RMS/XJTU1_RMS.mat')['XRMSB1_3'][1758:1896]
R5 = sci.loadmat(data_path+'RMS/XJTU_Data/RMS/XJTU1_RMS.mat')['XRMSB1_5'][553:624]
#R4 = sci.loadmat('G:/UNSW/UNSW1.mat')['rmsH'][505:960]


X6=sci.loadmat('G:/UNSW/UNSW-HHT/HHT_UNSW2.mat')['BJP'][1532-t:2004]  #目标域真实数据，用于测试
R6 = sci.loadmat('G:/UNSW/UNSW2.mat')['rmsH'][1532:2004]  #目标域RMS


S_index = [83,184,71]
V_index = [138]
T_index = [472]


dataS = [X1,X2,X5]
dataV = [X3]

dataT = [X6]

RMS_S = [R1,R2,R5,R3]
RMS_T = [R6]

Testt = X6


#转换成pytorch的浮点数张量
def To_tensor(dataX):
    return torch.tensor(dataX).float()

RMSS_set=[]
RMST_set=[]

for i in range(len(RMS_S)):
    RMSS_set.append(To_tensor(RMS_S[i]))
for i in range(len(RMS_T)):
    RMST_set.append(To_tensor(RMS_T[i]))

dataTrainS_list_all = []    
dataTrainV_list_all = []
dataTrainT_list_all = [] 
dataTrainSimulate_list_all = [] 

for i in range(len(dataS)):
    dataTrainS_list_all.append(To_tensor(dataS[i]))  
for i in range(len(dataT)):
    dataTrainT_list_all.append(To_tensor(dataT[i]))
for i in range(len(dataV)):
    dataTrainV_list_all.append(To_tensor(dataV[i]))
for i in range(len(SimuDatBaseBJPQ)):
    dataTrainSimulate_list_all.append(To_tensor(SimuDatBaseBJPQ[i]))


#构建数据成LSTM形式
def create_data_sample(dataX, time_step):   
    X = []
    for i in range(dataX.shape[0]-time_step):
        X.append(dataX[i:i+time_step])  
    return X


#构建RUL标签（真实标签）
def create_data_label(dataX, time_step): 
    y = np.linspace(1,0,len(dataX)-time_step)
    return (torch.tensor(y).float())


test_label = create_data_label(Testt[:],t)          #目标域真实数据标签

Y8_label = []
for i in range(len(SimuDatBaseBJPQ)):
    Y8_label.append(create_data_label(SimuDatBaseBJPQ[i],t))

# Y8_label = create_data_label(dataT[0][T_index_e[0]:],t)  #目标域模拟数据标签
def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))
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

#单调性
def Monotonicity(hx1,smt=2):
    dhdo=torch.zeros(1).to(device)
    dhxo=torch.zeros(1).to(device)
    hx_len=int(hx1.shape[0]/smt)
    hx_block=[hx1[i*smt:(i+1)*smt] for i in range(hx_len)]
    hx_smooth=[torch.mean(i) for i in hx_block]
    for i in range(len(hx_smooth)-1):
        if hx_smooth[i]<=hx_smooth[i+1]:
            dhdo=dhdo+1
        else:
            dhxo=dhxo+1
    Mon=torch.abs(dhdo-dhxo)/(len(hx_smooth)-1)
    return Mon
def cosine_similarity(vector1, vector2):
    # 将向量调整为一维数组
    vector1 = np.squeeze(vector1)
    vector2 = np.squeeze(vector2)
    
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

def growth_similarity(array1, array2):
    # 计算数组的一阶差分
    diff1 = np.diff(array1)
    diff2 = np.diff(array2)
    
    # 如果数组长度不同，则返回 None
    if len(diff1) != len(diff2):
        return None
    
    # 计算相关系数
    correlation_coefficient = np.corrcoef(diff1, diff2)[0, 1]
    
    # 计算线性回归斜率
    slope, _, _, _, _ = linregress(diff1, diff2)
    
    return correlation_coefficient, slope



#%% 模型参数
# Rs = [25, 8] # tucker分解等级
# k =  10 # 迭代
# tol = 0.001 # stop criterion
# Us_mode = 4 # orthogonality mode（正交模式）

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
tea_model1 = LSTMmodel(input_size,hidden_size,output_size,num_layers,dropout).to(device)
model2 = LSTM3().to(device)

optimizer = optim.Adam([{'params':autoencoder.parameters(),'lr':0.001,'weight_decay':0.00001},  #学习率（lr）、权重衰减（weight_decay）
                        {'params':model_domain.parameters(),'lr':0.001,'weight_decay':0.00001},
                        {'params':tea_model1.parameters(),'lr':0.001,'weight_decay':0.00001},
                        {'params':model2.parameters(),'lr':0.001,'weight_decay':0.00001}])

scheduler1 = optim.lr_scheduler.StepLR(optimizer,step_size=25,gamma=0.95)

criterion1 = nn.MSELoss()          #均方根误差
criterion2 = nn.CrossEntropyLoss() #交叉熵损失

#加载预训练模型参数
autoencoder.load_state_dict(torch.load(ResultData_path + 'pre/autoencoder21_150.pth'))
model_domain.load_state_dict(torch.load(ResultData_path + 'pre/model_domain21_150.pth'))
model2.load_state_dict(torch.load(ResultData_path + 'pre/model2_21_150.pth'))
load_model1 = torch.load(ResultData_path + 'pre/model1_21_150.pth')
tea_model1.load_state_dict(load_model1)

for i,p in enumerate(tea_model1.parameters()):   #冻结参数
    if i < 8:
        p.requires_grad = False
        
#%%  模型的测试部分！！！！！！！！
out_list = [1]
out_list1 = []
target_loss = []
def test(Testt):
    test_loss = 0
    with torch.no_grad():
        Testt = torch.tensor(Testt).float()       
        data1, target = Testt[:8+data_block*m],test_label[(m-1)*data_block:m*data_block]
        target = target.to(device)
        data1 = data1.to(device)
        optimizer.zero_grad()
        enc_t, dec_t = autoencoder(data1)
        fea_t = create_data_sample(enc_t,t)
        fea_test=torch.stack(fea_t, dim=0) 
        _,ts,_ = tea_model1(fea_test[(m-1)*data_block:m*data_block])
        test_output=model2(ts)
        out_list1.append(test_output.cpu()) 
        test_loss += float(criterion1(test_output.view(-1,1), target.view(-1,1)))
        target_loss.append(test_loss)  ##预测损失   
    print("测试的损失为：{}".format(test_loss))

#%%模型的训练部分
allepoch=80
train_out_list=[]
train_out_list1=[]
train_out_list_fd_12 = []
loss_list=[]
output_12=[]
hhh=[] 
out_list = [1]
start_time = time.time()
choice = []
length = []
for m in range(1,15):
    fea=[]
    for epoch in range(allepoch):
        i = 0
        autoencoder.to(device).train()
        model_domain.to(device).train()
        tea_model1.to(device).train()
        model2.to(device).train()
        hhhh=[]
        hhh=[] 
        hh=[]
        hh_s=[]
        hh_simulate=[]
        hh_t = []
        hh_v=[]       
        h_1=[]
        h_2=[]
        h_3 = []
        h_4 = []
        fea_loss=[]
        rms_s = []
        for batch_idx, (data_source) in enumerate(dataTrainS_list_all):
            p = float(epoch) / allepoch
            alpha = 2. / (1. + np.exp(-10. * p)) - 1
            
            temp1 = []
            temp11 = []
            temp2 = []
            temp22 = []
            temp = []
            index_temp = []

            if m == 14 and len(rms_t) < data_block * m:
                rms_t = R6[data_block * (m-1):len(R6)]
                rms_t_min = np.min(rms_t)
                rms_t_max = np.max(rms_t)
                rms_t_normalized = (rms_t - rms_t_min)/(rms_t_max - rms_t_min)
                for index, rms_s in enumerate(SimuDatBaseRMSQ):
                    if len(rms_s) < len(R6):
                        continue
                    else:
                        rms_s = rms_s[data_block * (m-1):len(R6)]
                        # rms_slv = rms_s[:len(rms_t)]
                        # 使用统一的scaler进行归一化
                        rms_s_min = np.min(rms_s)
                        rms_s_max = np.max(rms_s)
                
                        rms_s_normalized = (rms_s-rms_s_min)/(rms_s_max-rms_s_min)
                        # rms_slv_normalized = scaler.fit_transform(rms_slv)
                        # 计算DTW距离
                        distance1 = dtw(rms_t_normalized, rms_s_normalized, dist=manhattan_distance)
                        temp1.append((index, distance1[0]))
                        
                        
                        # # 计算线性回归斜率差异
                        
                        rms_s_normalized_squeeze = np.squeeze(rms_s_normalized)
                        rms_t_normalized_squeeze = np.squeeze(rms_t_normalized)
            
                        slope1, _, _, _, _ = linregress(range(len(rms_s_normalized_squeeze)), rms_s_normalized_squeeze)
                        slope2, _, _, _, _ = linregress(range(len(rms_t_normalized_squeeze)), rms_t_normalized_squeeze)
                        distance2 = abs(slope1 - slope2)
                        temp2.append((index, distance2))                                                            
                                
            else :
                rms_t = R6[data_block * (m-1):data_block * m]
                rms_t_min = np.min(rms_t)
                rms_t_max = np.max(rms_t)
                rms_t_normalized = (rms_t - rms_t_min)/(rms_t_max - rms_t_min)

                for index, rms_s in enumerate(SimuDatBaseRMSQ):
                   
                    if len(rms_s) < data_block * m:
                        continue
                    else :
                        rms_s = rms_s[data_block * (m-1):data_block * m]
                        # rms_slv = rms_s[data_block * (m - 1):data_block * m]
                        # rms_sdtw_normalized = scaler.fit_transform(rms_sdtw).flatten()  # 使用 fit_transform 进行归一化
                        # rms_slv_normalized = scaler.fit_transform(rms_slv).flatten()
                        rms_s_min = np.min(rms_s)
                        rms_s_max = np.max(rms_s)
                
                        rms_s_normalized = (rms_s-rms_s_min)/(rms_s_max-rms_s_min)
                        # rms_slv_normalized = scaler.fit_transform(rms_slv)
                        rms_t_normalized = (rms_t-rms_t_min)/(rms_t_max-rms_t_min)
                        # 计算DTW距离
                        distance1 = dtw(rms_t_normalized, rms_s_normalized, dist=manhattan_distance)          
                        temp1.append((index, distance1[0]))

                        # 计算线性回归斜率差异
                        rms_s_normalized_squeeze = np.squeeze(rms_s_normalized)
                        rms_t_normalized_squeeze = np.squeeze(rms_t_normalized)
                        # rms_slv_normalized = np.squeeze(rms_slv_normalized)
                        # rms_tlv = np.squeeze(rms_tlv)
                        slope1, _, _, _, _ = linregress(range(len(rms_s_normalized_squeeze)), rms_s_normalized_squeeze)
                        slope2, _, _, _, _ = linregress(range(len(rms_t_normalized_squeeze)), rms_t_normalized_squeeze)
                        distance2 = abs(slope1 - slope2)
                        temp2.append((index, distance2))
                                                                                                
            for z in range(len(temp1)):
                temp11.append(temp1[z][1])
                temp22.append(temp2[z][1])
                index_temp.append(temp1[z][0])


            temp11 = (temp11 - min(temp11)) / (max(temp11) - min(temp11))
            temp22 = (temp22 - min(temp22)) / (max(temp22) - min(temp22))

            # for z in range(len(temp2)):
            #     index = temp2[z][0]
            #     distance2 = datanorm(temp2[z][1],temp1[z][1])
            #     temp22.append((index,distance2))

            for y in range(len(temp11)):
                index = index_temp[y]
                distance_sum = temp11[y] + temp22[y] 
                temp.append((index, distance_sum))
                
                
            min_distance_tuple = min(temp, key=lambda x: x[1])

                # 提取最小距离对应的索引
            min_distance_index = min_distance_tuple[0]
            dataTrainSimulate_list_all[min_distance_index]

            data_source = data_source.to(device)
            optimizer.zero_grad()                       #梯度归零
            encs, decs = autoencoder(data_source)       #AE
            h_1.append(encs)                            #之后保存数据
            loss_sae = criterion1(decs, data_source)    #均方根误差

            data_valid = dataTrainV_list_all[0].to(device)
            optimizer.zero_grad()
            encv, decv = autoencoder(data_valid)
            h_3.append(encv)
            loss_vae = criterion1(decv, data_valid)

            data_target = dataTrainT_list_all[0][:8+data_block*m].to(device) 
            optimizer.zero_grad()
            enct, dect = autoencoder(data_target)
            h_2.append(enct)
            loss_tae = criterion1(dect, data_target)

            data_Simulate  = dataTrainSimulate_list_all[min_distance_index][:8+data_block*m].to(device)
            optimizer.zero_grad()
            enc_Simulate, dec_Simulate = autoencoder(data_Simulate)
            h_4.append(enc_Simulate)
            loss_sim = criterion1(dec_Simulate, data_Simulate)

            loss_ae = loss_sae + loss_vae + loss_tae + loss_sim  #ae的总损失：源域四个、目标域的真实和仿真

            Xs = create_data_sample(encs,t)       #以时间步形式把数据从（83+100+8）* 50变成（83+100）* 8 * 50
            Xv = create_data_sample(encv,t)
            Xt = create_data_sample(enct,t)
            XSimulate = create_data_sample(enc_Simulate,t)

            Xs=torch.stack(Xs, dim=0)             #stack是对输入的张量进行堆叠，但这里输入只有一个list，就只是把数据张量化了
            Xv=torch.stack(Xv, dim=0)  
            Xt=torch.stack(Xt, dim=0) 
            XSimulate=torch.stack(XSimulate, dim=0)

            #提取核心张量
            Xsv = torch.cat((Xs, Xv, XSimulate), 0)    #在0维进行拼接：[1, 2, 3]和[4, 5, 6]变成[1, 2, 3, 4, 5, 6]
            # Xtv = torch.cat((Xtest,Xt),0)
            xsv = Xsv.permute(*torch.arange(Xsv.ndim - 1, -1, -1))  #翻转张量维度
            # xtv = Xtv.permute(*torch.arange(Xtv.ndim - 1, -1, -1))
            Us_list1 = torch.load(ResultData_path + 'pre/Us_list.pt')            
            model_tucker2 = BHTTUCKER3_update(xsv,Xt.T,Us_list1[i])  #给模型传参数
            coress, corest,_,_  = model_tucker2.run()                   #提取源域目标域的核心张量
            coress = get_core(coress)
            corest = get_core(corest)            
            X_S1 = coress[:len(Xs)][:,-1,:]
            X_V1 = coress[len(Xs):len(Xs)+len(Xv)][:,-1,:]
            X_Simulate = coress[len(Xs)+len(Xv):][:,-1,:]
            X_T1 = corest[:,-1,:]
            hh_s.append(X_S1)
            hh_simulate.append(X_Simulate) 
            hh_t.append(X_T1)
            hh_v.append(X_V1)
            
            #域对抗
            combined_image = torch.cat((coress,corest), 0)     #进去的是核心张量
            combined_image = combined_image.float().to(device)
            optimizer.zero_grad()
            _,fea_domain, domain_pred =  model_domain(combined_image,alpha)
            domain_source_labels = torch.zeros(coress.shape[0]).type(torch.LongTensor)
            domain_target_labels = torch.ones(corest.shape[0]).type(torch.LongTensor)

            domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0)
            domain_loss = criterion2(domain_pred, domain_combined_label.to(device))

            if m == 14 and len(Xt[(m-1)*data_block:m*data_block])<data_block:
                s_label1 = Y8_label[min_distance_index][(m-1)*data_block:(m-1)*data_block+len(Xt[(m-1)*data_block:m*data_block])]
            else:
                s_label1 = Y8_label[min_distance_index][(m-1)*data_block:m*data_block]

            # s_img1, s_label1 = Xt[100+(m-1)*data_block:100+m*data_block], Y8_label[min_distance_index][(m-1)*data_block:m*data_block] 
            s_img1 = Xt[(m-1)*data_block:m*data_block]
            s_img1= s_img1.to(device)      
            s_label1 = s_label1.to(device).view(-1,1)          #view是用来调整张量的形状的，-1代表自动调整，保证元素数量保持不变。
            optimizer.zero_grad()
            _,fs1,_ = tea_model1(s_img1)
            aa1=model2(fs1)
            train_out_list.append(aa1)
            err_s_label = criterion1(aa1.view(-1,1) , s_label1.to(device))  #模拟数据的预测值和它的RUL标签的均方差
            
            with torch.no_grad():
                X_T1 = X_T1.cpu().detach().numpy()   
                pca=PCA(n_components=1)
                pca.fit(X_T1)
                fea_s1=pca.transform(X_T1)
                fea_s1 = torch.Tensor(fea_s1).float().to(device)

                X_S1 = X_S1.cpu().detach().numpy()
                pca=PCA(n_components=1)
                pca.fit(X_S1)
                fea_ss1=pca.transform(X_S1)
                fea_ss1 = torch.Tensor(fea_ss1).float().to(device)
            t_RMS = RMST_set[0][:data_block*m].to(device)  #目标域真实数据RMS
            kl_t = mmd(t_RMS, fea_s1[:data_block*m])      #这里是目标域真实数据
            s_RMS = RMSS_set[i].to(device)
            kl_s = mmd(s_RMS, fea_ss1)             #s_RMS是源域的RMS    fea_ss1是源域的核心张量

            if m==1:
                fea_loss=0
                fea_loss = torch.tensor(fea_loss).float()
            elif ((m>=2) and (m<4)):
                fea_loss = (Monotonicity(fea_s1[data_block*(m-1):])-Monotonicity(fea_s1[data_block*(m-2):data_block*(m-1)])) #拿后一段减前一段，fea_loss越大越好
                #print("m:{},epoch:{}, fea_loss:{}".format(m,epoch, fea_loss.item()))
            else:
                fea_loss = (Monotonicity(fea_s1[data_block*(m-1):])-Monotonicity(fea_s1[data_block*(m-2):data_block*(m-1)]))
                #print("m:{},epoch:{}, fea_loss:{}".format(m, epoch, fea_loss.item()))
            
            #收集总误差
            err_all = 10*loss_ae+1*err_s_label + domain_loss - fea_loss+0.01*kl_s+0.1*kl_t
            err_all.backward()
            optimizer.step()
                    
            loss_list.append(err_all.cpu().item())
            print("m:{},epoch:{}, loss_ae:{}, train_all_loss:{}, err_s1:{}, d_loss:{}, fea_loss:{}".format(   #输出各种误差
                     m,epoch,loss_ae.item(),err_all.item(),err_s_label.item(),domain_loss.item(), fea_loss.item()))
            if batch_idx == 2:
                print("  ")

            i = i+1
    choice.append(min_distance_tuple)
    length.append(len(SimuDatBaseRMSQ[min_distance_index])) 
    print(choice)  
    print(length)  
    test(Testt)
    test_time = time.time() - start_time
    print("train_time:",test_time) 

    #%% 结果保存
    train_res = [i.cpu().detach() for i in train_out_list[-len(dataS):]]
    train_res = torch.cat(tuple(train_res),0).numpy()
    resultAll =  np.array(torch.cat(tuple(out_list1),dim=0))
    
    fea11 = hh_s[0].cpu().detach().numpy()
    fea21 = hh_s[1].cpu().detach().numpy()
    fea31 = hh_s[2].cpu().detach().numpy()
    fea_sim = hh_simulate[0].cpu().detach().numpy()
    fea51 = hh_v[0].cpu().detach().numpy()
    fea61 = hh_t[0].cpu().detach().numpy()

    fea111 = h_1[0].cpu().detach().numpy()
    fea211 = h_1[1].cpu().detach().numpy()
    fea311 = h_1[2].cpu().detach().numpy()
    fea_sim_ae = h_4[0].cpu().detach().numpy()
    fea511 = h_3[0].cpu().detach().numpy()    
    fea611 = h_2[0].cpu().detach().numpy()
    if m % 1 == 0:
        sci.savemat(ResultData_path + "finetune/No_{}".format(m)+"rul_B21.mat",{'finetune_out':resultAll,
                        'outTrain':train_res,
                        'fea11':fea11,'fea21':fea21,'fea31':fea31,'fea51':fea51,'fea61':fea61,'fea_sim':fea_sim,
                        'fea111':fea111,'fea211':fea211,'fea311':fea311,'fea511':fea511,'fea611':fea611,'fea_sim_ae':fea_sim_ae,
                        'train_loss':loss_list})

