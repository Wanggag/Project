# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 20:46:59 2021

@author: WJ
"""
import tensorly as tl
tl.set_backend('pytorch')
import torch

class BHTTUCKER3_update_online(object):
    def __init__(self,X1,X2, X3,Us):
        """store all parameters in the class and do checking on taus"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._trans_data1 = X1
        self._trans_data2 = X2
        self._trans_data3 = X3
        self._Us = Us         
    
    def _get_Xs(self, trans_data):

        T_hat = trans_data.shape[-1]
        Xs = [ trans_data[..., t] for t in range(T_hat)]

        return Xs    
    def _get_cores(self, Xs, Us):
        cores = [ tl.tenalg.multi_mode_dot( x, [u.T for u in Us], modes=[i for i in range(len(Us))] ) for x in Xs]

        return cores
        
    
    def run(self):
        
        coress1, coress2,corest ,X_S,X_T   = self._run()
                  
        return  coress1, coress2,corest,X_S,X_T
    
    def _run(self):
        # trans_data1, mdt1 = self._forward_MDT(self._ts1, self._taus)
        # trans_data2, mdt2 = self._forward_MDT(self._ts2, self._taus)
        
        trans_data1=self._trans_data1
        trans_data2=self._trans_data2
        trans_data3=self._trans_data3

        X_source1=torch.cat([trans_data1],axis=2) # 在dim=2 处拼接
        X_source2=torch.cat([trans_data2],axis=2) # 在dim=2 处拼接
        X_target=torch.cat([trans_data3],axis=2) # 在dim=2 处拼接
        trans_data=torch.cat([X_source1,X_source2,X_target],axis=2) # 在dim=2 处拼接   

#将三维数组变成列表形式        
        Xss1 = self._get_Xs(X_source1)
        Xss2 = self._get_Xs(X_source2)
        Xst = self._get_Xs(X_target)

#得到更新后的源域和目标域的核张量             
        coress1 = self._get_cores(Xss1, self._Us)
        coress2 = self._get_cores(Xss2, self._Us)
        corest = self._get_cores(Xst, self._Us)
       
        coress11 = torch.cat([torch.unsqueeze(i,0) for i in coress1], dim=0)
        coress21 = torch.cat([torch.unsqueeze(i,0) for i in coress2], dim=0)
        corest1 = torch.cat([torch.unsqueeze(i,0) for i in corest], dim=0)

        coress11 = coress11.permute(1,2,0)
        coress21 = coress21.permute(1,2,0)
        corest1 = corest1.permute(1,2,0)

        X_S = tl.tenalg.multi_mode_dot(coress11, self._Us)
        X_T = tl.tenalg.multi_mode_dot(corest1, self._Us)

      
        return coress1, coress2,corest,X_S,X_T