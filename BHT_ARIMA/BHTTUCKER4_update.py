# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 20:46:59 2021

@author: WJ
"""

import copy
import scipy.io as sio
import tensorly as tl
import scipy as sp
import numpy as np
from tensorly.decomposition import tucker

from .util.MDT import MDTWrapper
from .util.functions import svd_init

class BHTTUCKER4_update(object):
    def __init__(self,X1,X2, Us):
        """store all parameters in the class and do checking on taus"""
        
        self._trans_data1 = X1
        self._trans_data2 = X2
        self._Us = Us         


    def _get_Xs(self, trans_data):

        T_hat = trans_data.shape[-1]
        Xs = [ trans_data[..., t] for t in range(T_hat)]

        return Xs    
    def _get_cores(self, Xs, Us):
        cores = [ tl.tenalg.multi_mode_dot( x, [u.T for u in Us], modes=[i for i in range(len(Us))] ) for x in Xs]
        return cores
    
    
    def run(self):
        
        coress, corest    = self._run()
                  
        return  coress, corest
    
    def _run(self):
        # trans_data1, mdt1 = self._forward_MDT(self._ts1, self._taus)
        # trans_data2, mdt2 = self._forward_MDT(self._ts2, self._taus)
        
        # trans_data1=np.transpose(np.array(self._trans_data1),(2,1,0))
        # trans_data2=np.transpose(np.array(self._trans_data2),(2,1,0))
        trans_data1=self._trans_data1
        trans_data2=self._trans_data2

        X_source=np.concatenate([trans_data1],axis=2) # 在dim=2 处拼接
        X_target=np.concatenate([trans_data2],axis=2) # 在dim=2 处拼接

#将三维数组变成列表形式        
        Xss = self._get_Xs(X_source)
        Xst = self._get_Xs(X_target)

        
#得到更新后的源域和目标域的核张量             
        coress = self._get_cores(Xss, self._Us)
        corest = self._get_cores(Xst, self._Us)
        
        # X_S = np.transpose(np.array(coress),(1,2,0))
        # X_T = np.transpose(np.array(corest),(1,2,0))


      
        return coress, corest 