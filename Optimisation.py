""" Optimize french trade policy
"""

import pandas as pd
import numpy as np
import scipy.optimize as sco

fileS0 = "MatMat-Trade/outputs/reference_2015_ref_ref/ghg_emissions_desag/S.txt"

S0_desag = pd.read_csv(fileS0, sep= "\t",index_col=[0,1],header=[0,1])
S0 = S0_desag.sum().sum(level=0)
reg_list = list(S0.index)
nbreg = len(reg_list)
print(nbreg)
# cas tout scenarise
scenario = []
# scenario = [ [ d_1, [s_1_1, s_1_2,..., s_1_R] ]  ,  [,[]] ]
cible_EC = 10

def critere(q):
    res=0
    for s in range(len(scenario)):
        EC=0
        for r in range(nbreg):
            EC+=q[r]*scenario[s][1][r]
        EC*=scenario[s][0]
        pis = 1/len(scenario)
        res+= pis * max(EC-cible_EC,0)
    return res

contrainte_demande = sco.LinearConstraint(A=np.ones(nbreg),lb=1,ub=1)

q0 = np.ones(nbreg)/nbreg
resu_opti = sco.minimize(critere, q0, constraints=(contrainte_demande))

# cas scenario seulement sur la demande
scenario = []
# scenario = [d_1_1,..., d_1_n] n scenario de demande
#distribution sur les s_r
S1_moy = S0.copy() #exemple
sigma_r = [1 for r in reg_list] #exemple
distrib_s = [ [np.random.randn, (S1_moy[reg_list[r]], sigma_r) ] for r in reg_list] #exemple
# distrib_s = [ [loi_1,params_1],...,[loi_R,params_R] ]

def critere2(q):
    res=0
    M=1e6
    for i in range(M):
        scen_dem = np.random.randint(1,len(scenario))
        d = scenario[scen_dem]
        S = np.array([distrib_s[r][0](*distrib_s[r][1]) for r in range(nbreg)])
        EC = d*np.dot(q,S)
        res+= max(0,EC-cible_EC)
    return res/M

q0 = np.ones(nbreg)/nbreg
resu_opti2 = sco.minimize(critere2, q0, constraints=(contrainte_demande))