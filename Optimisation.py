""" Optimize french trade policy
"""

import pandas as pd
import numpy as np
import scipy.optimize as sco
from scipy.stats import norm
import json


f = open('List_Emi.json')
list_emi = json.load(f)
f.close()

f = open('List_ChinaEmi.json')
list_chinaemi = json.load(f)
f.close()

f=open('GDP_data.json')
GDP_data = json.load(f)
f.close()

sources = list(list_chinaemi.keys())
ssps = list(list_emi['CEPII'].keys())
reg_list = list(list_emi['CEPII']['SSP1'].keys())

def moy_gdp(ssp,country):
    res = []
    for source in sources:
        res.append(GDP_data[source][ssp][country]["2030"])
    return res.mean()

def intensite(ssp,country):
    gdp_moy = moy_gdp(ssp,country)
    intensites={}
    for source in sources:
        intensites[source]=np.array(list_emi[source][ssp][country]['GHGtot']["2030"])/gdp_moy
    return intensites

nbreg = len(reg_list)
print(len(list_emi["CEPII"]["SSP1"]['European Union']['GHGtot']["2030"]))
exit()
# cas tout scenarise
# on tire au hasard un ssp

cible_EC = 10 ## A modifier

def critere(q):
    res=0
    for s in range(len(ssps)):
        gdpFR = moy_gdp("SSP.format{s}","France")
        EC=0
        for r in range(nbreg):
            distrib_intensite = intensite("SSP.format{s}",reg_list[r])
            moy_intensite = np.mean(distrib_intensite)
            EC+=q[r]*moy_intensite
        EC*=gdpFR
        pis = 1/len(scenario)
        res+= pis * max(EC-cible_EC,0)
    return res

contrainte_demande = sco.LinearConstraint(A=np.ones(nbreg),lb=1,ub=1)

q0 = np.ones(nbreg)/nbreg
resu_opti = sco.minimize(critere, q0, constraints=(contrainte_demande))

# cas scenario seulement sur la demande
scenario = []
# scenario = [d_1_1,..., d_1_5] 5 scenarios de demande
#distribution sur les s_r

def critere2(q):
    res=0
    M=1e6
    for i in range(M):
        scen_dem = np.random.randint(1,len(ssps))
        ssp = ssps[scen_dem]
        gdpFR = moy_gdp(ssp,"France")
        EC=0
        for r in range(nbreg):
            distrib_intensites = []
            for source in sources:
                distrib_intensites.append(list_emi[source][ssp][reg_list[r]]['GHGtot']["2030"])
            distrib_r = np.mean(distrib_intensites,axis=1) #ou = 0 vois si erreur
            mu = np.mean(distrib_r)
            std = np.std(distrib_r)
            my_intensite = std*np.random.randn()+mu
            EC+=q[r]*my_intensite
        EC*=gdpFR
        res+= max(0,EC-cible_EC)
    return res/M

q0 = np.ones(nbreg)/nbreg
resu_opti2 = sco.minimize(critere2, q0, constraints=(contrainte_demande))