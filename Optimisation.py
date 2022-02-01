""" Optimize french trade policy
"""

from calendar import c
import numpy as np
from numpy.core.einsumfunc import _greedy_path
import matplotlib.pyplot as plt
import scipy.optimize as sco
from scipy.stats import norm
import json
import cvxpy as cp
import pandas as pd

create_little_json =False
if create_little_json:
    f = open('List_Emi.json')
    list_emi = json.load(f)
    f.close()

    f = open('List_ChinaEmi.json')
    list_chinaemi = json.load(f)
    f.close()

    f=open('GDP_data.json')
    GDP_data = json.load(f)
    f.close()


sources = ['CEPII','OECD','IIASA','PIK']
ssps = ['SSP1','SSP2','SSP3','SSP4','SSP5']
reg_list=[]
if create_little_json:
    for r in list(list_emi['CEPII']['SSP1'].keys()):
        if r in ['OtherAnnex1', 'OtherEmerging', 'OtherOil', 'RestOfWorld', 'LEA', 'LENA','Transport', 'WorldOther','Int. Aviation','Int. Shipping',]:
            continue
        list_gdp = [[GDP_data[source][s][r]["2030"] for source in sources] for s in ssps]
        if np.min(list_gdp)!=0.:
            reg_list.append(r)

def moy_gdp(ssp,country):
    res = []
    for source in sources:
        res.append(GDP_data[source][ssp][country]["2030"])
    return np.mean(res)

if create_little_json:
    my_dict={}
    for s in range(len(ssps)):
        inner = {}
        for source in sources:
            inner[source] = GDP_data[source]["SSP{}".format(s+1)]["France"]["2030"]
        my_dict["SSP{}".format(s+1)] = inner
    json1 = json.dumps(my_dict)
    f = open("data_opti/GDP_France.json","w")
    f.write(json1)
    f.close()

    for r in reg_list:
        if r in list(GDP_data["CEPII"]["SSP1"].keys()):
            my_dict={}
            for s in range(len(ssps)):
                inner={}
                list_emi_pays = []
                for source in sources:
                    gdp = GDP_data[source]["SSP{}".format(s+1)][r]["2030"]
                    myints=list(np.array(list_emi[source]["SSP{}".format(s+1)][r]['GHGtot']["2030"])/gdp)
                    inner[source]=np.mean(myints)
                    list_emi_pays+=myints
                mu,sigma = norm.fit(list_emi_pays)
                inner["distrib norm"] = [mu, sigma]
                my_dict["SSP{}".format(s+1)] = inner                
            json1 = json.dumps(my_dict)
            f = open("data_opti/Emi_"+r+".json","w")
            f.write(json1)
            f.close()
            print(r)


#######################################################################################################
# PARAMETRES
#######################################################################################################

# reg_list = ['Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh',
#  'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana',
#  'Brazil', 'Brunei Darussalam', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon',
#  'Canada', 'Central African Republic', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo', 'Costa Rica',
#  "Cote d'Ivoire", 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Djibouti', 'Dominican Republic',
#  'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Estonia', 'Ethiopia', 'European Union', 'Fiji',
#  'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece','Guatemala', 'Guinea',
#  'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia',
#  'Iran, Islamic Republic of', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan',
#  'Kenya', 'Korea, Republic of', 'Kuwait', 'Kyrgyzstan', "Lao People's Democratic Republic", 'Latvia',
#  'Lebanon', 'Lesotho', 'Liberia', 'Libyan Arab Jamahiriya', 'Lithuania', 'Luxembourg',
#  'Macedonia, the former Yugoslav Republic of', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali',
#  'Malta', 'Mauritania', 'Mauritius', 'Mexico', 'Moldova, Republic of', 'Mongolia', 'Morocco', 'Mozambique',
#  'Namibia', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Norway', 'Oman',
#  'Pakistan', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal','Romania',
#  'Russian Federation', 'Rwanda', 'Saudi Arabia', 'Senegal', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia',
#  'Solomon Islands', 'South Africa', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden',
#  'Switzerland', 'Tajikistan', 'Tanzania_United Republic of', 'Thailand', 'Togo', 'Trinidad and Tobago',
#  'Tunisia', 'Turkey', 'Turkmenistan', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom',
#  'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Viet Nam', 'Yemen', 'Zambia']

data_countries=pd.read_excel("List_Countries.xlsx")
reg_list = np.array([data_countries.loc[:,'Countries']])

reg_list = np.append(reg_list,'France')

#reg_list = ['Austria', 'Belgium', 'China', 'France',  'Germany', 'Ireland', 'Italy', 'Netherlands', 'Poland',
# 'Portugal', 'Russian Federation', 'Saudi Arabia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'United Kingdom','United States']
# reg_list = ['Argentina', 'Australia', 'Austria', 'Belgium','Brazil', 'Canada', 'Croatia', 'Czech Republic', 'Denmark',
# 'Egypt', 'Estonia','Finland', 'France', 'Georgia', 'Germany', 'Greece', 'Hungary', 'India', 'Ireland', 'Israel',
# 'Italy', 'Japan', 'Korea, Republic of', 'Lithuania', 'Luxembourg', 'Mexico', 'Morocco', 'Netherlands',
# 'New Zealand', 'Norway', 'Poland', 'Portugal','Romania', 'Russian Federation' ,'South Africa', 'Spain', 'Sweden',
# 'Switzerland', 'Tunisia', 'Turkey',  'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States']

nbreg = len(reg_list)
print(nbreg)


cible_EC = 249000 ## A modifier

f = open('data_opti/GDP_France.json')
gdp_france = json.load(f)
f.close()


gdpFR = [[gdp_france["SSP{}".format(s+1)][source] for source in sources] for s in range(len(ssps))]
moy_intens = np.zeros((len(ssps),len(sources),nbreg))
distrib_intes = np.zeros((nbreg,len(ssps),2))

for s in range(len(ssps)):
    for r in range(nbreg):
        for source in range(len(sources)):
            f = open('data_opti/Emi_'+reg_list[r]+'.json')
            inte = json.load(f)[ssps[s]][sources[source]]
            moy_intens[s,source,r] = inte
            f.close()
        
        f = open('data_opti/Emi_'+reg_list[r]+'.json')
        dis = json.load(f)[ssps[s]]["distrib norm"]
        distrib_intes[r,s,:] = dis
        f.close()

#reg_list = ['Austria', 'Belgium', 'China', 'France',  'Germany', 'Ireland', 'Italy', 'Netherlands', 'Poland',
# 'Portugal', 'Russian Federation', 'Saudi Arabia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'United Kingdom','United States']
# Pour la contrainte de capcacite d exportation

qactuel = 0.3*np.array(data_countries.loc[:,'Share']/100)
#qactuel = np.append(qactuel,0)
qactuel = np.append(qactuel,0.7)
print(qactuel)

#qactuel = [0.3*0.012,0.3*0.1235,0.3*0.055,0.7,0.3*0.275,0.3*0.012,0.3*0.11,0.3*0.099,0.3*0.02,0.3*0.01,0.3*0.012,0.3*0.01,0.3*0.093,0.3*0.0075,
# 0.3*0.0275,0.3*0.0065,0.3*0.042,0.3*0.085]
pct_exportmax = 0.3
qmax = (1+ pct_exportmax)*np.array(qactuel)
#qmax[-1]=1.05*qactuel[-1]

###################################################################################################
# Méthode 1 : scenario avec prix du C
###################################################################################################
#%% Avec un prix du carbone et la nouvelle formulation
#construct the problem
q = cp.Variable(nbreg,nonneg=True)
v = cp.Variable(nonneg=True)
prix_quot = cp.Parameter(len(ssps))
prix_quot.value = [1,2,3,4,5]
constr = []

pis = cp.Parameter()
pis.value = 1/(len(ssps)*len(sources))
cost=v*pis*cp.sum(prix_quot)

for s in range(len(ssps)):
    for source in range(len(sources)):
        constr+= [ gdpFR[s][source]*(moy_intens[s,source]@q)-v <= cible_EC ]

constr+=[q <= 1, cp.sum(q) == 1, q <= qmax]
objective = cp.Minimize(cost)
prob = cp.Problem(objective,constr)
result = prob.solve()
print("Solution CVXPY")
print(q.value)
print("Respect contrainte somme : %s"%(np.isclose(np.sum(q.value),1., rtol=1e-5, atol=1e-5)))
print("Respect positivité : %s"%(q.value>=0.).all())
print("Respect capacité : %s"%(qmax-q.value >= -1e-6).all())
print("Valeur objectif : %s"%objective.value)

# print("Cas où SSP5 est connu")
# x = cp.Variable(nbreg,nonneg=True)
# v2 = cp.Variable(nonneg=True)
# cost2 = v*prix_quot[4]
# const2 = [gdpFR[4]*(moy_intens[4]@x)-v2 <= cible_EC,x <= 1, cp.sum(x) == 1]
# objective2 = cp.Minimize(cost2)
# prob2 = cp.Problem(objective2,const2)
# result2 = prob2.solve()
# print(x.value)

###################################################################################################
# Méthode 2 : scenario sans prix du C et avec formulation initiale
###################################################################################################
#%% CVXPY 2nd formulation
#construct the problem with the secon formulation :
q = cp.Variable(nbreg,nonneg=True)
constr = []

pis = cp.Parameter()
pis.value = 1/(len(ssps)*len(sources))

crit = 0
for s in range(len(ssps)):
    for source in range(len(sources)):
        crit += pis.value*cp.pos(gdpFR[s][source]*moy_intens[s,source]@q-cible_EC)

constr+=[ q <= 1, cp.sum(q) == 1, q <= qmax]
objective = cp.Minimize(crit)
prob = cp.Problem(objective,constr)
result = prob.solve()
print("Solution CVXPY - Scénarios")
print(q.value)
print("Respect contrainte somme : %s"%(np.isclose(np.sum(q.value),1., rtol=1e-5, atol=1e-5)))
print("Respect positivité : %s"%(q.value>=0.).all())
print("Respect capacité : %s"%(qmax-q.value >= -1e-6).all())
print("Valeur objectif : %s"%objective.value)
q_opti_scenar = q.value

###################################################################################################
# Méthode 3 : SSP pour GDP et distrib normale pour emi associée au SSP tiré
###################################################################################################
#%% Autre implementation de l'alea -> on a une distribution par SSP et le tirage d'un SSP 
# pour avoir un effet de tendance mondiale
# Avec une distribution normale des intensites pour chaque region
q = cp.Variable(nbreg,nonneg=True)
constr = []

M = 10000
crit = 0
for i in range(M):
    s= np.random.randint(0,len(ssps))
    source_gdp = np.random.randint(0,len(sources))
    intens = np.array([np.random.normal(distrib_intes[r,s,0],distrib_intes[r,s,1]) for r in range(nbreg)])
    gdp_FR = gdpFR[s][source_gdp]/M
    crit += cp.pos(gdp_FR*intens@q-cible_EC/M)

constr+=[q<=1, cp.sum(q) == 1, q <= qmax]
objective = cp.Minimize(crit)
prob = cp.Problem(objective,constr)
result = prob.solve(max_iters=1000)
print("Solution CVXPY - Gaussiennes")
print(q.value)
print("Respect contrainte somme : %s"%(np.isclose(np.sum(q.value),1., rtol=1e-5, atol=1e-5)))
print("Respect positivité : %s"%(q.value>=0.).all())
print("Respect capacité : %s"%(qmax-q.value >= -1e-6).all())
print("Valeur objectif : %s"%objective.value)
q_opti = q.value
###################################################################################################
# Optimisation dans le cas d'un futur connu - SSP connu et intensite = moyenne de la distrib gaussienne
###################################################################################################
q = cp.Variable(nbreg,nonneg=True)
constr = []
#SSP connu
s = 2
source_gdp=0
intens = moy_intens[s,source_gdp] #emissions connues
gdp_FR = gdpFR[s][source_gdp] #pib connu
crit = cp.pos(gdp_FR*intens@q-cible_EC)

constr+=[q<=1, cp.sum(q) == 1, q <= qmax]
objective = cp.Minimize(crit)
prob = cp.Problem(objective,constr)
result = prob.solve(max_iters=1000)
print("Solution CVXPY - Cas déterministe")
print(q.value)
print("Respect contrainte somme : %s"%(np.isclose(np.sum(q.value),1., rtol=1e-5, atol=1e-5)))
print("Respect positivité : %s"%(q.value>=0.).all())
print("Respect capacité : %s"%(qmax-q.value >= -1e-6).all())
print("Valeur objectif : %s"%objective.value)
q_deterministe = q.value

###################################################################################################
# Histogramme des réponses
###################################################################################################
#%% Histogrammes de reponse
# Test de la politique optimale
def criteremoy(ssp,source,my_q):
    curr_gdpFR = gdp_france["SSP{}".format(ssp+1)][sources[source]]
    return max(curr_gdpFR*(np.dot(moy_intens[ssp,source],my_q))-cible_EC,0)

def criteredis(my_q,intens,curr_gdpFR):
    return max(curr_gdpFR*(np.dot(intens,my_q))-cible_EC,0)


Nbiter = 100000
val_opti = np.zeros(Nbiter)
val_opti_scenar = np.zeros(Nbiter)
val_opti_deterministe = np.zeros(Nbiter)
val_stratmoy = np.zeros(Nbiter)
val_deuxpays = np.zeros(Nbiter)
val_actuelle = np.zeros(Nbiter)
q2p = np.zeros(nbreg)
q2p[3]=0.7
if qmax[4]>= 0.5:
    q2p[4]=0.5
else:
    q2p[4]=qmax[4]
if qmax[-1]>= 0.5:
    q2p[-1]=0.5
else:
    q2p[-1]=qmax[-1]
remaining = 0.3 - np.sum([qactuel[i]*(reg_list[i]!='France')*(reg_list[i]!='Germany')*(reg_list[i]!='United States') for i in range(nbreg)])
for i in range(nbreg):
    if reg_list[i]!='France' and reg_list[i]!='Germany' and reg_list[i]!='United States':
        q2p[i] = remaining/15

qmoy = np.zeros(nbreg)
for i in range(nbreg):
    if reg_list[i]!='France':
        qmoy[i] = 0.3/(nbreg-1)
    else:
        qmoy[i]=0.7

#reg_list = ['Austria', 'Belgium', 'China', 'France',  'Germany', 'Ireland', 'Italy', 'Netherlands', 'Poland',
# 'Portugal', 'Russian Federation', 'Saudi Arabia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'United Kingdom','United States']
#qactuel = [0.3*0.012,0.3*0.1235,0.3*0.055,0.7,0.3*0.275,0.3*0.012,0.3*0.11,0.3*0.099,0.3*0.02,0.3*0.01,0.3*0.012,0.3*0.01,0.3*0.093,0.3*0.0075,
# 0.3*0.0275,0.3*0.0065,0.3*0.042,0.3*0.085]
moyennes=False
for i in range(Nbiter):
    if moyennes:    
        s = np.random.randint(0,len(ssps))
        source = np.random.randint(0,len(sources))
        val_opti[i]=criteremoy(s,source,q_opti)
        val_opti_scenar[i]=criteremoy(s,q_opti_scenar)
        val_stratmoy[i]=criteremoy(s,source,qmoy)
        val_deuxpays[i]=criteremoy(s,source,q2p)
        val_actuelle[i]=criteremoy(s,source,qactuel)
        val_opti_deterministe[i]=criteremoy(s,source,q_deterministe)
    else:
        s = np.random.randint(0,len(ssps))
        source_gdp = np.random.randint(0,len(sources))
        gdp_FRcurr = gdp_france[ssps[s]][sources[source_gdp]]
        intens = np.array([np.random.normal(distrib_intes[r,s,0],distrib_intes[r,s,1]) for r in range(nbreg)])
        val_opti[i]=criteredis(q_opti,intens,gdp_FRcurr)
        val_opti_scenar[i]=criteredis(q_opti_scenar,intens,gdp_FRcurr)
        val_stratmoy[i]=criteredis(qmoy,intens,gdp_FRcurr)
        val_deuxpays[i]=criteredis(q2p,intens,gdp_FRcurr)
        val_actuelle[i]=criteredis(qactuel,intens,gdp_FRcurr)
        val_opti_deterministe[i]=criteredis(q_deterministe,intens,gdp_FRcurr)


fig,ax = plt.subplots(figsize=(18,12))
ax.hist(val_stratmoy,bins=300,color="green",alpha=0.75,label="strat moy")
ax.hist(val_deuxpays,bins=300,color="blue",alpha=0.25,label="Allemagne-USA")
ax.hist(val_actuelle,bins=300,color="orange",alpha=0.5,label="Actuelle")
ax.hist(val_opti,bins=300,color="black",label="Opti gaussiennes")
plt.xlabel("Critère",size=20)
plt.ylabel("Fréquence",size=20)
plt.legend(prop={'size': 20})
plt.grid()
plt.show()

fig,ax = plt.subplots(figsize=(18,12))
ax.hist(val_stratmoy,bins=300,color="green",alpha=0.75,label="strat moy")
ax.hist(val_deuxpays,bins=300,color="blue",alpha=0.25,label="Allemagne-USA")
ax.hist(val_actuelle,bins=300,color="orange",alpha=0.5,label="Actuelle")
ax.hist(val_opti_scenar,bins=300,color="black",label="Opti par scenario")
plt.xlabel("Critère",size=20)
plt.ylabel("Fréquence",size=20)
plt.legend(prop={'size': 20})
plt.grid()
plt.show()

fig,ax = plt.subplots(figsize=(18,12))
ax.hist(val_opti_deterministe,bins=300,color="red",alpha=0.45,label="Opti déterministe")
ax.hist(val_opti_scenar,bins=300,color="green",alpha=0.45,label="Opti par scenario")
ax.hist(val_opti,bins=300,color="blue",alpha=0.45,label="Opti gaussiennes")
plt.xlabel("Critère",size=20)
plt.ylabel("Fréquence",size=20)
plt.legend(prop={'size': 20})
plt.grid()
plt.show()