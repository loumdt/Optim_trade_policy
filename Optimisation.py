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
reg_list = ['Austria', 'Belgium', 'China', 'France',  'Germany', 'Ireland', 'Italy', 'Netherlands', 'Poland',
 'Portugal', 'Russian Federation', 'Saudi Arabia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'United Kingdom','United States']
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

constr+=[ q <= 1, cp.sum(q) == 1]
objective = cp.Minimize(cost)
prob = cp.Problem(objective,constr)
result = prob.solve()
print("Solution CVXPY")
print(q.value)
print("Respect contrainte somme : %s"%(np.isclose(np.sum(q.value),1., rtol=1e-5, atol=1e-5)))
print("Respect positivité : %s"%(q.value>=0.).all())
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

constr+=[ q <= 1, cp.sum(q) == 1]
objective = cp.Minimize(crit)
prob = cp.Problem(objective,constr)
result = prob.solve()
print("Solution CVXPY")
print(q.value)
print("Respect contrainte somme : %s"%(np.isclose(np.sum(q.value),1., rtol=1e-5, atol=1e-5)))
print("Respect positivité : %s"%(q.value>=0.).all())
print("Valeur objectif : %s"%objective.value)


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
    intens = np.array([np.random.normal(distrib_intes[r,s,0],distrib_intes[r,s,1]) for r in range(nbreg)])
    gdp_FR = np.mean([gdpFR[s][source] for source in range(len(sources))])/M
    crit += cp.pos(gdp_FR*intens@q-cible_EC/M)

constr+=[ q <= 1, cp.sum(q) == 1]
objective = cp.Minimize(crit)
prob = cp.Problem(objective,constr)
result = prob.solve()
print("Solution CVXPY")
print(q.value)
print(np.sum(q.value))
print("Respect contrainte somme : %s"%(np.isclose(np.sum(q.value),1., rtol=1e-5, atol=1e-5)))
print("Respect positivité : %s"%(q.value>=0.).all())
print("Valeur objectif : %s"%objective.value)


###################################################################################################
# Histogramme des réponses
###################################################################################################
#%% Histogrammes de reponse
# Test de la politique optimale
q_opti = q.value

def criteremoy(ssp,source,my_q):
    curr_gdpFR = gdp_france["SSP{}".format(ssp+1)][sources[source]]
    return max(curr_gdpFR*(np.dot(moy_intens[ssp,source],my_q))-cible_EC,0)

def criteredis(ssp,my_q):
    curr_gdpFR = np.mean([gdp_france["SSP{}".format(ssp+1)][sources[source]] for source in range(len(sources))])
    intens = np.array([np.random.normal(distrib_intes[r,ssp,0],distrib_intes[r,ssp,1]) for r in range(nbreg)])
    return max(curr_gdpFR*(np.dot(intens,my_q))-cible_EC,0)


Nbiter = 100000
val_opti = np.zeros(Nbiter)
val_stratmoy = np.zeros(Nbiter)
val_china = np.zeros(Nbiter)
val_deuxpays = np.zeros(Nbiter)
val_actuelle = np.zeros(Nbiter)
q2p = np.zeros(nbreg)
q2p[4]=0.5
q2p[-1]=0.5
qturq = np.zeros(nbreg)
qturq[2] = 1
#reg_list = ['Austria', 'Belgium', 'China', 'France',  'Germany', 'Ireland', 'Italy', 'Netherlands', 'Poland',
# 'Portugal', 'Russian Federation', 'Saudi Arabia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'United Kingdom','United States']
qactuel = [0.3*0.012,0.3*0.125,0.3*0.055,0.7,0.3*0.275,0.3*0.012,0.3*0.11,0.3*0.099,0.3*0.02,0.3*0.01,0.3*0.012,0.3*0.01,0.3*0.093,0.3*0.012,
 0.3*0.032,0.3*0.011,0.3*0.042,0.3*0.085]
moyennes=False
for i in range(Nbiter):
    if moyennes:    
        s = np.random.randint(0,len(ssps))
        source = np.random.randint(0,len(sources))
        val_opti[i]=criteremoy(s,source,q_opti)
        val_stratmoy[i]=criteremoy(s,source,np.ones(nbreg)/nbreg)
        val_china[i]=criteremoy(s,source,qturq)
        val_deuxpays[i]=criteremoy(s,source,q2p)
        val_actuelle[i]=criteremoy(s,source,qactuel)
    else:
        s = np.random.randint(0,len(ssps))
        val_opti[i]=criteredis(s,q_opti)
        val_stratmoy[i]=criteredis(s,np.ones(nbreg)/nbreg)
        val_china[i]=criteredis(s,qturq)
        val_deuxpays[i]=criteredis(s,q2p)
        val_actuelle[i]=criteredis(s,qactuel)

fig,ax = plt.subplots()
#ax.hist(val_stratmoy,bins=300,color="green",alpha=0.5,label="strat moy")
ax.hist(np.log(val_china+1),bins=300,color="red",alpha=0.5,label="100% Chine")
ax.hist(np.log(val_opti+1),bins=300,color="black",label="Opti")
ax.hist(np.log(val_deuxpays+1),bins=300,color="blue",alpha=0.5,label="Germany-USA")
ax.hist(np.log(val_actuelle+1),bins=300,color="orange",alpha=0.5,label="Actuelle")
plt.xlabel("echelle log(x+1)")
plt.legend()
plt.show()
