""" Optimize french trade policy
"""

from cvxpy import constraints
import numpy as np
from numpy.core.einsumfunc import _greedy_path
import scipy.optimize as sco
import json
import cvxpy as cp

create_little_json = False
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
        if GDP_data["CEPII"]["SSP1"][r]["2030"]!=0:
            reg_list.append(r)


def moy_gdp(ssp,country):
    res = []
    for source in sources:
        res.append(GDP_data[source][ssp][country]["2030"])
    return np.mean(res)

if create_little_json:
    my_dict={}
    for s in range(len(ssps)):
        moy_gdp_france = moy_gdp("SSP{}".format(s+1),"France")
        my_dict["SSP{}".format(s+1)] = moy_gdp_france
    json1 = json.dumps(my_dict)
    f = open("data_opti/GDP_France.json","w")
    f.write(json1)
    f.close()

if create_little_json:
    def intensite(ssp,country):
        gdp_moy = moy_gdp(ssp,country)
        intensites=[]
        for source in sources:
            intensites.append(np.array(list_emi[source][ssp][country]['GHGtot']["2030"])/gdp_moy)
        return intensites

    for r in reg_list:
        my_dict={}
        if r in list(GDP_data["CEPII"]["SSP1"].keys()) and GDP_data["CEPII"]["SSP1"][r]["2030"]!=0:
            for s in range(len(ssps)):
                distrib_intensite = intensite("SSP{}".format(s+1),r)
                moy_intensite = np.mean(distrib_intensite)
                my_dict["SSP{}".format(s+1)] = moy_intensite
            json1 = json.dumps(my_dict)
            f = open("data_opti/Emi_"+r+".json","w")
            f.write(json1)
            f.close()


# reg_list = ['Afghanistan', 'Albania', 'Algeria', 'Angola', 'Argentina', 'Armenia', 'Australia', 'Austria',
# 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin','Bhutan',
# 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei Darussalam', 'Bulgaria','Burkina Faso',
# 'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon', 'Canada', 'Central African Republic','Chile', 'China',
# 'Colombia', 'Comoros', 'Congo', 'Congo_the Democratic Republic of the', 'Costa Rica',"Cote d'Ivoire", 'Croatia',
# 'Cyprus', 'Czech Republic', 'Denmark', 'Djibouti', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador',
# 'Equatorial Guinea', 'Estonia', 'Ethiopia', 'European Union', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia',
# 'Georgia', 'Germany', 'Ghana', 'Greece', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti',
# 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran, Islamic Republic of', 'Ireland', 'Israel',
# 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Korea, Republic of', 'Kuwait', 'Kyrgyzstan',
# "Lao People's Democratic Republic", 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libyan Arab Jamahiriya',
# 'Lithuania', 'Luxembourg', 'Macedonia, the former Yugoslav Republic of', 'Madagascar', 'Malawi',
# 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Mauritania', 'Mauritius', 'Mexico', 'Moldova, Republic of',
# 'Mongolia', 'Morocco', 'Mozambique', 'Namibia', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua',
# 'Niger', 'Nigeria', 'Norway', 'Oman', 'Pakistan', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru',
# 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russian Federation', 'Rwanda',
# 'Saudi Arabia', 'Senegal', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Solomon Islands',
# 'South Africa', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland',
# 'Tajikistan', 'Tanzania_United Republic of', 'Thailand', 'Togo', 'Trinidad and Tobago', 'Tunisia',
# 'Turkey', 'Turkmenistan', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 
# 'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Viet Nam', 'Yemen', 'Zambia']
reg_list = ['Algeria','Australia', 'Austria','Belgium', 'Brazil', 'Canada', 'China', 'Croatia','Czech Republic',
 'Denmark', 'France', 'Germany', 'Greece', 'India', 'Ireland','Italy', 'Japan','Lithuania', 'Luxembourg',
 'Netherlands', 'Norway', 'Portugal', 'Russian Federation','South Africa', 'Spain', 'Sweden','Switzerland',
 'Tunisia','Turkey', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States']
nbreg = len(reg_list)
print(nbreg)

# cas tout scenarise
# on tire au hasard un ssp

cible_EC = 249000 ## A modifier

f = open('data_opti/GDP_France.json')
gdp_france = json.load(f)
f.close()

def critere(q):
    res=0
    for s in range(len(ssps)):
        gdpFR = gdp_france["SSP{}".format(s+1)]
        EC=0
        for r in range(nbreg):
            f = open('data_opti/Emi_'+reg_list[r]+'.json')
            moy_intensite = json.load(f)["SSP{}".format(s+1)]
            f.close()
            EC+=q[r]*moy_intensite
        EC*=gdpFR
        pis = 1/len(ssps)
        res+= pis * max(EC-cible_EC,0)
    return res

def critere_f_connu(q,ssp):
    gdpFR = gdp_france[ssp]
    EC=0
    for r in range(nbreg):
        f = open('data_opti/Emi_'+reg_list[r]+'.json')
        moy_intensite = json.load(f)[ssp]
        f.close()
        EC+=q[r]*moy_intensite
    EC*=gdpFR
    res= max(EC-cible_EC,0)
    return res

#construct the problem
q = cp.Variable(nbreg,nonneg=True)
v = cp.Variable(nonneg=True)
prix_quot = cp.Parameter(len(ssps))
prix_quot.value = [1,2,3,4,5]
constr = []
gdpFR = cp.Parameter(len(ssps))
gdpFR.value = [gdp_france["SSP{}".format(s+1)] for s in range(len(ssps))]
moy_intens = np.zeros((len(ssps),nbreg))

for s in range(len(ssps)):
    for r in range(nbreg):
        f = open('data_opti/Emi_'+reg_list[r]+'.json')
        moy_intens[s,r] = json.load(f)["SSP{}".format(s+1)]
        f.close()

pis = cp.Parameter()
pis.value = 1/len(ssps)
cost=v*pis*cp.sum(prix_quot)

for s in range(len(ssps)):
    constr+= [ gdpFR[s]*(moy_intens[s]@q)-v <= cible_EC ]

constr+=[ q <= 1, cp.sum(q) == 1]
objective = cp.Minimize(cost)
prob = cp.Problem(objective,constr)
result = prob.solve()
print("Solution CVXPY")
print(q.value)
print("Respect contrainte somme : %s"%(np.isclose(np.sum(q.value),1., rtol=1e-10, atol=1e-10)))
print("Respect positivité : %s"%(q.value>=0.).all())
print("Valeur objectif : %s"%objective.value)


print("Cas où SSP5 est connu")
x = cp.Variable(nbreg,nonneg=True)
v2 = cp.Variable(nonneg=True)
cost2 = v*prix_quot[4]
const2 = [gdpFR[4]*(moy_intens[4]@x)-v2 <= cible_EC,x <= 1, cp.sum(x) == 1]
objective2 = cp.Minimize(cost2)
prob2 = cp.Problem(objective2,const2)
result2 = prob2.solve()
print(x.value)


cons=[{'type': 'eq', 'fun': lambda x:  np.sum(x)-1}]
cons=tuple(cons)

q0 = np.ones(nbreg)/nbreg
#q0 = np.zeros(nbreg)
#q0[0] = 0.5
#q0[5]=0.5
#print(critere(q0))
#print(critere_f_connu(q0,"SSP5"))
resu_opti = sco.minimize(critere, q0,method='SLSQP',bounds=[(0.,1.) for i in range(nbreg)], constraints=cons)
print("Solution de scipy")
print(resu_opti.x)

#%% CVXPY 2nd formulation
#construct the problem with the secon formulation :
q = cp.Variable(nbreg,nonneg=True)
constr = []
gdpFR = cp.Parameter(len(ssps))
gdpFR.value = [gdp_france["SSP{}".format(s+1)] for s in range(len(ssps))]
moy_intens = np.zeros((len(ssps),nbreg))

for s in range(len(ssps)):
    for r in range(nbreg):
        f = open('data_opti/Emi_'+reg_list[r]+'.json')
        moy_intens[s,r] = json.load(f)["SSP{}".format(s+1)]
        f.close()

pis = cp.Parameter()
pis.value = 1/len(ssps)

crit = 0
for s in range(len(ssps)):
    crit += pis.value*cp.pos(gdpFR[s]*moy_intens[s]@q-cible_EC)

constr+=[ q <= 1, cp.sum(q) == 1]
objective = cp.Minimize(crit)
prob = cp.Problem(objective,constr)
result = prob.solve()
print("Solution CVXPY")
print(q.value)
print("Respect contrainte somme : %s"%(np.isclose(np.sum(q.value),1., rtol=1e-10, atol=1e-10)))
print("Respect positivité : %s"%(q.value>=0.).all())
print("Valeur objectif : %s"%objective.value)