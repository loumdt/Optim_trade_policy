---
title: Optimisation de la politique commerciale française sous contraintes de la réduction de l'empreinte carbone
authors: Julien Ancel, Théo Mandonnet
---
# Optimisation dans l'incertain
## Politique commerciale optimale française en vue d'atteindre son objectif d'empreinte carbone, en deux pas de temps : 2015 et 2030. 

Objectif : Réduire de 40 % les émissions de gaz à effet de serre entre 1990 et 2030.

Ce travail est basé sur l'article de [Beneviste $\textit{et al}$ (2018)](https://doi.org/10.1088/1748-9326/aaa0b9).

Les pays sont agrégés en différents blocs, appelés par la suite régions du monde, notées $r \in R$, pour simplifier la lisibilité des résultats et l'interprétation. Ces blocs sont construits de façon à avoir des NDC cohérentes et similaires au sein d'un bloc.

Les différentes régions du monde $r \in R$ sont donc les suivantes : 

| **Groupe**  | **Pays** |  **Description** |
| :-----: | :----: |  :----: |
| FRA | France | France |
| USA | Etats-Unis | Etats-Unis |
| CHN  | Chine | Chine |
| IND | Inde | Inde |
| RUE | Austria, Belgium, Bulgaria, Croatia, Cyprus, Czech Republic, Denmark, Estonia, Finland, Germany, Greece, Hungary, Ireland, Italy, Latvia, Lithuania,Luxembourg, Malta, Netherlands, Poland, Portugal, Romania, Slovakia, Slovenia, Spain, Sweden, United Kingdom | Reste de l'Union Européenne |
| LEA | Australia, Brazil, Canada, Japan, Kazakhstan, Russian Federation, Ukraine |  Large Emitters with Absolute reduction |
| LENA | Egypt, Indonesia, Iran, South Korea, Malaysia, Mexico, Saudi Arabia, South Africa, China_Taiwan, Thailand, Turkey, United Arab Emirates | Large Emitters with Non Absolute reduction |
| OA1 | Andorra, Belarus, Iceland, Liechtenstein, Monaco, New Zealand, Norway, Switzerland | Autres pays de l'Annexe 1 |
| OEC | Chile, Philippines, Viet Nam, Singapore | Autres pays émergents |
| OOEC | Brunei Darussalam, Bahrain, Kuwait, Oman | Autres pays exportateurs de pétrole |
| ROW NDC | Afghanistan, Albania, Angola, Argentina, Azerbaijan, Bangladesh, Barbados, Benin, Bhutan, Bosnia and Herzegovina, Botswana, Burkina Faso, Burundi, Cambodia, Cameroon, Central African Republic, Chad, Colombia, Comoros, Congo, Congo Democratic Republic, Costa Rica, Djibouti, Dominica, Dominican Republic, Equatorial Guinea, Eritrea, Ethiopia, Gambia, Georgia, Ghana, Grenada, Guatemala, Guinea, Haiti, Jamaica, Kenya, Kiribati, Korea Democratic People's Republic of, Lebanon, Liberia, Macedonia, Madagascar, Maldives, Marshall Islands, Mauritania, Mauritius, Micronesia, Moldova, Mongolia, Morocco, Namibia, Niger, Nigeria, Pakistan, Paraguay, Peru, Saint Kitts and Nevis, Saint Lucia, Saint Vincent and the Grenadines, San Marino, Sao Tome and Principe, Senegal, Serbia and Montenegro, Seychelles, Solomon Islands, Tajikistan, Tanzania, Togo, Trinidad and Tobago, Tunisia, Uganda, Venezuela, Zambia | Reste du monde ayant une NDC  |
| ROW no NDC | Algeria, Antigua and Barbuda, Armenia, Bahamas, Belize, Bolivia, Cabo Verde, Cook Islands, Cote d'Ivoire, Cuba, Ecuador, El Salvador, Fiji, Gabon, Guinea-Bissau, Guyana, Honduras, Iraq, Israel, Jordan, Kyrgyzstan, Laos, Lesotho, Libya, Malawi, Mali, Mozambique, Myanmar, Nauru, Nepal, Nicaragua, Niue, Palau, Panama, Papua New Guinea, Qatar, Rwanda, Samoa, Sierra Leone, Somalia, Sri Lanka, Sudan, Suriname, Swaziland, Syria, Timor-Leste, Tonga, Turkmenistan, Tuvalu, Uruguay, Uzbekistan, Vanuatu, Yemen, Zimbabwe | Reste du monde n'ayant pas de NDC |
| Transport | International Aviation, International Shipping | International Transport |

&nbsp;

&nbsp;


Soit $q_0$ le vecteur des fractions de demande française importées par la France depuis chaque région du monde $r \in R$ en $t=0$, des variables de décision :
$$ q_0 = (q_{0r})_{r \in R} $$


Soit $s_0$ le vecteur des coefficients d'intensité carbone des différentes régions du monde $r \in R$ avec lesquelles la France commerce en $t=0$, des paramètres connus et fixés : 
$$ s_0 = (s_{0r})_{r \in R} $$

Soit $S_1$ le vecteur des coefficients d'intensité carbone des différentes régions $r \in R$ en $t=1$ : 
$$ S_1 = (S_{1r})_{r \in R} $$
où les $S_{1r}$ sont des variables aléatoires, et les $(s_{0r})_{r \in R}$ des paramètres des $(S_{1r})_{r \in R}$.

Soit $\overline{e}_1 = 249$ MtCO$_2$eq la cible d'empreinte carbone française en 2030.

Soit $d_0$ et $D_1$ la demande française respectivement en $t=0$ et en $t=1$.

$D_1$ est une variable aléatoire. 
Nous proposons une analyse par scénario de cette variable $D_1$ ainsi que des $(S_{1r})_{r \in R}$. Sur chaque scénario est posé une probabilité. Ici, les scénarios seront considérés comme équiprobables : 5 scénarios, les SSPs, conduisent à des valeurs différentes du PIB ainsi que des émissions de chaque région du monde $r$. En particulier, ces scénarios permettront de déterminer une valeur pour $D_1$, à partir du PIB, ainsi que pour les $(S_{1r})_{r \in R}$, considérés comme des proxys des intensités carbone du PIB.


En cas de dépassement de la cible $\overline{e}_1$, la France recourt à l'achat de quotas : le montant de quotas achetés est appelé $V$.

La demande est nécessairement satisfaite par les quantités produites et importées, malgré le caractère aléatoire de $D_1$. Il n' y a donc pas de variable de décision supplémentaire à introduire.


Critère : 

$$ \underset{q_0, V}{\text{min}} \: \left( \mathbb{E}\left[ P*V  \right] \right) $$

sous contraintes que :

$$ \forall r \in R, q_{0r} \geq 0 $$
$$ \sum_{r \in R} q_{0r} = 1 $$
$$ \sum_{r \in R} q_{0r}* D_1 * S_{1r} \leq \overline{e}_1 - V $$

$$\sigma ((q_{0r})_{r \in R}) = \{ \empty , \Omega \}$$

$$\sigma (V) \subset \sigma \left( (S_r)_{r \in R}, D_1 \right) $$

En réalité, nous connaissons la valeur de $V$ :

$$ V = \left[ D_1 \sum_r q_{0r} S_r - \overline{e} \right]_+ $$

Ainsi, on peut reformuler le critère : 

$$ \underset{q_0}{\text{min}} \: \sum_{s} \pi_s \left( d_1^s \sum_r q_{0r} S_{rs} - \overline{e} \right)_+ $$

sous contraintes que :

$$ \forall r \in R, q_{0r} \geq 0 $$
$$ \sum_{r \in R} q_{0r} = 1 $$
