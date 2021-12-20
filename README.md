---
title: Optimisation de la politique commerciale française sous contraintes de la réduction de l'empreinte carbone
authors: Julien Ancel, Théo Mandonnet
---
# Optimisation dans l'incertain
## Politique commerciale optimale française en vue d'atteindre son objectif d'empreinte carbone, en deux pas de temps : 2015 et 2030. 

Objectif : Réduire de 40 % les émissions de gaz à effet de serre entre 1990 et 2030.

Soit $q_0$ le vecteur des fractions de demande française importées par la France depuis chaque région du monde $r \in R$ en $t=0$, des variables de décision :
$$ q_0 = (q_{0r})_{r \in R} $$



Soit $s_0$ le vecteur des coefficients d'intensité carbone des différentes régions du monde $r \in R$ avec lesquelles la France commerce en $t=0$, des paramètres connus et fixés : 
$$ s_0 = (s_{0r})_{r \in R} $$

Soit $S_1$ le vecteur des coefficients d'intensité carbone des différentes régions $r \in R$ en $t=1$ : 
$$ S_1 = (S_{1r})_{r \in R} $$
où les $S_{1r}$ sont des variables aléatoires, et les $(s_{0r})_{r \in R}$ des paramètres des $(S_{1r})_{r \in R}$.

Soit $\overline{e}_1$ la cible d'empreinte carbone française.

Soit $d_0$ et $D_1$ la demande française respectivement en $t=0$ et en $t=1$.

$D_1$ est une variable aléatoire. L'ADEME propose 4 scénarios de croissance. Soit $G$ la croissance économique moyenne sur la période 2015-2050, une variable aléatoire. Nous choisissons cette variable aléatoire comme suivant une  loi uniforme sur ces 4 scénarios : $1\%$, $1.3\%$, $1.5\%$, $1.8\%$ ; 

$D_1 = (1+G)^{2050-2015}*d_0$.

En cas de dépassement, le pays recourt à l'achat de quotas : le montant de quotas achetés est appelé $V$.

La demande est nécessairement satisfaite par les quantités produites et importées, malgré le caractère aléatoire de $D_1$. Il n' y a donc pas de variable de décision supplémentaire à introduire.


Critère : 

$$ \underset{q_0, V}{\text{min}} \: \left( \mathbb{E}\left[ V  \right] \right) $$

sous contraintes que :

$$ \sum_{r \in R} q_{0r} \geq 1 $$
$$ \sum_{r \in R} q_{0r}* D_1 * S_{1r} \leq \overline{e}_1 - V $$

$$\sigma ((q_{0r})_{r \in R}) = \{ \empty , \Omega \}$$

$$\sigma (V) \subset \sigma \left( (S_r)_{r \in R}, D_1 \right) $$
