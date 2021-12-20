---
title: Optimisation de la politique commerciale française sous contraintes de la réduction de l'empreinte carbone
authors: Julien Ancel, Théo Mandonnet
---
# Optimisation dans l'incertain
## Politique commerciale optimale française en vue d'atteindre son objectif d'empreinte carbone, en deux pas de temps : 2015 et 2050. 



Soit $s_0$ le vecteur des coefficients d'intensité carbone des différentes régions du monde $r \in R$ avec lesquelles la France commerce en $t=0$, des paramètres connus et fixés : 
$$ s_0 = (s_{0r})_{r \in R} $$

Soit $q_0$ le vecteur des quantités importées par la France depuis chaque région $r \in R$ en $t=0$, des variables de décision :
$$ q_0 = (q_{0r})_{r \in R} $$

Soit $S_1$ le vecteur des coefficients d'intensité carbone des différentes régions $r \in R$ en $t=1$ : 
$$ S_1 = (S_{1r})_{r \in R} $$
où les $S_{1r}$ sont des variables aléatoires.

Soit $q_1$ le vecteur des quantités importées par la France depuis chaque région $r \in R$ en $t=1$, des variables de décision :
$$ q_1 = (q_{1r})_{r \in R} $$

Soit $\overline{e}$ la cible d'empreinte carbone française.

Soit $c_0(q_{0r})$ les coûts associées aux quantités $q_{0r}$ en $t=0$, connus.
Soit $c_1(q_{1r})$ les coûts associées aux quantités $q_{1r}$ en $t=0$, variables aléatoires.

Soit $d_0$ et $d_1$ la demande française respectivement en $t=0$ et en $t=1$.

Prévision de croissance économique autour de $1.5\%$ chaque année. Soit $G$ la croissance moyenne sur la période 2015-2050 une variable aléatoire. Ainsi, $d_1 = G^{2050-2015}*d_0$

En cas de dépassement, le pays recourt à l'achat de quotas : les quotas ont un prix unitaire $p^{quota}$, et le montant dde quotas achetés est appelé $v$.

Critère : 

$$ \underset{q_0, q_1, v}{\text{min}} \: \left( \mathbb{E}\left[ \sum_r( c_0(q_{0r}) + c_1(q_{1r}) + p^{quota}v )  \right] \right) $$

sous contraintes que :

$$ \sum_r q_{1r} + u \geq d_0$$
$$ \sum_r q_{0r} \geq d_1 $$
$$ \sum_r q_r*S_{1r} \leq \overline{e} - v$$
