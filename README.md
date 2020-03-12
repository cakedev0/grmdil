# grmdil
Projet du cours MVA : "Discrete inference and learning"

## Modèle graphique : 

**Format de l'entrée :** 
- 1ère ligne : $n, m$ deux entiers. $n$ le nombre points, $m$ nombre d'arrêtes.
- $n$ lignes suivantes : matrice $P \in [0, 1]^{n \times 6}$. $P_{ic} =$ probabilité que le points $i$ soit de label $c$ (d'après le classifier).
- $m$ lignes suivantes : arrêtes au format $i, j, d_{ij}$ (de type `int, int, float`)

On a $n \approx 10^6$ et $m \approx 5 n$

**Format de la sorties :**
- n lignes : $x \in \{1, ..., 6\}^n$ résulats de la minimisation de l'énergie $E(x)$ avec BP ou TRW. 

**Energie $E(x)$:**

$E(x) = \sum\limits_{i=1}^n f(P_{i, x_i}) +  \sum\limits_{(i, j) \in \mathcal E} g(d_{ij}) \mathbb 1_{x_i \neq x_j} $

Avec $f : [0, 1] \rightarrow \mathbb R$ décroissante et $g : \mathbb R_+ \rightarrow \mathbb R_+$ décroissante. Pour commencer on peut prendre : 

$E(x) = \sum\limits_{i=1}^n - P_{i, x_i} +  \sum\limits_{(i, j) \in \mathcal E} \alpha \frac{1}{d_{ij}} \mathbb 1_{x_i \neq x_j} $ avec $\alpha \in \mathbb R_+$ à régler 

Cette modélisation vient exprimer le fait suivant : "deux points qui sont proches ont une forte chance de partager le même label". Le but de cette modélisation/minimisation est de trouver des labels $x$ meilleurs que ceux du classifier (qui sont les $(\arg\max_c P_{ic})_i$).

**Score** :

Le score est calculer par rapports aux vrais labels $x^{true} \in \{0, 1, ..., 6\}^n$ (le label $0$ représente les "unclassifed". Les points "unclassifed" n'interviennent pas dans le score.) 

$\text{Score}(x^{true}, x) = \frac 1 6 \sum_{c = 1}^6 \text{IoU}(x^{true}, x, c)$ où $\text{IoU}$ signifie Intersection over Union : $$\text{IoU}(x^{true}, x, c) = \frac{ \{i, x^{true}_i = c\} \cap  \{i, x_i = c\} }{\{i, x^{true}_i = c\} \cup  \{i, x_i = c \text{ et } x^{true}_i \neq 0\}}$$
