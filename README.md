# Projet RI (2026)
### Introduction rapide
L'objectif est la reproduction d'expérience décrit sur le papier de recherche "Explicit Search Result Diversification through Sub-Queries"
"Explicit Search Result Diversification through Sub-Querie" (2010) propose un algorithme appelé xQuAD(explicit query aspect diversification),
contrairement aux approches classiques(e.g MMR), il modélise explicitement (presque identique) les aspects de la sub-queries(sous-requête en fr).
### problème introduit
Le problème donné ici se base sur l'ambigüité de requête et ensuite se part sur la diversité et la pertinence de la recherche: Dans la plupart des cas, une requête donnnée par l'utilisateur
est ambiguë pour le moteur, comme par exemple "the mirror" peut être le nom d'un site de reportage journalière ou encore le nom d'un film, les résultats de la recherche sont donc diversifiés.
Sache qu'un moteur classique(e.g.BM25) retourne souvent qu'un seul aspect ce qui est insuffisant pour assurer cette diversité, dans ce document, ils nous décrit un autre algorthme qui permet
prendre en compte cette caractérique.

### formule à connaitre
- le score pour un document: $S(d) = \text{relevance}(d,q) \cdot (\text{diversity part})^{w}$
  - diversity part peut s'exprimer en: $$\sum_{q_i \in Q(q)} \frac{P(q_i \mid q)\, P(d \mid q_i)}{m(q_i)}$$
- le score principal de xQuAD est donné par $$r(d, q, Q(q)) = \sum_{q_i \in Q(q)} \frac{\mathcal{I}_X(q_i, q)\, r(d, q_i)}{m(q_i)}$$
Remarque: xQuAD peut être interpréter de manière probabiliste : $$P(d \mid q) \cdot \sum_i P(q_i \mid q)\, P(d \mid q_i)\, P(\text{novelty})$$
=>c'est une approximation d'un modèle probabilste de couverture d'intention
### implémentation
Cette algorithme est greedy(iteratif) et réalisable en 2 étapes:
```
1. initialisation
   S <- stocke les résultats finaux, vide au départ
   R <- contient l'ensemble de document candidats 
3. boucle itérative
   pour chaque document d appartenant par R
     on calcul le score xQuAD
   choisir le document le plus pertinent selon la masse m(q)
  mettre à jour la couverture des aspects
```
Remarque: forte ressemblance avec MMR-like mais avec des aspects explicites
#### plus de détail:
(petite rappelle sur l'objectif: le but est de construire un classifieur s(q)={d1,d2,..dt} 
avec pertinence(relevance) et diversité(coverage des aspects))
|variable/symbole|signification|type|Dimension/longueur|
|----------------|-------------|----|------------------|
|q|requête(query)|str/vecteur|len(q)|
|d|document|identifiant/objet|len(d)|
|R(q)|classement initial(init ranking)|liste de document|N|
|S(q)|classement final(final ranking)|liste de document|t|
|$q_i$|sous-requêtes(sub-queries)|str/vecteur|len($q_i$)|
|Q(q)|liste de sous-requêtes(list of sub-queries)|liste des sous-requêtes|K|
|R(d,q)|pertinence des documents/requête|float|scalaire|
|R(d, $q_i$ )|pertinence des documents/sous-requête|float|scalaire|
|$I_X(qi,q)$|importance d'aspect|float|scalaire|
|M($q_i$)|couverture accumulée(masse)|float|scalaire|
|$w$|poid diversité|float|scalaire|

-  R(d,q) permet de décomposer terme par terme (pertinence classique (BM25/DPH))
-  R(d,qi) permet de donner la pertinence du document pour chaque aspect(sous-requête), calculée comme une requête normale
-  $I_X(q_i,q)$ avec la somme = 1
-  M($q_i$) permet de mémoriser la couverture(ou masses même notion utilisé dans le document pour simplifier la compréhension), en initialisation, on l'initialise avec des 1 pour éviter la division par 0 et à chaque itération, la mise à jour est fait avec $$M(q_i) \leftarrow M(q_i) + r(d^\*, q_i)$$
-  $1/M(q_i)$ représent "novelty", si un aspect est déjà couvert alors il est pénalisé et si peu couvert alors on considère comme bon
-  w permet de gérer la balance relevance vs la diversité (plus il est proche de 0 plus l'algorithme s'intéresse sur la pertinence plus il est proche de 1 plus il concentre sur la diversité)
  - cas extrêmes: 0-> pure relevance 1-> forte diversité
### 4 fonctions d'importance $I_x$ proposé dans le papier
|caractèristique|formule|description|avantages|limites|
|---|-------|-----------|---------|-------|
|Uniforme|$\mathcal{I}_u(q_i, q)=\frac{1}{\|Q(q)\|},\quad \sum_i \mathcal{I}_u(q_i, q)=1$|il donne même l'importance à tous les aspects, chaque résultat a un poids=1/K, on l'utilise pour baseline|simple et robuste, pas besoin de données externes|ignore la réalité, certains aspects sont beaucoup plus fréquents que les autres|
|Basé sur nombre de document|$\mathcal{I}_N(q_i, q) = \frac{n(q_i)}{\sum_j n(q_j)}$, où $n(q_i)$=nombre de documents récupérés pour $q_i$|plus un aspect retoune de documents plus il est important,l'idée ici est la popularité des aspects, beaucoup de documents pertinents <=> plus de chance que l'utilisateur trouve ce qu'il cherche|facile et reflète la taille d'un aspect|la présence de biais provient d'un aspect vague s'exprime aussi par la massive quantité de documents qui ne sont pas forcément pertinent(e.g recherche sur l'apple peut retourner beaucoup de document sur le fruit mais aussi l'entreprise, donc distribution est biaisée), donc il favorise le bruit et dépend fortement au système de retrieval|
|inspiré du resource selection(fédération de moteurs) - ReDDE| $\mathcal{I}_R(q_i, q) = \sum_{d : r(d,q_i) > 0} r(d,q)\, r(d,q_i)\, n(q_i)$ |similaire à $I_N$, il combine l'importance globale du document, l'importance du document par l'aspect et la taille de l'aspect. Un aspect est important si ses documents sont pertinent pour la requête et pour l'aspect, un document est bon pour q AND $q_i$ alors boots $q_i$|plus intelligent et combine plusieurs signaux|dépend fortement du classement initial et peu amplifier des erreurs|
|CRCS| $\mathcal{I}_R(q_i, q) = \sum_{d : r(d,q_i) > 0} r(d,q)\, r(d,q_i)\, n(q_i)$ , avec j(d,q)=range du document, t=top-k documents à retourner, $\frac{1}{\hat{n}(q_i)}$ nombre de documents de $q_i$ dans top-k|un aspect est important si ses documents apparaissent haut dans le classement $\frac{n(q_i),max_j n(q_j)}$ donne la taille relative, $\sum_{d|r(d,q_i)>0} (r-j(d,q))$ donne le score basé sur position(un document haut classé <=> score élevé), un aspect est important s'il retourne beaucoup de documents et ces documents sont bien classés|très performant en combinant la qualité et la quantité|dépend fortment du classement initial et plus complexe que les 3 précédents|
### Complexité
$\mathcal{O}(t \cdot N \cdot K)$
- t: taille du classement final
- N: nombre de documents
- K: nombre de sous-requête
