Big Data C'est quoi ?
=====================
  C'est l'utilisation de gros volumes de données à des fins de visualisation, d'analyse et de décisions.

Big Data Pourquoi ?
===================
  - marketing (ciblage commercial)
  - finance (scoring des dossiers de crédit)
  - détection de fraude / anomalies
  - predictions

Les métiers du Big Data
=======================

- Data Analyst : interrogation et visualisation de petit volumes de donnés, création de rapports.
- Business Intelligence Developper : mise en place de la collecte, intégration et diffusion des donnés.
- Data Engineer : DevOps qui met en place l'infra BigData, facilite et optimise l'accès au données.
- Data Architect: Data Engineer avancé, ayant des connaissances plus vaste et pointus, sur les architectures Big Data (clusters, ...).
- Data Scientist : transforme les données en informations qui ont de la valeur. C'est un Data Analyst dopé au BigData et au Machine Learning.

Big Data Comment ?
==================

## Architecture Lambda #
Une architecture lambda se compose de 3 couches:
- batch layer
- speed layer
- serving layer

![lambda architecure](https://user.oc-static.com/upload/2017/12/14/15132725019668_lambda.jpeg "Architecture Lambda")

### Batch layer
Cette couche inclus :

- master Dataset (stockage des données brutes). C'est la source de vérité
- calculs distribués

La master Dataset est contenu dans un data lake (grâce par ex. à HDFS).
Afin d'éviter la corruption des données, on stocke le schéma des données avec elles.
ex: JSON n'inclus pas le schéma alors que Apache Avro, oui.

### Serving layer
Cette couche doit supporter :

- écriture par lot
- lecture aléatoire

Techniquement, il s'agit d'une base NoSQL parmis Cassandra, MongoDb, ElascticSearch,...
Les base SQL ne passent pas à l'échelle horizontalement, et les base clé/valeur n'ont pas (en général) d'index.

La serving layer est un vue des données traités par la batch layer.

### Speed layer

Dans cette couche, on stocke les données de façon dénormalisées et aggrégés, pour un accès rapide.
Les données sont effacées une fois disponibles, dans la serving layer.

## Base de données

D'après le théorème de CAP, formulé par Eric A. Brewer :
> Dans toute base de données, vous ne pouvez respecter au plus que 2 propriétés parmi la cohérence, la disponibilité et la distribution.

![triangleCAP](https://user.oc-static.com/upload/2017/05/26/14958217637026_triangleCAP.png "Triangle de CAP")

### familles de bases NoSQL

- clé/valeur : utilisé pour son efficacité et sa simplicité. Pas de langage de requête, il faut connaître la clé. Utilisations : détection de fraude en temps réel, IoT, e-commerce, gestion de cache, transactions rapides, fichiers de logs, chat.

- colonnes (HBase, Spark SQL, Elastic Search, ...) : utilisé pour les traitements sur des colonnes comme les agrégats. Adaptée à de gros calculs analytiques. Utilisations : Comptage (vote en ligne, compteur, etc), journalisation, recherche de produits dans une catégorie, reporting à large échelle.

- documents (MongoDB, Cassandra, ...): utilisé pour manipuler des documents avec une structure complexe. Utilisations : gestion de contenu (bibliothèques numériques, collections de produits, dépôts de logiciels, collections multimédia, etc.), framework stockant des objets, collection d’événements complexes, gestion des historiques d’utilisateurs sur réseaux sociaux.

- graphes (Neo4j, OrientDB): utilisé pour résoudre les problèmes de corrélations entre les éléments. Utilisations : réseaux sociaux (recommandation, plus court chemin, cluster...), réseaux SIG (routes, réseau électrique, fret...), web social (Linked Data).


## Principaux langages utilisés ##

- R (implementation de S issue des laboratoires Bell) utilisé par les statisticiens, data miner, data sctientist. Né en 93, 1ere release en 1995. Avangtage: Simpmlicité, utilisation possbile de librairies Python. Inconveniant: mauvaises performances.
- Python. 1ere release 1991. Avantage: nombreuses bibliothèques de ML, ainsi qu'utilisation possible des bibliotheques R. Inconveniant : performance inférieure à Scala.
- Scala: 1ere release en 2004. Inconvenient: courbe d'aprentissage longue, avantage: performance

## Outillage ##

![Plan du paysage BigData](https://user.oc-static.com/upload/2017/03/20/14900195310816_Data-Platform-Map.jpeg)

- Rstudio (IDE pour R) execution sur un seul serveur
- Tensorflow (ML lib for Python)
- Apache Hadoop (applications distribuées)
- Apache Storm (traitement de flux distribués)
- Apache Spark (calcul distribué) API: Scala, Java, Python, R, Javascript ( projet indépendant [EclairJS](https://github.com/EclairJS/eclairjs) ). En moyenne 10x plus rapide que HDP Map Reduce et jusqu'à 100x si la RAM peut absorber toutes les données.
- Apache Kafka Stream (pour des cas simples, pas de ML)

### Apache Storm
  
Écrit en Clojure (dialect Lisp compilant en bytecode Java, javascript, ou bytecode .NET). Une *topology* (application Storm) traite des flux, en provenance de *Spouts* avec des *Bolts* (workers), ou des micro-batch (avec le plugin Trident).

## Fonctionnement de Spark ##

Spark fonctionne au sein d'un cluster de machines (ou containers).
Il a besoin d'un gestionnaire de ressources parmis : Spark Standalone, Apache YARN, Apache Mesos, k8s.
Spark est réparti entre un (ou plusieurs) driver et des workers
Le driver est responsable de l'execution d'une app (le main), et délegue l'execution des calculs (les fonctions) aux workers via les libs fournis.
Libs Spark: Spark SQL, Spark Streaming, Spark GraphX, Spark MLlib

## Se former ##

- Apprendre Scala : [Programming in Scala, 3rd Edition](https://libgen.pw/download/book/5a1f058c3a044650f51255eb)

- Aller plus loin, en Scala (Optionel) : [Functional programming in scala](https://libgen.pw/download/book/5a1f05453a044650f50e3aba)

- Prendre en main Spark : [Learning Spark: Lightning-Fast Big Data Analysis](https://libgen.pw/download/book/5a1f054d3a044650f50eb805)


Machine Learning (ML)
====================

## C'est quoi le ML ##

Création d'un modèle à partir d'un gros volume de données puis utilisation du modèle pour une prédiction quantitative (régression), ou une prédiction qualitative (classification)

3 Types de modèles:

- Predictif : prédire des évenements futurs ou classifier des observations via apprentissage supervisé.
- Descriptif : groupement des observations par similitudes via apprentissage non supervisé (clustering).
- Adaptif : généré via l'apprentissage par renforcement (un ou plusieurs agent de prise de décision, qui recommendent ou exécutent des actions)

Les modèles prédictifs peuvent être subdivisé en 2 catégories : 
- modèles génératifs (généralement introduit par la règle de Bayes, fondation de la Classification Naïve Bayésienne)
- modèles discriminants


## Le clustering ##

Les algorithmes de clustering sont le plus souvent utilisés pour une analyse exploratoire des données.
Ils permettent, par exemple, d'identifier :
-  des clients qui ont des comportements similaires (segmentation de marché)
-  des utilisateurs qui ont des usages similaires d'un outil
-  des communautés dans des réseaux sociaux
-  des motifs récurrents dans des transactions financières.

On peut établir la performance d'un clustering, en mesurant la séparation (S) inter-cluster (que l'on veut grande), et l'homogénéité (T, pour tightness) intra-clusters (que l'on veut petite, c'est à dire une petite distance entre les points).
On peut également utiliser un seul critère qui s'appuie sur S et T : l'indice de Davies-Bouldin (D).
Ou bien le coefficient de silhouette, qui détermine si chaque point est à la fois proche des points du cluster auquel il appartient, et s'il est loin des points des autres clusters.

Une autre mesure à prendre en compte et la stabilité des clusters.
Plusieurs exécutions, avec des initialisations ou des sous-ensembles de données différents, donnent-elles les mêmes résultats ?

Le clustering hierarchique procède par itération, pour agglomérer (dans le cas d'un clustering agglomératif) les clusters les plus proches (calcule de la distance via une méthode de lien, ou 'linkage method'), avec au départ, un cluster par point, pour obtenir un cluster pour tout les points. Dans le cas du clustering divisif, le processus est inversé. Ce qui permet d'obtenir un arbre, représenté par un dendogramme. Son coût élevé (quadratique), le réserve plutôt à un petit jeu de données.

## Deep Learning ##

Le deep learning est une spécialité du machine learning qui s'appuie sur les réseaux de neurones artificiels profonds, et des algorithmes d'analyse discriminante et apprenants. Profond car il s'agit d'un traitement multi-couche. Chaque couche correspondant à un niveaux différent, d’abstraction des données.

Le perceptron multi-couche (ou multilayer perceptron, "MLP), permet de modéliser des fonctions complexes, en empilant des perceptrons (attribution de poids à chaque variable, puis utilisation d'une fonction d'activation, sur la somme des variables pondérées).

### algo de deep learning ###

Le bagging (pour "bootstrap aggregation") utilise le méthode de bootstrap (génération de nouveau dataset par réenchantillonnage avec replacement), afin de combiner des apprenants faibles (modèle à forte variance). Les différents peuvent être créés en parallèle. Une prédiction est ensuite effectuée par un vote à la majorité (classification), ou par une moyenne (régression).
Il permet de réduire la variance des estimateurs, pour une meilleure performance et stabilité.

Les forêts aléatoires utilise également le bootstraping. Mais elles utilisent un sous-ensemble de features (définition d'un hyperparamètre qui détermine le nombre de features à utiliser pour chaque arbre) pour créer chaque arbre de décision.
Augmenter l'hyperparamètre permet de réduire la corrélation entre chaque arbre, pour de meilleures performances, mais au détriment de la vitesse d'éxécution, puisque le modèle est plus complexe.

Le gradient boosting, est un méta-algorithme (comme le bagging), qui va créer des modèles de manière séquentielle, par itération. A chaque itération, on applique une fonction de perte (pondération des observations, pour l'adaboost), pour créer un nouveau modèle plus performant.


Les réseaux de neurones récurrents (Recurrent neural network, RNN) sont une famille de réseau de neurones qui traite les données de façon séquentielles. Prise en compte de l'état préceant (complet), à chaque nouvelle prédiction. Cette architecture pose des problèmes de gradient (disparition et explosion).
Pour palier les problèmes des RNN, il existe les LSTM Networks (Long Short Term Memory Networks). Ils utilisent des "gates" qui déterminent l'importance des entrées, pour enregister ou non, l'information qui en sort.

Les Réseaux de neurones convolutifs (Convolutional Neural Network, CNN), sont utilisés pour le classement de données visuelles. Ils se constitue de 4 couches:
- convolution : repérer la présence d'un ensemble de features
- pooling : réduction du nombre de paramètres et de calculs dans le réseau.
- correction ReLU (Rectified Linear Units) : fonction d'activation
- fully connected : c'est la dernière couche dans tous les réseau de neurones. C'est somme des entrées pondérées, passé dans une fonction d'activation (logistique, si classification binaire, softmax, si classification mutli-classe)

Le Transfer Learning (ou apprentissage par transfert), permet de réutiliser un réseau pré-entrainé (de préférence sur un problème procher). Il peut exploiter un réseau suivant 3 stratégies :
- fine-tuning total : remplacement de la dernière couche (fully-connected) par un nouveau classifieur, puis entrainements de toutes les couches avec les nouvelles images. Utilisation avec une nouvelle collection de grande taille
- extractions de features : on retire la couche fully-connected, réutilisation des features, et on fixe les autres paramètres. Utilisé avec une petite collection, similaire.
- fine-tuning partiel : mélange des stratégies 1 et 2, en gardant certains paramètres (couches basses). Utilisé avec une petite collection, très différente.


## Algorithmes ##

- Supervisés : Apprentissage via un classement des données par l'humain
  * Arbre de décision (Decision trees)
  * Réseaux Bayésiens (Bayesian network or Probabilistic DAG model)
  * Méthodes des moindres carrés, permet d'effectuer principalement une Regression linéaire (Least squares)
  * Regression logistique (Logistic Regression) : classification binaire (en pourcentage de confiance). Idem régression linéaire avec un facteur sigmoid en plus.
  * Séparateurs à vaste marge (SVM - Support Vector Machine) : classification binaire (bon pour classification non linéaire)
  * Méthode des ensembles

- Non Supervisés: Apprentissage autonomne, sans retour humain. Création de class par similitudes
  * Regroupement d'algorithmes
  * Analyse en composante principale (PCA - Principal component analysis) : décorrélation des variables (features) afin d'en réduire leur nombre.
  * t-SNE (t-distributed stochastic neighbor embedding) : permet de visualiser des données à grandes dimensions non-linéaires sur 2 ou 3 dimensions, afin de repérer des structures locales intéressantes pour le travail de modélisation.
  * Décomposition en valeurs singulière (SVD - Singular-value decomposition)
  * Analyse en composantes indépendantes (ICA - Independent Component Analysis)
  * K-moyennes : Effectue une classification

Il existe aussi 2 autres familles d'algo, peu utilisés :
- semi-supervised learning : prend en entrée des données annotés, et d'autres non.
- reinforcement learning : basé sur un cycle expéreience / récompense. Amélioration à chaque itération.

### Utilisation des algo ###

#### Classification linéaire multi-classe ####

Il est possible d'utiliser un classifieur linéaire pour faire de la classification multi-classe.
Pour cela, on a 2 approches : 
- one-versus-rest (OVR ou OVA pour one-versus-all).
 On va créer autant de classifieur que de classe en comparant chaque classe avec l'union des autres classes.
 On prédit la classe pour laquelle la fonction de décision est maximale.

- one-versus-one (OVO).
 On va créer K(K -1) / 2 classifieurs, qui vont comparer chaque classe une à une.
 On prédit par un vote de la majorité (la classe prédite par le plus grand nombre de classifieurs).


#### Utiliser un algo linéaire pour résoudre un problème non-linéaire ####

On peut redécrire les données dans un nouvel espace de redescription H (pour Hilbert) grâce à une application [Φ](#Φ "Phi"). L'espace de redescription sera généralement beaucoup plus grand que l'espace initial.

On peut également utiliser l'astuce du noyau (kernel trick), noté k,  afin d'éviter le calcul de [Φ](#Φ "Phi").
Il existe plusieurs types de noyaux (linéaire, polynomial, RBF, ...)

### Fonctionnement des algo ###


#### Régression linéaire ####

La régression linéaire n'est pas un algo mais un type de modèle.

Ce type de modèle peut être estimé par :
- maximisation de vraissemblance
- méthode des moindres carrés (réduction du carré des erreurs)
- inférence baysienne

Dans le cas le plus standard, on considère les points indépendants et identiquement distribués (independent and identically distributed, i.i.d.).

Lorsque les variables sont corrélées, ou lorsqu'on a plus de variables que d'observations, on risque de faire du sur-apprentissage. Pour limiter ce risque, on va ajouter un hyperparamètre  : un coefficient de régularisation. La régularisation, consiste à contrôler simultanément l'erreur du modèle sur le jeu d'entraînement et la complexité du modèle.
Parmis les algo de régularisation, on trouve :
- régression ridge : réduit l'amplitude des coefficients d'une régression linéaire
- lasso : annule certains coefficients
- elastic net : combinaison des 2 précédents

#### SVM ####

L'algo SVM pour "Séparatrices à vaste marge" ou "Machines à Vecteurs de Support" ou "Support Vector Machine", aussi appelé SVC pour "Support Vector Classification", est un algo de classification linéaire visant à maximiser la marge entre chaque classe.
Il s'appuie sur les points (vecteurs de support) les plus proches de l'hyperplan séparateur.
Il existe 2 formulations permettant d'obtenir la SVM : 
- On optimise le primal si on a de faible dimensions.
- On optimise le dual si on a peu d'observations.

Il existe également la régression SVM aussi appelée SVR pour "Support Vector Regression" lorsqu'on l'utilise pour résoudre des problèmes de régression.


#### K plus proches voisins (k-NN, k-nearest neighbors) ####

Algo de classification, peu utilisé, car très coûteux. Il nécessite de garder en mémoire toutes les observations (memory-based).
Il ne s'agit pas d'un algo paramétrique, donc pas d'apprentissage. Il associe à une observation la même étiquette que la majorité de ses k plus proches points d'entrainement. On fera varier k (hyperparamètre) pour obtenir la plus faible erreur.

Note: On parle d'hyperparamètre lorsqu'il s'agit d'un paramètre de l'algorithme d'apprentissage, et de paramètre (souvant noté [𝜽](#𝜽 "theta")), lorsqu'il s'agit d'un paramètre du modèle, trouvé par apprentissage.

#### K-moyennes (K-means) ####

Partionnement des observations en K partitions (clusters).
Chaque cluster possède un Centroid qui est le barycentre de ses membres. Chaque centroid est recalculé de façon itérative (et chaque point réassigné au centroid le plus proche), afin de minimisé le total des distances de chaque point à son centroid.
La phase d'initialisation des centroid peut-être optimisé via l'algo de Agha-Ashour.  De même, le critère de similarité (distance euclidienne vs cosinus) ne doit pas être négligé. Il peut avoir un impact sur la densité du cluster.

#### DBSCAN ####

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) partitionne les observations par densité. Pour chaque observation il construit son epsilon-voisinage. Si son ε-voisinage contient le minimum de voisins, chaque voisin est ajouté au cluster. Sinon, il est considéré comme du bruit.

Inconvéniants : 
- DBSCAN est difficile à utiliser en très grande dimension
- le choix des paramètres ε et n-min (nombre de voisins minimums) peut être délicat.
- ne trouvera pas de cluster de densité différente

Avantages :
- efficace en temps de calcul
- pas besoin de prédifinir le nombre de clusters
- permet de trouver des cluster de forme arbitraire

#### Espérance-maximisation (Expectation-maximization, EM) ####

Contrairement au K-means, il prends en compte (par déduction) les valeurs non observés (latentes).
K-means is a special case of the EM for Gaussian mixtures.


#### Analyse en composantes principales (Principal components analysis, PCA) ####

Transforme et extrait les caractéristiques les plus critiques (en terme de variance).
Extrait les vecteurs et valeurs propres, à partir de la matrice de covariance des observations. Puis les classes par valeurs propres (composante principale)
L'ACP cherche à maximiser la variance de X selon des directions orthogonales.


#### kPCA ####

Utilise un noyau (kernel trick) pour permettre le plongement d'un algorithme linéaire dans une variété non-linéaire. Rarement utilisée dans sa forme première : elle nécessite de garder en mémoire la matrice K(x,y) (son stockage nécessite le carré de la taille des données).

#### Analyse factorielle ####

L'analyse factorielle cherche à modéliser la structure de la covariance des variables observées et ne définit pas nécessairement des axes orthogonaux.

#### Naïve Bayes ####

L'inférence bayésienne calcule les probabilités de diverses causes hypothétiques à partir de l'observation des conséquences connues.
Le raisonnement bayésien interprète la probabilité comme le degré de confiance a accorder à une cause hypothétique.

Naïve Bayes considère chaque feature comme indépendante.
Cette restriction est assouplie par le modèle Hidden Naïve Bayes (HNB), qui utilise l'information mutuelle conditionnelle, pour décrire l'interdépendance entre certaines features.

La Classification mutltinomiale naïve bayésienne est particulièrement adapté à la fouille de textes (text mining).


#### Multivariate Bernoulli classification ####

idem Naïve bayes, mais au lieu d'ignorer l'abscence d'une observation dans une feature, il la pénalise.


## Workflow ##

### 1. Préparation des données ###

#### Extracting features ####

A partir des données brutes, il faut obtenir des valeurs numériques.
Plusieurs méthodes existent, et dépendent de la nature des données (texte, image, ...)

Afin que chaque variable ait la même importance, il faut standardiser chaque variable, pour que chacune ait le même ordre de grandeur.

#### Feature engineering / learning ####

2 méthodes :

- feature engineering : Utilisation de la connaissance métier des données, pour transformer les variables de départ (difficile et coûteux). 
- feature learning : 
Création d'un vecteur de caractéristiques (feature vector) de façon automatique, par aprentissage. L'apprentissage peut être supervisé ou non supervisé.


Sélection des données minimum nécessaires à l'aprentissage (élémination des doublons, et des données non pertinentes). Techniques utilisés : PCA, Régression des moindres carrés partiels dit "Régression PLS" (Partial Least Squares regression), analyse factorielle ...

Filtrer le bruit avec des techniques comme : la moyenne mobile (moving average), la Transformation de Fourier (rapide), le filtre de Kalman, un processus autorégressif (pour les séries temporelles), un ajustement de courbe (moindre carrés, ...)

Pour réduire le nombre de features, on peut soit retrouver les dimensions principales, qui peuvent être une combinaison de différentes observations, ou retrouver la variété sous-jacente (manifold). Dans ce dernier cas, on parle de manifold learning.

La réduction de dimension permet de résoudre plusieurs problématiques :
- le fléau de la dimension (curse of dimensionality), qui est la difficulté d'apprentissage en haute dimension.
- la visualisation des données. Il difficile de lire un graphique au delà de 3-4 dimensions
- la réduction des coûts de calcul, de stockage et d'acquisition des données.
 
##### Analyse de données textuelles #####

Plusieurs techniques sont généralement mises en oeuvres pour l'analyse de texte :
- récupération du texte : par scraping ou téléchargement de fichiers texte.
- tokenisation : séparation du texte en mots (unigramme), ou n-gram.
- normalisation :
  * filtrer les caractères alphanumériques par une regex.
  * suppression des stopwords : les stopwords, sont les mots les plus fréquents de la langue (à, et, de, ...)
  * lemmatisation : remplacement des mots par leur forme canonique (infinitif, masculin singulier, ...)
  * stemming : remplacement des mots par leur racine (suppression des préfixes, suffixes, ...)

On peut représenter un document, par un sac de mots (bag-of-words, ou bag-of-ngrams) :
- en comptant la fréquence d'apparition des mots dans le document.
- en pondérer cette fréquence par rapport à l'ensemble des documents via tf-idf (Term-Frequency - Inverse Document Frequency)

On peut également représenter un document par un plongement de mots (word embeddings).
Cette méthode prend en compte le contexte, en réprésentant les mots par des vecteurs, dans un espace avec une forme de similarité entre eux (probabiliste).
Par ex. vec("Madrid") - vec("Spain") + vec("France") donne une position dont le vecteur le plus proche est vec("Paris")
On utilisera principalement 2 algos d'apprentissage (word2vec):
- Continuous Bag of Words (CBOW) : entraîne le réseau de neurones pour prédire un mot en fonction de son contexte (mots avant/après).
- skip-gram : prédiction du contexte en fonction du mot.

word2vec, permet la création de word embeddings.

On peut également utiliser d'autres méthodes de plongement, tels que [gloVe](https://nlp.stanford.edu/projects/glove/) et [FastText](https://fasttext.cc/). Ou une simple [décomposition SVD sur une matrice PMI](http://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/)

Entraîner son [propre embedding](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/word2vec.ipynb), permet d'avoir un plongement spécifique à notre corpus (plus performant).

Extraire les informations :
- NER (Named Enity Recognition) : reconnaître des personnes, endroits, entreprises, ...
- Extraction des relations : extraire des relations sémantiques (familiales, spatiales)
- Extraction d'évenements : extraction des actions qui arrivent aux entités ("le président à déclaré X dans son discours", "X a augmenté son CA de 20 %", ...)
- POS Tagging (Part-of-Speech Tagging) : identifier la nature grammatical des mots (nom, verbe, ...)

modélisation de sujet automatique non supervisée
- LDA (Latent Dirichlet Allocation)
- NMF (Negative Matrix Factorisation)


#### Gérer les jeux de données désiquilibrés ####

(https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)
- rééchantillonner les données (over-sampling ou under-sampling)
- stratifier les données pour la validation croisée
- changer l'indicateur de performances


### 2. Création du modèle ###

Avant d'entrainer le modèle, on divise le jeu de données en portions. Le training set (~80%), et le testing set (~20%). Le testing set ne sera pas utilisé pour la création du modèle, mais pour vérifier qu'il fonctionne sur de nouvelles données.

3 principaux algo non-supervisés : 
- K-means: Clustering observed features
- Expectation-maximization (EM): Clustering observed and latent features
- Principal components analysis (PCA): Reducing the dimension of the model

La création d'un modèle supervisé se fait par optimisation des paramètres notés [𝜽](#𝜽 "theta") de l'ago choisi. L'optimisation consiste à converger vers le minimum de la fonction loss (perte d'information), en prenant par exemple, la distance euclidienne. Pour cela, on peut minimiser le risque empirique (somme des erreurs constatés), ou maximiser la vraissemblance.

#### Bias-variance tradeoff ####

Biais : erreur provenant d’hypothèses erronées. Un biais trop élevé conduit à un modèle trop simpliste, qui ne sera pas représentatif du comportement observé.
Variance : erreur due à la sensibilité aux petites fluctuations de l’échantillon. Une trop grande variance considérera du bruit comme faisant partie du modèle.

Overfitting = biais faible et variance élevée
Underfitting = biais élevé et variance faible

Afin de minimiser l'erreur totale du modèle, il faut trouver le compromis biais-variance (bias-variance tradeoff). Ce qui reviens à trouver un modèle ni trop, ni trop peu complexe.
On peut réduire la complexité d'un modèle (et donc sa variance) en réduisant les dimensions. On peut également réduire la variance en utilisant des méthodes ensemblistes, qui combinent et aggrègent plusieurs modèles à haute variance.
Avec un modèle à haute variance, on est trop dépendant des données d'entrainement. Tandis qu'avec un modèle trop simple, on ne capture pas toute la complexité du phénomène.
Une haute variance conduit à du surapprentissage (overfitting). Alors qu'un biais élevé conduit à du sous-apprestissage (underfitting).

### 3. Évaluation du modèle ###

#### Évaluation d'un algorithme de classification binaire ####
On peut utiliser la matrice de confusion qui représente :
- les vrais positifs ou TP (true positives)
- les vrais négatifs ou TN (true negatives)
- les faux positifs ou FP (false positives) ou erreur de type I
- les faux négatifs ou FN (false negatives) erreur de type II

Validation du modèle par des mesures selon des critères de sensibilité (appelé aussi *rappel* ou *recall*), qui est le taux de vrais positifs (TP ÷ nombre réels de positifs), et de précision (TP ÷ nombre de positifs prédits).
La mesure F1 (ou F1 score) est la moyenne harmonique du rappel et de la précision: 2 x Precision x Rappel ÷ (Precision + Rappel).
On peut également calculer la spécificité (taux de vrais négatifs)

On peut utiliser des algo comme le kNN en remplaçant une décision à la majorité par une décision au ratio. Il faut alors décider du seuil à partir duquel attribuer un évenement à une classe positive. Pour déterminer ce seuil, on peut uiliser la courbe ROC (Receiver-Operator Characteristic). Cette courbe fait correspondre la sensibilité et la (anti-)spécificité, pour chaque interval de seuil. On choisi ensuite le seuil, en se fixant la spécificité ou la sensibilité désirée.
On peut choisir entre plusieurs modèles en comparant leur air sous la courbe, appelée AUROC (Area Under the ROC).

On peut également utilisé la courbe précision-rappel (PR curve), ou la courbe lift (utilisé surtout dans le ciblage marketing).

On peut utiliser une approcher naïve pour avoir un point de comparaison pour évaluer des modèles. Pour une classification, on peut :
- retourner toujours la même classe
- retourner une classe aléatoire
- retourner un score aléatoire,  puis utiliser un seuil.

#### Évaluation d'un algo de régression ####

On mesure l'erreur moyenne. Avec f(xi), valeur prédite et yi, valeur réelle :
- somme des carrés des résidus, ou Residual Sum of Squares, RSS = Σ(f(xi) - yi)²
- erreur quadratique moyenne ou Mean Squared Error, MSE = RSS ÷ n
- Root Mean Squared Error, RMSE = √MSE
- Root Mean Squared Log Error, RMSLE = √Σ(log(f(xi) + 1) - log(yi + 1))² ÷ n
- erreur carré relative, Relative Squared Error, RSE = Σ(yi - f(xi))² ÷ Σ(yi - ȳ)² avec ȳ==Σyi ÷ n
- le coefficient de détermination, R² = 1 - RSE

#### Évaluation d'un algorithme ... ####
La qualité de l'estimation est estimée par validation croisée : K-fold cross-validation

Pour sélectionner les valeurs des hyperparamètres, on fait un grid search, afin de tester toutes les valeurs pertinates. Comme alternative, on peut utiliser "le critère d'information d'Akaike" (Akaike information criterion, AIC) qui repose sur un compromis entre la qualité de l'ajustement et la complexité du modèle.


### 4. Ajustement du modèle ###

- lorsque le modèle donne de mauvais résultats sur les données de d'entrainement, on a sous-ajustement des données (underfitting). Le modèle n'a pas réussi à saisir la relation entre les features et les labels.

- lorsque le modèle effectue une bonne préduction sur les données d'entrainement, mais ne résussi pas sur de nouvelles données, on a un sur-ajustement (overfittting). Le modèle n'a pas réussi à généraliser.

- lorsque le modèle est mauvais, à la fois, sur les données d'entrainement et de test, on vérifiera que la quantité de données est suffisante pour capter la complexité du modèle.


#### Underfitting

Face à un sous-ajustement, on essaiera d'ajouter de nouvelles features, ou de trouver des features plus représentatives. On pourra également diminuer le degré de régularisation.

#### Overfitting

Face à un sur-ajustement, on utilisera moins de features. Et, on augmentera le degré de régularisation.


### 5. Utilisation du modèle ###

Une fois le modèle créé, on peut réaliser des prédictions sur de nouvelles données.
Ces données doivent être préparés pour pouvoir être traitées par le modèle, de la même façon que les données d'entrainement pour constuire le modèle.


## Se former ##

### Se former de zéro ###

0. Pré-requis en maths
 - [dérivés](https://www.methodemaths.fr/derivee/)
 - [intégrales](https://www.methodemaths.fr/integrale/)
 - [probabilités (OC)](https://openclassrooms.com/fr/courses/4525296-maitrisez-les-bases-des-probabilites)
 - Matrices

Pour suivre les cours d'Open Classrooms (OC) sur le Machine Learning, des base en [Python](https://openclassrooms.com/courses/demarrez-votre-projet-avec-python) et en particulier sur les librairies [Numpy, Matplotlib et Pandas](https://openclassrooms.com/courses/decouvrez-les-librairies-python-pour-la-data-science) est recommandé

1. [Initiez-vous au machine learning (OC)](https://mooc-francophone.com/cours/initiez-vous-au-machine-learning/)
  Prérequis : connaissance de C# et XAML recommandés
2. [Explorez vos données avec des algorithmes (OC)](https://mooc-francophone.com/cours/explorez-vos-donnees-avec-des-algorithmes-non-supervises/)
  Prérequis : Pyhton, notions d'algèbre linéaire, notions de probabilités et statistiques.

3. [Évaluez et améliorez les performances d'un modèle de machine learning (OC)](https://openclassrooms.com/fr/courses/4297211-evaluez-et-ameliorez-les-performances-dun-modele-de-machine-learning/)
  Prérequis : Pyhton, notions d'algèbre linéaire, notions de probabilités et statistiques.

4. [Entraînez un modèle prédictif linéaire (OC)](https://openclassrooms.com/fr/courses/4444646-entrainez-un-modele-predictif-lineaire)

5. [Utilisez des modèles supervisés non linéaires (OC)](https://openclassrooms.com/courses/utilisez-des-modeles-supervises-non-lineaires)

6. [Modélisez vos données avec les méthodes ensemblistes (OC)](https://openclassrooms.com/fr/courses/4470521-modelisez-vos-donnees-avec-les-methodes-ensemblistes)

### S'exercer ###

S'entrainer avec Kaggle. Selon Wikipédia :
> [Kaggle](https://www.kaggle.com/) est une plateforme web organisant des compétitions en science des données. Sur cette plateforme, les entreprises proposent des problèmes en science des données et offrent un prix aux datalogistes obtenant les meilleures performances.
Kaggle permet également d'obtenir des jeux de données sans concourir à une compétition.


Annexe
======

## Codes UTF-8 des symboles

- <a name="𝜽">𝜽</a>: theta, U+1D73D
- <a name="Φ">Φ</a>: Phi, u03A6
- <a name="Σ">Σ</a>: sigma (somme), u3A3
- <a name="√">√</a>: racine carrée, u221A
- <a name="ȳ">ȳ</a>: y surligné (moyenne), u0233
- <a name="÷">÷</a>: signe division, u00F7
