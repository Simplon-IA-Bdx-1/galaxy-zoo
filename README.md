# Brief projet fil-rouge classification données numériques: "Give Me Some Credit"



## Description

Participez au Challenge Kaggle [Give Me Some Credit](http://kaggle.com/c/GiveMeSomeCredit/data). Les activités proposées vous aideront à obtenir les meilleurs résultats possibles. Vous serez amené à travailler en groupes de taille variable mais vous serez évalué individuellement, sur votre propre code, rapport d'analyse, et vos propres résultats.

Tags: classification, Kaggle, machine learning, données numériques

### Pré-requis

* Concepts de base de l'apprentissage supervisé
* Python
* Docker

### Livrables

Repo contenant:

* Notebooks Jupyter
* Configuration Docker permettant de servir vos notebooks Jupyter, avec toutes les dépendances nécessaires
* Fichier README.md modifié:
  * Pour chaque partie du projet, expliquez où se trouve ce que vous avez fait et mettez un lien vers le notebook approprié
  * Indiquez votre score Kaggle

Le code et les explications fournis doivent permettre de reproduire votre travail et votre score Kaggle.

### Evaluation

* Score Kaggle
* Code:
  * Bon fonctionnement: je lance les commandes / scripts / notebooks indiqués par l'apprenant, ça marche et je suis capable de reproduire le score Kaggle
  * Découpage du code en plusieurs notebooks
  * Respect des best practices Python
  * Respect du "Don't Repeat Yourself"
* Versionnage:
  * Fréquence des commits? Commentaires? (EN ou FR?)
* Rapport d'analyse et documentation:
  * Qualité et nombre de visualisations pertinentes
  * Ratio blocs de texte / blocs de code dans notebooks créés
  * Qualité des explications. Concision, clarté.

### Compétences visées

* C12. Constituer un jeu de données exploitable de manière à entraîner un modèle d'apprentissage en utilisant la méthodologie et/ou l'outil approprié en fonction des standards de l'écosystème
  * Objectifs pédagogiques:
    * Nettoyage et traitement des données exploitables à l’aide d’une bibliothèque logicielle => pandas
  * Niveau(x) d'acquisition visé(s): 1, 2

* C13. Interpréter les données grâce à des outils de visualisation de données en vue d’expliquer les caractéristiques du jeu de données
  * Objectifs pédagogiques:
    * Visualisation des données à l'aide d'outils => matplotlib et BigML; voir partie [Exploration des données](#Exploration-des-données)
  * Niveau(x) d'acquisition visé(s): 1, 2

* C14. Exploiter un modèle d’apprentissage supervisé ou non supervisé permettant la classification ou la prédiction d’une variable en fonction des données disponibles et des outils sélectionnés
  * Objectifs pédagogiques:
    * Entraînement et exploitation d’un modèle d'apprentissage supervisé à l’aide d’outils préalablement sélectionnés => scikit-learn et BigML
    * Classification ou prédiction d’une variable à partir d’un modèle d’apprentissage supervisé => classification
  * Niveau(x) d'acquisition visé(s): 1, 2

* C15. Améliorer les performances d’un modèle d’apprentissage à l’aide d’une évaluation de la qualité des données et de la technique de modélisation afin de réduire les biais et les anomalies de résultats
  * Objectifs pédagogiques:
    * Evaluation de la performance d’un modèle d’apprentissage avec les métriques standards et spécifiques => classification, matrice de confusion, gain total, AUC
    * Amélioration de données d’apprentissage d’après une analyse des métriques de performance => partie [Analyse de modèle](#Analyse-de-modèle)
  * Niveau(x) d'acquisition visé(s): 1, 2

* C19. Développer une application et/ou des fonctionnalités utilisant le traitement de données généré par l’IA de manière à être exploitable par le client/utilisateur final
  * Utilisation d’un gestionnaire de conteneur => Docker
  * Versionnage du code source => Git

## Démarrage de l'environnement de développement Jupyter

Nous utiliserons un environnement Jupyter fourni par un conteneur Docker. L'image utilisée est définie par le [Dockerfile](docker/Dockerfile) fourni. Celui-ci est basé sur [handson-ml2](https://github.com/ageron/handson-ml2/tree/master/docker) et il installe les packages Python listés dans [requirements.txt](requirements.txt). Ces fichiers sont à customiser au besoin.

1. Créez un fichier `docker/auth.env` basé sur `docker/auth-sample.env`, qui contiendra vos noms d'utilisateur et clés d'API pour [BigML](https://bigml.com) et Kaggle.
2. A partir du dossier `docker/`, exécutez la commande shell:

    ```bash
    docker-compose up
    ```

3. Ouvrez votre navigateur à l'adresse qui s'affiche suite à l'exécution de la commande précédente, puis cliquez sur [Intro-Jupyter.ipynb](Intro-Jupyter.ipynb) dans la liste.

## Exploration des données

* Nombre de features. Nombre de lignes dans train ("train full") et test.
* Inspectez (sous-ensemble des) données d'apprentissage en les parcourant dans un tableur et via la visualisation en histogrammes (sous Kaggle, BigML web, ou avec pandas profiling). Que remarquez-vous? (Mentionnez noms de features et ids d'inputs notables.)
* Partage de train full en train et validation.
* Inspection du train set (histogrammes, statistiques des distributions).

## Préparation des données de train, val et test (avec [Pandas](http://pandas.pydata.org))

* Remplacement valeurs manquantes / erronées / aberrantes
* Ajouts de features

Note: vous pouvez utiliser le notebook [Intro-Pandas.ipynb](Intro-Pandas.ipynb) pour vous familiariser avec la librairie Pandas.

## Création et sélection de modèle (avec [BigML Python](http://bigml.readthedocs.io))

* Comparer AUC de `ensemble` et de `deepnet` sur les données préparées (apprentissage sur train, evaluation sur val), et choisissez la meilleure des deux techniques pour la suite.

## Analyse de modèle

### Comportement global

* Créer modèle sur train
* Regarder feature importances
* Regarder Partial Dependence Plots (via interface graphique BigML pour l'instant; plus tard: avec matplotlib ou plotly)

### Erreurs sur val set

* Téléchargement des probabilités de classe faites par le modèle, sur le dataset de validation
* Ajout de colonne d'erreur
* Calcul mesures agrégées de performance: implémenter chaque métrique en Python, puis vérifier valeurs calculées en comparant avec ce que donne BigML.
  * Métriques qui sont fonction du seuil: matrice confusion et accuracy
  * Métrique indépendante du seuil: AUC
* Export des lignes avec les 100 plus importantes erreurs
  * Inspection dans tableur. N'hésitez pas à rajouter une colonne de commentaires dans votre tableur.
  * Interprétations à noter dans le notebook (mentionner l'id de l'objet, et votre interprétation, par ex: valeurs de features anormales, etc.)
* Proposez des idées pour améliorer la préparation des données, en vous basant sur l'analyse des erreurs

## Optimisation du seuil de classification

Trouvez le seuil qui optimise le gain total sur le validation set, pour la matrice de gains/coûts suivante:

  | Actual / Predicted | 0 | 1 |
  |--------------------|---|---|
  | 0 | $500 | -$500 |
  | 1 | -$2500 | $0 |

## Envoi de prédictions à Kaggle

* Création d'un modèle à partir de train full
* Création des probabilités de classe sur le test set
* Envoi des prédictions à Kaggle [via Python](https://github.com/kaggle/kaggle-api)
* Essayez d'obtenir le meilleur résultat possible!

## Learning curves

* Comparez performance de “ensemble” et de “deepnet” sur val set, pour afficher des courbes (learning curves) comme à la page 11 de Machine Learning Yearning: en abscisse, pourcentage de données de train utilisées pour créer un modèle; en ordonnée, AUC du modèle créé, sur le val set.
* Afficher également la performance sur train set.

TODO lien vers ML Yearning

## Scikit-learn et XGBoost

* Parcourez le notebook [Intro-sklearn-xgboost.ipynb](Intro-sklearn-xgboost.ipynb) et adaptez votre code pour créer des modèles sans avoir à utiliser BigML

## Prédiction sur une nouvelle entrée

L’application utilisée par le banquier nous envoie les informations suivantes:

{
    "RevolvingUtilizationOfUnsecuredLines": 0.01703559,
    "NumberOfDependents": 1,
    "DebtRatio": 0,
    "age": 42,
    "NumberOfOpenCreditLinesAndLoans": 6,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime30-59DaysPastDueNotWorse": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfTimes90DaysLate": 0
}

Créez un nouveau notebook, dans lequel vous calculerez une probabilité de défaut de remboursement. Attention à ne pas dupliquer de code (DRY).
## Copyright

[Louis Dorard](https://www.louisdorard.com/) © 2019 - Tous droits réservés

Suivez-moi sur Twitter [@louisdorard](https://twitter.com/louisdorard)
