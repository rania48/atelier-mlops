Atelier MLOps : Git + DVC + MLflow

MLOps Pipeline

Projet de classification du dataset Iris construit comme un workflow MLOps versionné, traçable et reproductible :

Git versionne le code et la configuration (branches main / develop, tags v1.0 / v2.0) ;
DVC versionne les datasets et les modèles, et orchestre le pipeline ;
MLflow suit les expérimentations (paramètres, métriques, modèles) ;
GitHub Actions relance le pipeline à chaque push sur main.

Auteure : Rania Oulmidi

Structure du projet
atelier-mlops/
├── .dvc/config                  # remote DVC "storage"
├── .github/workflows/mlops.yml  # CI GitHub Actions
├── data/raw/
│   ├── iris.csv.dvc             # dataset Iris (suivi par DVC)
│   └── wine.csv.dvc             # dataset Wine (suivi par DVC)
├── src/
│   ├── prepare_data.py          # génère data/raw/iris.csv
│   ├── prepare_wine.py          # génère data/raw/wine.csv
│   ├── train.py                 # Random Forest + tracking MLflow
│   └── train_knn.py             # KNN + tracking MLflow
├── models/                      # model.pkl, knn.pkl (sorties DVC)
├── reports/
│   ├── metrics.json             # métriques Random Forest
│   └── metrics_knn.json         # métriques KNN
├── params.yaml                  # hyperparamètres
├── dvc.yaml                     # définition du pipeline
├── dvc.lock                     # état exact de la dernière exécution
└── requirements.txt
Installation

Prérequis : Python 3.10+ et Git.

bash
git clone https://github.com/rania48/atelier-mlops.git
cd atelier-mlops
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/macOS : source venv/bin/activate)
pip install -r requirements.txt
Récupérer les données

Le remote DVC de l'atelier est un dossier local (../dvc-storage). Sur une autre machine, indiquez son emplacement puis récupérez les données :

bash
dvc remote modify --local storage url <chemin/vers/dvc-storage>
dvc pull

Sans accès au remote, le dataset Iris peut être régénéré à l'identique :

bash
python src/prepare_data.py
Exécuter le pipeline
bash
dvc repro            # exécute les étapes train et train_knn si nécessaire
dvc dag              # affiche le graphe du pipeline
dvc metrics show     # affiche les métriques

Pipeline :

data/raw/iris.csv.dvc
        ├──> train      (Random Forest) -> models/model.pkl, reports/metrics.json
        └──> train_knn  (KNN)           -> models/knn.pkl,   reports/metrics_knn.json

Les hyperparamètres se modifient dans params.yaml :

yaml
train:
  n_estimators: 200
  max_depth: 8
  random_state: 42
  test_size: 0.2
knn:
  n_neighbors: 5
Suivre les expérimentations avec MLflow
bash
mlflow ui --backend-store-uri sqlite:///mlflow.db

Ouvrir http://localhost:5000, puis l'expérience TP_Iris_Classification. Sélectionner plusieurs runs et cliquer sur Compare pour les comparer.

Résultats

Jeu de test : 30 exemples (20 %, découpage stratifié, random_state = 42), identique pour tous les modèles.

Modèle	Paramètres	Accuracy	Precision	Recall	F1-score
KNN	n_neighbors = 5	1.000	1.000	1.000	1.000
Random Forest (v1.0)	n_estimators = 100, max_depth = 5	0.933	0.933	0.933	0.933
Random Forest (v2.0)	n_estimators = 200, max_depth = 8	0.900	0.902	0.900	0.900

Un nombre d'arbres plus élevé n'améliore pas forcément le modèle. Avec seulement 30 exemples de test, l'écart entre les modèles tient à une ou deux prédictions.

Restaurer une version
bash
git checkout v1.0
dvc checkout          # restaure le modèle correspondant (dvc pull si absent du cache)
dvc metrics show      # accuracy 0.93333
git checkout develop
dvc checkout
Intégration continue

Le workflow .github/workflows/mlops.yml installe les dépendances, régénère le dataset Iris, exécute dvc repro et affiche les métriques à chaque push ou pull request sur main.

Le remote DVC local n'étant pas accessible depuis GitHub, la CI régénère les données au lieu d'exécuter dvc pull. En production, on utiliserait un remote cloud (S3, Azure Blob, GCS) avec des identifiants stockés dans les secrets GitHub.

Technologies

Python 3.12 · pandas · scikit-learn · Git · DVC · MLflow · GitHub Actions
