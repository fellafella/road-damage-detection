# 🛣️ Road Damage Detection – YOLOv8

Détection automatique des dégradations routières (fissures, nids-de-poule, etc.) à partir d'images, en utilisant **YOLOv8** (Ultralytics) entraîné sur le dataset **RDD2022** (Road Damage Dataset).

## 📋 Description

Ce projet entraîne un modèle de détection d'objets **YOLOv8** capable d'identifier et de localiser 5 types de dégradations routières sur des images :

| Classe | Description |
|--------|-------------|
| 0 | Longitudinal crack (fissure longitudinale) |
| 1 | Transverse crack (fissure transversale) |
| 2 | Alligator crack (fissure en peau de crocodile) |
| 3 | Other corruption (autre dégradation) |
| 4 | Pothole (nid-de-poule) |

## 📂 Dataset

Le projet utilise le dataset **RDD2022** (Road Damage Dataset), organisé au format YOLO :

```
RDD_SPLIT/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Chaque image possède un fichier `.txt` associé contenant les annotations au format YOLO (`classe x_centre y_centre largeur hauteur`, normalisées entre 0 et 1).

> ⚠️ Le dataset n'est pas inclus dans ce dépôt en raison de sa taille. Vous pouvez le télécharger via [Kaggle - RDD2022](https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022) ou toute autre source équivalente.

## ⚙️ Prérequis

- Python 3.8+
- GPU recommandé (CUDA) pour l'entraînement
- Jupyter Notebook / JupyterLab ou Kaggle Notebooks

## 📦 Installation

```bash
git clone https://github.com/<votre-utilisateur>/<nom-du-repo>.git
cd <nom-du-repo>
pip install -r requirements.txt
```

### `requirements.txt`

```
ultralytics
torch
matplotlib
pillow
```

## 🚀 Utilisation

### 1. Configurer le dataset

Adaptez les chemins dans le notebook (`base_path`, `data.yaml`) selon l'emplacement de votre dataset.

Le fichier `data.yaml` généré ressemble à ceci :

```yaml
path: /chemin/vers/RDD_SPLIT
train: train/images
val: val/images
test: test/images

nc: 5
names: ['longitudinal crack', 'transverse crack', 'alligator crack', 'other corruption', 'Pothole']
```

### 2. Lancer l'entraînement

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    project="runs",
    name="road_damage_full",
    patience=10
)
```

### 3. Évaluer et visualiser les résultats

Le notebook affiche automatiquement :
- 📊 Courbes de résultats (loss, mAP, precision, recall)
- 🔲 Matrice de confusion (brute et normalisée)
- 📈 Courbes Precision / Recall / F1
- 🏷️ Distribution des labels
- ✅ Comparaison labels réels vs prédictions sur les batchs de validation

### 4. Faire une prédiction

```python
from ultralytics import YOLO

model = YOLO("runs/road_damage_full/weights/best.pt")
results = model.predict(source="chemin/vers/image.jpg", conf=0.25, save=True)
```

## 📁 Structure du projet

```
.
├── projet-roat2.ipynb      # Notebook principal (exploration, entraînement, évaluation)
├── data.yaml                # Configuration du dataset (généré automatiquement)
├── requirements.txt
└── README.md
```

## 📊 Résultats

Après entraînement, les résultats (poids du modèle, graphiques, matrices de confusion) sont enregistrés dans `runs/road_damage_full/`.

## 🛠️ Technologies utilisées

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- PyTorch
- Matplotlib / PIL
- Jupyter Notebook

## ✍️ Auteur

Projet réalisé dans le cadre d'un projet de détection de dégradations routières par vision par ordinateur.
