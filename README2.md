# 🎯 Projet Devoir : Fine-Tuning de Depth Anything avec LoRA

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.30+-FFD21E?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 👥 Auteurs

| Nom                  |
| -------------------- | 
| **Abdelali Chikhi** | 
| **Ayman Zejli**      | 
| **Mouad Azenag**      | 
| **Loic Magnan**      | 

---

## 📖 Contexte et Objectif

### Contexte

L'estimation de profondeur monoculaire est une tâche fondamentale en vision par ordinateur qui consiste à prédire la distance des objets dans une scène à partir d'une seule image RGB. Cette capacité est cruciale pour de nombreuses applications :

- 🚗 **Véhicules autonomes** : Navigation et évitement d'obstacles
- 🤖 **Robotique** : Manipulation d'objets et navigation
- 🏭 **Industrie 4.0** : Contrôle qualité et inspection automatisée
- 🎮 **Réalité augmentée** : Placement précis d'objets virtuels

### Objectif du Projet

Ce projet vise à **adapter le modèle Depth Anything** (un modèle pré-entraîné de pointe pour l'estimation de profondeur) au **jeu de données Zivid** spécifique à un contexte industriel, en utilisant la technique de **LoRA (Low-Rank Adaptation)** pour un fine-tuning efficace.

#### Pourquoi LoRA ?

- ✅ Réduction drastique des paramètres entraînables (~1.75% des paramètres totaux)
- ✅ Préservation des connaissances du modèle pré-entraîné
- ✅ Entraînement rapide avec moins de ressources GPU
- ✅ Fusion facile des adaptateurs avec le modèle de base

---

## 🧠 Architecture et Algorithmes

### 1. Le Modèle Pré-entraîné : Depth Anything V2

**Depth Anything V2** est un modèle de fondation de pointe pour l'estimation de profondeur monoculaire. Il s'appuie sur une architecture **DPT (Dense Prediction Transformer)** propulsée par un encodeur **Vision Transformer (ViT)**. Cette architecture permet de capturer des relations globales dans l'image grâce au mécanisme d'attention, surpassant les CNNs classiques sur la préservation des détails fins.

**Implémentation via Hugging Face :**
Pour ce projet, nous n'avons pas téléchargé manuellement les poids depuis le dépôt GitHub officiel. Nous avons privilégié l'intégration native via la bibliothèque **Transformers** de Hugging Face.

Le modèle est chargé dynamiquement depuis le **Hugging Face Hub** (ID : `depth-anything/Depth-Anything-V2-Small-hf`). Cette approche simplifie le pipeline (via `AutoModelForDepthEstimation`), assure la compatibilité des versions et évite la gestion complexe de fichiers de poids locaux.

#### Architecture du Modèle

```
┌─────────────────────────────────────────────────────────────┐
│                    Depth Anything Small                      │
├─────────────────────────────────────────────────────────────┤
│  Input Image (H × W × 3)                                    │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────────────────────────────────────┐        │
│  │         Vision Transformer (ViT) Backbone       │        │
│  │  - Patch Embedding (16 × 16 patches)            │        │
│  │  - Multi-Head Self-Attention (Query, Key, Value)│        │
│  │  - Feed-Forward Networks                        │        │
│  └─────────────────────────────────────────────────┘        │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────────────────────────────────────┐        │
│  │              DPT Decoder Head                   │        │
│  │  - Feature Reassembly                           │        │
│  │  - Progressive Upsampling                       │        │
│  └─────────────────────────────────────────────────┘        │
│         │                                                    │
│         ▼                                                    │
│  Output Depth Map (H × W × 1)                               │
└─────────────────────────────────────────────────────────────┘
```

#### Caractéristiques Clés

- **Modèle utilisé** : `LiheYoung/depth-anything-small-hf`
- **Paramètres totaux** : ~25.2 millions
- **Résolution d'entrée** : 518 × 840 pixels (adaptée aux images Zivid)
- **Sortie** : Carte de profondeur normalisée [0, 1]

### 2. L'Algorithme de Fine-Tuning : LoRA (Low-Rank Adaptation)

**LoRA** est une technique de fine-tuning efficace qui permet d'adapter de grands modèles pré-entraînés sans modifier leurs poids originaux.

#### Principe Mathématique

Au lieu de mettre à jour les poids $W$ directement, LoRA décompose la mise à jour en deux matrices de faible rang :

$$
W_{new} = W_{original} + \Delta W = W_{original} + B \cdot A
$$

Où :

- $W_{original} \in \mathbb{R}^{d \times k}$ : Poids gelés du modèle original
- $A \in \mathbb{R}^{r \times k}$ : Matrice "down-projection" (compresse)
- $B \in \mathbb{R}^{d \times r}$ : Matrice "up-projection" (décompresse)
- $r$ : Rang (hyperparamètre, $r \ll \min(d, k)$)

#### Configuration LoRA Utilisée

```python
lora_config = LoraConfig(
    r=16,                                    # Rang de la décomposition
    lora_alpha=32,                           # Facteur d'échelle (α/r)
    target_modules=["query", "key", "value"],# Couches d'attention ciblées
    lora_dropout=0.05,                       # Régularisation
    bias="none",                             # Pas d'adaptation des biais
)
```

#### Statistiques d'Entraînement

```
trainable params: 442,368 || all params: 25,227,457 || trainable%: 1.7535
```

---

## 💻 Détails de l'Implémentation et Pipeline (Code)

### A. Pipeline de Données Custom (Dataset Zivid)

Le jeu de données Zivid est un dataset industriel contenant des paires image RGB / profondeur XYZ capturées par une caméra Zivid.

#### Structure des Fichiers

```
DATASET_DEVOIR/
├── images/                    # Images RGB (.png)
│   ├── 21-12-03-18-52-37_Zivid_acquisition_color.png
│   ├── 22-09-21-14-12-11_Zivid_acquisition_color.png
│   └── ...
└── depth/                     # Cartes de profondeur XYZ (.npy)
    ├── 21-12-03-18-52-37_Zivid_acquisition_rawDepth.npy
    ├── 22-09-21-14-12-11_Zivid_acquisition_rawDepth.npy
    └── ...
```

#### Statistiques du Dataset

| Métrique                    | Valeur              |
| ---------------------------- | ------------------- |
| Nombre total d'échantillons | 58                  |
| Résolution des images       | 1200 × 1944 pixels |
| Profondeur MIN               | 251.74 mm           |
| Profondeur MAX               | 3907.45 mm          |
| Moyenne des profondeurs      | 1542.16 mm          |
| Écart-type                  | 295.35 mm           |
| Pixels valides (moyenne)     | 68.5%               |


### B. Configuration du Modèle et LoRA

```python
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from peft import get_peft_model, LoraConfig

# 1. Chargement du modèle pré-entraîné
MODEL_NAME = "LiheYoung/depth-anything-small-hf"
base_model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME)
image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

# 2. Configuration LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "key", "value"],
    lora_dropout=0.05,
    bias="none",
)

# 3. Application de LoRA
model_lora = get_peft_model(base_model, lora_config)
```
## ▶ Comment exécuter le projet

Placez votre dataset dans la structure suivante :
```text
dataset/
  images/
  depth/
```
Puis ouvrez et lancez le notebook :
```text
jupyter notebook Projet_lora.ipynb
```
Le notebook contient toutes les étapes du pipeline :
```text
-chargement et prétraitement des données

-création du modèle LoRA

-entraînement

-calcul des métriques

-visualisation (RGB, profondeur GT, profondeur prédite, carte d’erreur)
```
## 📊 Métriques

Nous utilisons les métriques classiques de l’estimation de profondeur :
```text
RMSE

AbsRel

δ₁, δ₂, δ₃
```

Les métriques sont calculées sur la profondeur normalisée (inverse depth) afin de garantir la stabilité de l’entraînement. Les prédictions peuvent ensuite être reconverties en profondeur réelle 
𝑍
Z (en millimètres ou mètres) pour la visualisation et l’analyse physique.

## 🖼 Visualisation

Le notebook génère automatiquement des figures contenant :
```text
l’image RGB

la profondeur de vérité terrain (GT)

la profondeur prédite

l’erreur absolue
```

## 🧪 Résultats 

Après 15 époques d’entraînement :
| Métrique | Valeur |
| -------- | ------ |
| RMSE     | ~0.42  |
| δ₁       | ~0.45  |
| δ₃       | ~0.69  |
Malgré un dataset très réduit, le modèle apprend la géométrie des pneus empilés et reconstruit leur forme de manière cohérente.

## 📄 Rapport

Un rapport technique complet  est disponible 

Il contient :
```text
la description du problème

le dataset

la méthode (Depth Anything + LoRA)

les fonctions de perte

les métriques

les résultats quantitatifs et qualitatifs
```
