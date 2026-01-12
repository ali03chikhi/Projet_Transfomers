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

### C. Boucle d'Entraînement (Trainer API)

#### Arguments d'Entraînement

```python
training_args = TrainingArguments(
    output_dir="./depth_anything_finetuned_lora_zivid",
    learning_rate=5e-5,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_steps=500,
    save_steps=500,
    logging_steps=50,
    report_to="tensorboard",
    save_total_limit=3,
    fp16=True,                    # Mixed precision pour accélération
    remove_unused_columns=False,
)
```

#### Trainer Personnalisé avec Loss Masquée

```python
class DepthTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Forward du modèle
        outputs = model(pixel_values=inputs.get("pixel_values"))
        predicted_depth = outputs.predicted_depth
      
        # Récupération des labels et masque
        labels = inputs.get("labels")
        valid_mask = inputs.get("valid_mask")
      
        # Interpolation pour aligner les dimensions
        labels = F.interpolate(labels.unsqueeze(1), size=predicted_depth.shape[-2:])
        valid_mask = F.interpolate(valid_mask.unsqueeze(1), size=predicted_depth.shape[-2:])
      
        # MSE masquée (on ignore les pixels NaN/invalides)
        diff = (predicted_depth - labels) ** 2
        masked_diff = diff * valid_mask
        loss = masked_diff.sum() / (valid_mask.sum() + 1e-8)
      
        return (loss, outputs) if return_outputs else loss
```

### D. Inférence et Calcul Métrique "Réel"

```python
# Chargement du modèle fine-tuné
from peft import PeftModel

model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME)
model = PeftModel.from_pretrained(model, "./depth_anything_finetuned_lora_zivid")

# Inférence
with torch.no_grad():
    outputs = model(pixel_values=image_tensor)
    predicted_depth = outputs.predicted_depth

# Dé-normalisation pour obtenir les valeurs en mm
depth_mm = predicted_depth * (DEPTH_MAX - DEPTH_MIN) + DEPTH_MIN
```

---

## 📊 Analyse des Performances

### Résultats du Dernier Run

#### Métriques d'Entraînement

| Métrique                          | Valeur                                      |
| ---------------------------------- | ------------------------------------------- |
| **Steps totaux**             | 120                                         |
| **Epochs**                   | 10                                          |
| **Temps d'exécution total** | **37 min 29 sec** (~2273.52 secondes) |
| **Samples par seconde**      | 0.202                                       |
| **Steps par seconde**        | 0.053                                       |
| **FLOPS totaux**             | 9.09 × 10¹⁶                              |

#### Évolution de la Loss

| Step | Training Loss    | Learning Rate |
| ---- | ---------------- | ------------- |
| 50   | **1.2008** | 3.0 × 10⁻⁵ |
| 100  | **0.0441** | 9.0 × 10⁻⁶ |

#### Loss Finale

- **Train Loss Finale** : `0.0441`
- **Train Loss Moyenne** : `0.5232`

### Interprétation des Résultats

1. **Convergence Rapide** : La loss chute drastiquement de 1.2 à 0.044 en seulement 100 steps, démontrant l'efficacité de LoRA pour l'adaptation de domaine.
2. **Efficacité du Fine-Tuning** : Avec seulement 1.75% des paramètres entraînés, le modèle atteint une loss très faible sur le dataset Zivid.
3. **Temps d'Entraînement Raisonnable** : ~38 minutes pour 10 epochs sur un dataset de 58 images haute résolution.

---

## 📂 Livrables

```
Projet_Transfomers/
├── 📓 projet_transfomers.ipynb        # Notebook principal avec tout le code
├── 📄 README.md                        # Ce fichier de documentation
├── 📋 requirements.txt                 # Dépendances Python
├── 📁 DATASET_DEVOIR/                  # Dataset Zivid (58 paires image/depth)
│   ├── images/                         # Images RGB (.png)
│   └── depth/                          # Cartes de profondeur XYZ (.npy)
├── 📁 depth_anything_finetuned_lora_zivid/  # Modèle fine-tuné
│   ├── checkpoint-120/                 # Dernier checkpoint
│   └── runs/                           # Logs TensorBoard
└── 📄 Sujet FineTuning Transformers.pdf # Énoncé du devoir
```

---

## 🚀 Guide de Reproduction

### 1. Installation de l'Environnement

#### Option A : Installation avec Conda 

```bash
# Créer un environnement conda
conda create -n transformers_depth python=3.11 -y
conda activate transformers_depth

# Installer les dépendances
pip install -r requirements.txt
```

#### Vérifier l'installation CUDA

```python
import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 3. Préparation des Données

Le dataset doit suivre la structure suivante :

```
DATASET_DEVOIR/
├── images/
│   └── *.png        # Images RGB
└── depth/
    └── *.npy        # Fichiers XYZ (profondeur en mm)
```

⚠️ **Important** : Les fichiers `.npy` doivent avoir la shape `(H, W, 3)` où le canal 2 (Z) contient les valeurs de profondeur en millimètres.

### 4. Lancement du Fine-Tuning

1. **Ouvrir le notebook** :

```bash
jupyter notebook projet_transfomers.ipynb
```

2. **Exécuter les cellules** dans l'ordre :

   - 📦 Installation des dépendances
   - 📊 Analyse du dataset Zivid
   - 🔧 Configuration du modèle et LoRA
   - 🚀 Entraînement
   - 📈 Visualisation des résultats
3. **Suivre l'entraînement avec TensorBoard** :

```bash
tensorboard --logdir ./depth_anything_finetuned_lora_zivid
```

### 5. Inférence avec le Modèle Fine-Tuné

```python
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from peft import PeftModel
from PIL import Image
import torch

# Charger le modèle
MODEL_NAME = "LiheYoung/depth-anything-small-hf"
model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME)
model = PeftModel.from_pretrained(model, "./depth_anything_finetuned_lora_zivid")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

# Charger une image
image = Image.open("votre_image.png")
inputs = processor(images=image, return_tensors="pt")

# Inférence
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    depth = outputs.predicted_depth

# Visualiser
import matplotlib.pyplot as plt
plt.imshow(depth.squeeze().numpy(), cmap='plasma')
plt.colorbar(label='Profondeur normalisée')
plt.show()
```

---

## 📚 Références

- [Depth Anything Paper](https://arxiv.org/abs/2401.10891)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Library](https://huggingface.co/docs/peft)

---

## 📝 License

Ce projet est réalisé dans le cadre d'un devoir académique. Tous droits réservés aux auteurs.

---

<div align="center">
  <sub>Projet Transformers - Polytech Clermont IMDS5A</sub>
</div>
