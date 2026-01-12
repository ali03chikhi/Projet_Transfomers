# 🎯 Projet Devoir : Fine-Tuning de Depth Anything V2 avec LoRA (Transformers)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.30+-FFD21E?style=for-the-badge)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-6A5ACD?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 👥 Auteurs

| Nom |
| --- |
| **Abdelali Chikhi** |
| **Ayman Zejli** |
| **Mouad Azennag** |
| **Loic Magnan** |

---

## 📖 Contexte et Objectif

L’estimation de profondeur monoculaire (MDE) consiste à prédire une carte de profondeur dense à partir d’une seule image RGB.

**Objectif du projet :** adapter **Depth Anything V2** (Transformers) au dataset industriel **Zivid** (paires RGB + nuage de points XYZ par pixel) en utilisant **LoRA** (fine-tuning paramètre-efficiente).

🎯 Focus de la version finale : améliorer la précision sur les **objets proches** et les **détails fins** (ex. contours / rainures de pneus) grâce à :
- une **normalisation inverse** de la profondeur,
- une **loss mixte** : **L1 masquée + loss de gradient** (bords),
- une entrée **haute résolution** et un **upsampling bicubique** vers la GT.

---

## 🧠 Modèle & Méthode

### 1) Modèle pré-entraîné : Depth Anything V2 (HF)

- **Model ID (Hugging Face)** : `depth-anything/Depth-Anything-V2-Small-hf`
- Chargement via Transformers :
  - `AutoImageProcessor`
  - `AutoModelForDepthEstimation`

### 2) Fine-tuning LoRA (PEFT)

LoRA apprend une mise à jour de rang faible :

\[
W' = W + \Delta W,\quad \Delta W = BA
\]

Configuration LoRA (version finale) :
- `r = 16`
- `lora_alpha = 32`
- `target_modules = ["query","key","value"]`
- `lora_dropout = 0.05`
- `bias = "none"`

---

## 📂 Dataset Zivid & Structure attendue

Chaque échantillon :
- une image RGB (`.png`)
- un fichier profondeur (`.npy`) de shape `(H, W, 3)` contenant `(X, Y, Z)` par pixel
- la GT profondeur = **canal Z** en **mm**

Structure recommandée :
DATASET_DEVOIR/
├── images/ # Images RGB (.png)
└── depth/ # Nuages de points XYZ (.npy)

### Statistiques typiques (dataset fourni)
- Nb total : **58** échantillons
- Résolution brute : **1200 × 1944**
- Profondeur min/max (mm) : **251.74** / **3907.45**

---

## 🔧 Prétraitement (version finale)

### 1) Masque de validité (NaN / trous capteur)
On construit un masque de pixels valides :
- `Z` fini (pas NaN/inf)
- `0 < Z < 10000` (filtrage valeurs aberrantes)
Les pixels invalides sont remplacés par 0 pour stocker, mais **ignorés dans la loss**.

### 2) Normalisation inverse (améliorer les objets proches)
Au lieu de normaliser linéairement, on applique :

\[
z_{inv} = \frac{1}{z + \varepsilon}
\]

Avec :
\[
z_{min}^{inv} = \frac{1}{z_{max}},\quad z_{max}^{inv} = \frac{1}{z_{min}}
\]
\[
z_{norm} = \mathrm{clip}\left(\frac{z_{inv}-z_{min}^{inv}}{z_{max}^{inv}-z_{min}^{inv}}, 0, 1\right)
\]

✅ Effet : les petites distances (objets proches) occupent une plage plus large → meilleurs détails.

### 3) Haute résolution en entrée
Dans le `Dataset`, le processor impose :
- **height = 756**
- **width = 1260**
(choisi car multiple de 14, et bon compromis détails / mémoire)

---

## 🧾 Entraînement (version finale)

### 1) Alignement des dimensions
La sortie `predicted_depth` n’a pas forcément la taille de la GT.
On **upsample** la prédiction vers la taille GT (1200×1944) via :

- `F.interpolate(..., mode="bicubic", align_corners=False)`

### 2) Loss : L1 masquée + loss de gradient (bords)

#### a) L1 masquée
Calculée uniquement sur les pixels valides :
\[
\mathcal{L}_{L1} = \frac{\sum M| \hat{d}-d |}{\sum M + \varepsilon}
\]

#### b) Loss de gradient (netteté des contours)
On calcule des gradients par différences finies (x/y) et on applique :
- un masque de validité voisinage (`mask_x`, `mask_y`)
- une pondération plus forte sur les pixels “bords” :
  - seuil `tau = 0.02`
  - multiplicateur `+10` quand `|grad(GT)| > tau`

Loss totale (version finale) :
\[
\mathcal{L} = \mathcal{L}_{L1} + 3.0 \cdot \mathcal{L}_{grad}
\]

### 3) Hyperparamètres (TrainingArguments)
Configuration finale :
- `num_train_epochs = 15`
- `per_device_train_batch_size = 1` (obligatoire en haute résolution)
- `gradient_accumulation_steps = 8` (batch effectif ≈ 8)
- `learning_rate = 5e-5`
- `fp16 = True`
- `eval_strategy = "epoch"`
- `save_strategy = "epoch"`
- `load_best_model_at_end = True`
- `output_dir = "./resultats_pneu_v5"` (ou équivalent)

---

## 🚀 Reproduire le projet

### 1) Installation
Option conda :
```bash
conda create -n depth_lora python=3.10 -y
conda activate depth_lora
pip install -r requirements.txt
2) Préparer le dataset

Place DATASET_DEVOIR/images et DATASET_DEVOIR/depth comme décrit plus haut.

3) Lancer le notebook

Ouvre le notebook principal (ex. transfomers_code.ipynb) et exécute les cellules dans l’ordre :

imports / install

lecture dataset + stats globales min/max

création Dataset (use_inverse=True)

chargement modèle + LoRA

Trainer custom (loss L1 + gradient)

entraînement + visualisation qualitative
🧪 Inférence et dénormalisation (retour en mm)

Après prédiction, si ta sortie est une profondeur normalisée inverse depth_norm dans [0,1] :
import torch

DEPTH_MIN = 251.74
DEPTH_MAX = 3907.45

depth_min_inv = 1.0 / DEPTH_MAX
depth_max_inv = 1.0 / DEPTH_MIN

# depth_norm : (H,W) en [0,1]
depth_inv = depth_norm * (depth_max_inv - depth_min_inv) + depth_min_inv
depth_mm = 1.0 / (depth_inv + 1e-6)
⚠️ Si tu compares à la GT en mm : applique le masque (pixels valides uniquement).
🧩 Arborescence
Projet_Transformers/
├── transfomers_code.ipynb              # Notebook final
├── README.md
├── requirements.txt
├── DATASET_DEVOIR/
│   ├── images/
│   └── depth/
└── resultats_pneu_v5/
    ├── checkpoint-.../
    └── ...
🛠️ Dépannage rapide

CUDA OOM (mémoire GPU) :

garder batch_size=1

augmenter gradient_accumulation_steps

réduire la résolution (si nécessaire)

Profondeur “pas parfaite” sur pneus :

GT bruitée/incomplète

pneus sombres/reflets → ambiguïtés monoculaires

upsampling bicubique aide, mais les micro-détails restent difficiles
📚 Références

Depth Anything (arXiv): https://arxiv.org/abs/2401.10891

LoRA (arXiv): https://arxiv.org/abs/2106.09685

Transformers docs: https://huggingface.co/docs/transformers

PEFT docs: https://huggingface.co/docs/peft
