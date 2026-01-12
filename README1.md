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
