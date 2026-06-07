# 📊 Data-Driven Decision Making — E-Commerce Sales Forecasting

> Projet académique complet couvrant l'intégralité du pipeline décisionnel basé sur la donnée.
> **Domaine :** E-Commerce | **Modèle retenu :** XGBoost (MAPE < 10%) | **Dataset :** 205,000 lignes

---

## 📋 Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture du projet](#architecture-du-projet)
3. [Description des fichiers](#description-des-fichiers)
4. [Installation](#installation)
5. [Lancement du Dashboard](#lancement-du-dashboard)
6. [Lancement du Notebook](#lancement-du-notebook)
7. [Sources de données](#sources-de-données)
8. [Résultats clés](#résultats-clés)

---

## Vue d'ensemble

Ce projet implémente un système d'aide à la décision complet pour un retailer e-commerce. Il couvre 6 phases :

| Phase | Description | Livrable |
|-------|-------------|----------|
| **Phase 1** | Définition du problème & KPIs | Business Case + KPI Tree |
| **Phase 2** | Collecte & Audit des données | Data Dictionary + Dataset fusionné |
| **Phase 3** | Exploration & Analyse statistique (EDA) | Visualisations + Tests statistiques |
| **Phase 4** | Modélisation prédictive & Interprétabilité | Modèles SARIMA / Prophet / XGBoost + SHAP |
| **Phase 5** | Visualisation | Dashboard interactif 5 vues |
| **Phase 6** | Décision, A/B Testing & Mesure d'impact | 3 Recommandations + Plan A/B Test |

**Question décisionnelle centrale :**
> *"Comment prévoir avec précision le chiffre d'affaires hebdomadaire par catégorie pour les 8 prochaines semaines afin d'optimiser la gestion des stocks et les investissements marketing ?"*

---

## Architecture du Projet

```
DDDM_Projet/
│
├── 📓 NOTEBOOKS
│   └── DDDM_Projet_Complet.ipynb          ← Notebook unifié (toutes phases)
│
├── dashboard.py                        ← Application Dash (localhost:8050) 
│
├──  Ecommerce_Sales_Data_2024_2025.csv  ← Source 1 : EC Sales (Inde)
├── product_sales_dataset_final.csv     ← Source 2 : Product Sales (USA)
├── merged_ecommerce_dataset.csv        ← Dataset fusionné (205,000 lignes) ← GÉNÉRÉ
│
│
├── 📁 PRÉDICTIONS (générées par Phase 4)
│   ├── predictions_ec.csv                  ← Prévisions XGBoost — EC Sales
│   ├── predictions_ps.csv                  ← Prévisions XGBoost — Product Sales
│   ├── ps_weekly_features.csv              ← Features hebdomadaires — PS
│   ├── ec_segmented.csv                    ← Segments clients — EC
│   └── ps_segmented.csv                    ← Segments clients — PS
│
├── 📁 VISUALISATIONS (générées par les notebooks)
│   ├── phase1_roi_business_case.png
│   ├── phase2_bias_analysis.png
│   ├── phase6_recommandations_impact.png
│   ├── phase6_ab_test_simulation.png
│   └── phase6_impact_financier.png
│
│
├── README.md                           ← Ce fichier
└── requirements.txt                    ← Dépendances Python

```

---

## Description des Fichiers

### Notebooks

| Fichier | Description | Cellules |
|---------|-------------|----------|
| `DDDM_Projet_Complet.ipynb` | Notebook principal unifié — contient les 6 phases complètes, entièrement documenté avec cellules Markdown | 248 |

### Dashboard

| Fichier | Description | Utilisation |
|---------|-------------|-------------|
| `dashboard.py` | Application Dash interactive avec 5 vues, filtres dynamiques, drill-down par catégorie et région, simulateur de stock | `python dashboard.py` → http://localhost:8050 |

### Données

| Fichier | Source | Lignes | Description |
|---------|--------|--------|-------------|
| `Ecommerce_Sales_Data_2024_2025.csv` | [Kaggle](https://www.kaggle.com/datasets/prince7489/e-commerce-sales) | 5,000 | Transactions e-commerce marché indien — avec Discount et Payment Mode |
| `product_sales_dataset_final.csv` | [Kaggle](https://www.kaggle.com/datasets/yashyennewar/product-sales-dataset-2023-2024) | 200,000 | Transactions marché US — avec State et Country |
| `merged_ecommerce_dataset.csv` | Généré par Phase 2 | 205,000 | Dataset fusionné et harmonisé — **fichier principal utilisé par toutes les phases** |

### Fichiers de Prédictions (générés par Phase 4)

| Fichier | Description |
|---------|-------------|
| `predictions_ec.csv` | Prévisions XGBoost sur EC Sales — colonnes : `week_start`, `revenue`, `y_pred` |
| `predictions_ps.csv` | Prévisions XGBoost sur Product Sales — colonnes : `week_start`, `revenue`, `y_pred` |
| `ps_weekly_features.csv` | Features hebdomadaires enrichies pour PS (lags, moving averages, variables saisonnières) |
| `ec_segmented.csv` | Résultats de clustering K-Means sur les clients EC |
| `ps_segmented.csv` | Résultats de clustering K-Means sur les clients PS |

---

## Installation

### Prérequis

- Python 3.9 ou supérieur
- pip

### Étape 1 — Cloner le dépôt

```bash
git clone https://github.com/soumiaaaen/DDDM.git
cd DDDM
```

### Étape 2 — Installer les dépendances

```bash
pip install -r requirements.txt
```

### Étape 3 — Vérifier les fichiers de données

Assurez-vous que ces fichiers sont présents dans le dossier du projet :

```
✅ Ecommerce_Sales_Data_2024_2025.csv
✅ product_sales_dataset_final.csv
```

Le fichier `merged_ecommerce_dataset.csv` sera généré automatiquement à l'exécution de la Phase 2 du notebook.

---

## Lancement du Dashboard

### Option A — Dashboard Dash (interactif complet)

```bash
python dashboard.py
```

Puis ouvrir dans le navigateur : **http://localhost:8050**

Le dashboard démarre en quelques secondes et charge automatiquement les données. Les 5 vues sont accessibles via les onglets :

| Vue | Profil | Contenu |
|-----|--------|---------|
| Vue 1 — KPIs Direction | Direction | KPIs financiers, croissance MoM, mix catégories |
| Vue 2 — Prévisions 8 sem. | Direction | Prévisions CA + intervalle de confiance, performance modèle |
| Vue 3 — Catégories | Marketing | CA par catégorie, marges, saisonnalité, impact discount |
| Vue 4 — Géographie | Marketing | Carte US, top états, drill-down région |
| Vue 5 — Stocks | Opérations | Couverture stock, alertes réapprovisionnement, simulateur |


---

## Lancement du Notebook

### Option A — Google Colab (recommandé, sans installation)

1. Aller sur [colab.research.google.com](https://colab.research.google.com)
2. **File → Upload notebook** → sélectionner `DDDM_Projet_Complet.ipynb`
3. Uploader les fichiers de données via le panneau gauche (icône dossier)
4. **Runtime → Run all**

### Option B — Jupyter Lab / Notebook local

```bash
pip install jupyterlab
jupyter lab
```

Puis ouvrir `DDDM_Projet_Complet.ipynb` depuis l'interface.

### Option C — VS Code

Installer l'extension **Jupyter** depuis le marketplace VS Code, puis ouvrir directement le fichier `.ipynb`.

> **Ordre d'exécution recommandé :** Exécuter les cellules dans l'ordre (Phase 1 → Phase 6). La Phase 2 génère `merged_ecommerce_dataset.csv` requis par toutes les phases suivantes.

---

## Sources de Données

### Source 1 — E-Commerce Sales Data 2024-2025

- **URL :** https://www.kaggle.com/datasets/prince7489/e-commerce-sales
- **Marché :** Inde (villes : Bangalore, Mumbai, Delhi, Chennai...)
- **Période :** Octobre 2023 – Octobre 2025
- **Volume :** 5,000 transactions
- **Colonnes clés :** `Order Date`, `Category`, `Region`, `Sales`, `Profit`, `Discount`, `Payment Mode`
- **Particularité :** Seul dataset avec `Discount` et `Payment Mode` (UPI, Net Banking, COD)

### Source 2 — US Product Sales Dataset 2023-2024

- **URL :** https://www.kaggle.com/datasets/yashyennewar/product-sales-dataset-2023-2024
- **Marché :** États-Unis (47 états)
- **Période :** Janvier 2023 – Décembre 2024
- **Volume :** 200,000 transactions
- **Colonnes clés :** `Order_Date`, `Category`, `Region`, `State`, `Revenue`, `Profit`, `Unit_Price`
- **Particularité :** Seul dataset avec `State` et `Country` — permet l'analyse géographique US

### Dataset Fusionné

| Propriété | Valeur |
|-----------|--------|
| Lignes totales | 205,000 |
| Période couverte | Janvier 2023 – Octobre 2025 |
| Colonnes | 17 (harmonisées) |
| Colonne `source` | `EC_2024_2025` ou `PS_2023_2024` |

---

## Résultats Clés

### Performance des Modèles

| Modèle | MAPE (PS) | MAPE (EC) | Statut |
|--------|-----------|-----------|--------|
| SARIMA | 86.75% | — | ❌ |
| Prophet | 95.10% | — | ❌ |
| XGBoost Baseline | 18.55% | 18.55% | ⚠️ |
| **XGBoost Optimisé** | **< 10%** | **18.55%*** | **✅** |

*EC Sales : 5,000 lignes seulement — dataset trop petit pour atteindre MAPE < 10%

### Top Features SHAP (XGBoost)

1. `lag_1` — CA de la semaine précédente
2. `is_black_friday` — Indicateur novembre
3. `week_of_year` — Numéro de semaine
4. `category_encoded` — Catégorie produit
5. `ma_4` — Moyenne mobile 4 semaines

### 3 Recommandations Actionnables

| Priorité | Recommandation | Impact Estimé |
|----------|---------------|---------------|
| 🥇 Critique | Anticiper Black Friday — stocks +200% dès sem. 40 | +£320,000 CA |
| 🥈 Élevée | Rééquilibrer budget marketing Clothing → Accessories | +£142,000 Profit |
| 🥉 Modérée | Campagne promotionnelle région South (A/B Test) | +£68,000 CA |

**Bénéfice net total : +£502,000 | ROI : 1,793% | Délai de retour : 0.6 mois**

### Plan A/B Test (Recommandation 3)

| Paramètre | Valeur |
|-----------|--------|
| Hypothèse H₁ | Campagne South +10% CA |
| Niveau α | 0.05 |
| Puissance | 80% |
| Taille/groupe | 1,402 commandes |
| Durée | 4 semaines |
| Résultat simulation | p=0.0042 ✅ H₀ rejetée |

---

## requirements.txt

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
statsmodels
prophet
shap
plotly
dash
scipy
openpyxl
ipywidgets
jupyterlab
```

---


*Projet réalisé dans le cadre du module Data-Driven Decision Making — Juin 2026*
