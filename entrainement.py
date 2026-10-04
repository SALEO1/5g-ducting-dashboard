import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib, shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, precision_recall_curve,
                              average_precision_score)
from sklearn.tree import export_graphviz, plot_tree
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
sns.set_theme(style="darkgrid", palette="deep")

# ──────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────
# ⚠  Modifiez ce chemin si votre fichier est ailleurs
CHEMIN_EXCEL = r"C:\Users\%USERNAME%\Desktop\meteo_maroc_toutes_zones.xlsx"
CHEMIN_EXCEL = os.path.expandvars(CHEMIN_EXCEL)

# Fallback : même dossier que ce script
if not os.path.exists(CHEMIN_EXCEL):
    CHEMIN_EXCEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "meteo_maroc_toutes_zones.xlsx")

os.makedirs("models",  exist_ok=True)
os.makedirs("figures", exist_ok=True)

PALETTE = {
    "ducting":   "#E74C3C",
    "normal":    "#2980B9",
    "accent":    "#F39C12",
    "dark":      "#1A1A2E",
    "mid":       "#16213E",
    "highlight": "#0F3460",
}

print("=" * 65)
print("  PROJET DUCTING — PARTIE 1 : ENTRAÎNEMENT")
print("=" * 65)


# ══════════════════════════════════════════════════════════════════
# PARTIE 1A — CHARGEMENT ET NETTOYAGE DES DONNÉES
# ══════════════════════════════════════════════════════════════════
print("\n▶  1A — Chargement des données Excel...")

if not os.path.exists(CHEMIN_EXCEL):
    raise FileNotFoundError(
        f"\n  Fichier introuvable : {CHEMIN_EXCEL}\n"
        "  Vérifiez la variable CHEMIN_EXCEL en haut du script."
    )

df = pd.read_excel(CHEMIN_EXCEL, sheet_name="Toutes les zones", engine="openpyxl")
df = df.rename(columns={"sst (°C)": "sst"}) if "sst (°C)" in df.columns else df
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values(['zone', 'time']).reset_index(drop=True)

print(f"   Dimensions       : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
print(f"   Zones            : {sorted(df['zone'].unique().tolist())}")
print(f"   Période          : {df['time'].min()} → {df['time'].max()}")
print(f"   Ducting (=1)     : {(df['duction']==1).sum():,} obs.")
print(f"   Normal  (=0)     : {(df['duction']==0).sum():,} obs.")
print(f"   Valeurs manquantes: {df.isnull().sum().sum():,}")

# Imputation des valeurs manquantes par zone
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df.groupby('zone')[col].transform(
        lambda x: x.fillna(x.median())
    )

print(f"   Après imputation : {df.isnull().sum().sum()} valeurs manquantes")


# ══════════════════════════════════════════════════════════════════
# PARTIE 1B — INGÉNIERIE DES VARIABLES PHYSIQUES
# ══════════════════════════════════════════════════════════════════
print("\n▶  1B — Calcul des variables physiques du ducting...")

# ── Inversions thermiques (mécanisme central du ducting)
df['inversion_basse']    = df['temperature_925hPa (°C)'] - df['temperature_1000hPa (°C)']
df['inversion_moyenne']  = df['temperature_900hPa (°C)'] - df['temperature_925hPa (°C)']
df['inversion_totale']   = df['temperature_900hPa (°C)'] - df['temperature_1000hPa (°C)']

# ── Gradient thermique vertical (lapse rate)
# Normal : -6.5°C/km. Si positif → inversion
df['lapse_rate_bas']     = (df['temperature_925hPa (°C)'] - df['temperature_1000hPa (°C)']) / 0.75
df['lapse_rate_haut']    = (df['temperature_900hPa (°C)'] - df['temperature_925hPa (°C)']) / 0.25

# ── Différence air/mer (inversion marine côtière)
df['diff_air_sst']       = df['temperature_2m (°C)'] - df['sst']
df['diff_app_sst']       = df['apparent_temperature (°C)'] - df['sst']

# ── Gradients d'humidité (réfractivité atmosphérique)
df['grad_hum_bas']       = df['relative_humidity_2m (%)'] - df['relative_humidity_1000hPa (%)']
df['grad_hum_moyen']     = df['relative_humidity_1000hPa (%)'] - df['relative_humidity_950hPa (%)']
df['hum_total_gradient'] = df['relative_humidity_2m (%)'] - df['relative_humidity_900hPa (%)']

# ── Réfractivité atmosphérique N (formule de Bean & Dutton)
# N = 77.6*P/T + 3.73e5*e/T² où e = pression de vapeur
T_K = df['temperature_2m (°C)'] + 273.15
e_s = 6.112 * np.exp(17.67 * df['temperature_2m (°C)'] / (df['temperature_2m (°C)'] + 243.5))
e   = (df['relative_humidity_2m (%)'] / 100.0) * e_s
df['refractivite_N']     = 77.6 * df['pressure_msl (hPa)'] / T_K + 373e3 * e / T_K**2

# ── Score physique de ducting (index composite)
# Normalisation 0-1 de chaque composante
def norm(s):
    r = s.max() - s.min()
    return (s - s.min()) / r if r > 0 else s * 0

df['score_inversion']    = norm(df['inversion_basse'].clip(lower=0))
df['score_humidite']     = norm(df['relative_humidity_2m (%)'])
df['score_vent_calme']   = norm(1 / (df['wind_speed_10m (km/h)'].clip(lower=0.5)))
df['score_pression']     = norm(df['pressure_msl (hPa)'])
df['score_sst_diff']     = norm(df['diff_air_sst'].clip(lower=0))

# Score composite pondéré (poids physiquement justifiés)
df['ducting_score'] = (
    0.35 * df['score_inversion']  +
    0.20 * df['score_humidite']   +
    0.20 * df['score_vent_calme'] +
    0.15 * df['score_pression']   +
    0.10 * df['score_sst_diff']
)

# ── Variables temporelles
df['heure']   = df['time'].dt.hour
df['mois']    = df['time'].dt.month
df['saison']  = df['mois'].map({12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3})
df['jour_an'] = df['time'].dt.dayofyear

# ── Encodage de la zone
le = LabelEncoder()
df['zone_code'] = le.fit_transform(df['zone'])

# ── Indicateurs binaires physiques
df['is_inversion']  = (df['inversion_basse'] > 1.0).astype(int)
df['is_vent_calme'] = (df['wind_speed_10m (km/h)'] < 15).astype(int)
df['is_anticyclone']= (df['pressure_msl (hPa)'] > 1020).astype(int)
df['is_mer_froide'] = (df['diff_air_sst'] > 3).astype(int)

print(f"   Variables créées : {len([c for c in df.columns if c not in pd.read_excel(CHEMIN_EXCEL, sheet_name='Toutes les zones', nrows=0).columns])}")
print(f"   Score ducting moyen (duction=1) : {df[df['duction']==1]['ducting_score'].mean():.3f}")
print(f"   Score ducting moyen (duction=0) : {df[df['duction']==0]['ducting_score'].mean():.3f}")


# ══════════════════════════════════════════════════════════════════
# PARTIE 1C — VISUALISATION 1 : ANALYSE EXPLORATOIRE
# ══════════════════════════════════════════════════════════════════
print("\n▶  1C — Génération des visualisations exploratoires...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor(PALETTE['dark'])
fig.suptitle("ANALYSE EXPLORATOIRE — DUCTING vs NORMAL",
             fontsize=16, color='white', fontweight='bold', y=0.98)

for ax in axes.flat:
    ax.set_facecolor(PALETTE['mid'])
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#555')
    ax.spines['left'].set_color('#555')
    ax.spines['top'].set_color('#555')
    ax.spines['right'].set_color('#555')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

kw = dict(bins=40, alpha=0.75, edgecolor='none')

# 1. Inversion thermique
d1 = df[df['duction']==1]['inversion_basse']
d0 = df[df['duction']==0]['inversion_basse']
axes[0,0].hist(d0, **kw, color=PALETTE['normal'],  label='Normal')
axes[0,0].hist(d1, **kw, color=PALETTE['ducting'], label='Ducting')
axes[0,0].axvline(d1.mean(), color='yellow', lw=2, ls='--', label=f'Moy ducting={d1.mean():.1f}')
axes[0,0].set_title("Inversion Thermique (925–1000 hPa)")
axes[0,0].set_xlabel("°C"); axes[0,0].legend(fontsize=8, facecolor='#333', labelcolor='white')

# 2. Pression MSL
d1 = df[df['duction']==1]['pressure_msl (hPa)']
d0 = df[df['duction']==0]['pressure_msl (hPa)']
axes[0,1].hist(d0, **kw, color=PALETTE['normal'],  label='Normal')
axes[0,1].hist(d1, **kw, color=PALETTE['ducting'], label='Ducting')
axes[0,1].axvline(d1.mean(), color='yellow', lw=2, ls='--')
axes[0,1].set_title("Pression MSL (hPa)")
axes[0,1].set_xlabel("hPa"); axes[0,1].legend(fontsize=8, facecolor='#333', labelcolor='white')

# 3. Score ducting composite
d1 = df[df['duction']==1]['ducting_score']
d0 = df[df['duction']==0]['ducting_score']
axes[0,2].hist(d0, **kw, color=PALETTE['normal'],  label='Normal')
axes[0,2].hist(d1, **kw, color=PALETTE['ducting'], label='Ducting')
axes[0,2].set_title("Score Physique Ducting (composite)")
axes[0,2].set_xlabel("Score 0-1"); axes[0,2].legend(fontsize=8, facecolor='#333', labelcolor='white')

# 4. Humidité 2m
d1 = df[df['duction']==1]['relative_humidity_2m (%)']
d0 = df[df['duction']==0]['relative_humidity_2m (%)']
axes[1,0].hist(d0, **kw, color=PALETTE['normal'],  label='Normal')
axes[1,0].hist(d1, **kw, color=PALETTE['ducting'], label='Ducting')
axes[1,0].set_title("Humidité Relative 2m (%)")
axes[1,0].set_xlabel("%"); axes[1,0].legend(fontsize=8, facecolor='#333', labelcolor='white')

# 5. Vent 10m
d1 = df[df['duction']==1]['wind_speed_10m (km/h)']
d0 = df[df['duction']==0]['wind_speed_10m (km/h)']
axes[1,1].hist(d0, **kw, color=PALETTE['normal'],  label='Normal')
axes[1,1].hist(d1, **kw, color=PALETTE['ducting'], label='Ducting')
axes[1,1].set_title("Vitesse Vent 10m (km/h)")
axes[1,1].set_xlabel("km/h"); axes[1,1].legend(fontsize=8, facecolor='#333', labelcolor='white')

# 6. Différence Air-SST
d1 = df[df['duction']==1]['diff_air_sst']
d0 = df[df['duction']==0]['diff_air_sst']
axes[1,2].hist(d0, **kw, color=PALETTE['normal'],  label='Normal')
axes[1,2].hist(d1, **kw, color=PALETTE['ducting'], label='Ducting')
axes[1,2].set_title("Différence Air–SST (°C)")
axes[1,2].set_xlabel("°C"); axes[1,2].legend(fontsize=8, facecolor='#333', labelcolor='white')

plt.tight_layout()
plt.savefig("figures/fig1_analyse_exploratoire.png", dpi=150, bbox_inches='tight',
            facecolor=PALETTE['dark'])
plt.close()
print("   ✓ figures/fig1_analyse_exploratoire.png")


# ══════════════════════════════════════════════════════════════════
# PARTIE 1D — VISUALISATION 2 : MATRICE DE CORRÉLATION
# ══════════════════════════════════════════════════════════════════
print("\n▶  1D — Matrice de corrélation...")

corr_cols = [
    'inversion_basse', 'inversion_totale', 'lapse_rate_bas',
    'diff_air_sst', 'grad_hum_bas', 'refractivite_N',
    'pressure_msl (hPa)', 'wind_speed_10m (km/h)',
    'relative_humidity_2m (%)', 'cloud_cover (%)',
    'visibility (m)', 'temperature_2m (°C)', 'sst',
    'ducting_score', 'duction'
]
corr_labels = [
    'Inv. basse', 'Inv. totale', 'Lapse rate bas',
    'Air-SST diff', 'Grad. hum.', 'Réfractivité N',
    'Pression MSL', 'Vent 10m',
    'Humidité 2m', 'Couv. nuageuse',
    'Visibilité', 'Temp. 2m', 'SST',
    'Score ducting', 'DUCTION'
]

corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(14, 12))
fig.patch.set_facecolor(PALETTE['dark'])
ax.set_facecolor(PALETTE['mid'])

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-1, vmax=1,
            center=0, annot=True, fmt='.2f', linewidths=0.5,
            linecolor='#333', ax=ax, annot_kws={'size': 8},
            xticklabels=corr_labels, yticklabels=corr_labels,
            cbar_kws={'shrink': 0.8})

ax.set_title("MATRICE DE CORRÉLATION — Variables Ducting",
             fontsize=14, color='white', fontweight='bold', pad=15)
ax.tick_params(colors='white', labelsize=9)
plt.xticks(rotation=45, ha='right', color='white')
plt.yticks(rotation=0, color='white')

# Encadrer la colonne DUCTION
ax.add_patch(plt.Rectangle((len(corr_cols)-1, 0), 1, len(corr_cols)-1,
                             fill=False, edgecolor='yellow', lw=3))

plt.tight_layout()
plt.savefig("figures/fig2_correlation.png", dpi=150, bbox_inches='tight',
            facecolor=PALETTE['dark'])
plt.close()
print("   ✓ figures/fig2_correlation.png")

# Top corrélations avec 'duction'
top_corr = corr_matrix['duction'].drop('duction').abs().sort_values(ascending=False)
print("\n   Top 10 corrélations avec DUCTION :")
for feat, val in top_corr.head(10).items():
    bar = "█" * int(val * 30)
    print(f"   {feat:<35} {val:.3f}  {bar}")


# ══════════════════════════════════════════════════════════════════
# PARTIE 1E — SÉLECTION DES VARIABLES ET PRÉPARATION ML
# ══════════════════════════════════════════════════════════════════
print("\n▶  1E — Préparation des données pour ML...")

FEATURES = [
    # Météo brute
    'temperature_2m (°C)', 'temperature_80m (°C)',
    'temperature_1000hPa (°C)', 'temperature_975hPa (°C)',
    'temperature_950hPa (°C)', 'temperature_925hPa (°C)', 'temperature_900hPa (°C)',
    'relative_humidity_2m (%)', 'relative_humidity_1000hPa (%)',
    'relative_humidity_975hPa (%)', 'relative_humidity_950hPa (%)',
    'pressure_msl (hPa)', 'surface_pressure (hPa)',
    'wind_speed_10m (km/h)', 'wind_speed_80m (km/h)', 'wind_speed_1000hPa (km/h)',
    'cloud_cover (%)', 'cloud_cover_low (%)', 'visibility (m)',
    'vapour_pressure_deficit (kPa)', 'sst',
    # Variables dérivées
    'inversion_basse', 'inversion_moyenne', 'inversion_totale',
    'lapse_rate_bas', 'lapse_rate_haut',
    'diff_air_sst', 'grad_hum_bas', 'grad_hum_moyen', 'hum_total_gradient',
    'refractivite_N', 'ducting_score',
    'is_inversion', 'is_vent_calme', 'is_anticyclone', 'is_mer_froide',
    # Temporelles
    'heure', 'mois', 'saison', 'zone_code',
]

df_ml = df[FEATURES + ['duction']].dropna()
X = df_ml[FEATURES].values
y = df_ml['duction'].values

print(f"   Dataset ML : {X.shape[0]:,} obs. × {X.shape[1]} features")
print(f"   Classe 1 (ducting) : {y.sum()} ({y.mean()*100:.1f}%)")
print(f"   Classe 0 (normal)  : {(y==0).sum()} ({(1-y.mean())*100:.1f}%)")

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE pour rééquilibrer les classes (20/80 → équilibré)
print("\n   Application SMOTE (rééquilibrage)...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_smote, y_smote = smote.fit_resample(X_scaled, y)
print(f"   Après SMOTE : {X_smote.shape[0]:,} obs. (classe 1 : {y_smote.sum()}, classe 0 : {(y_smote==0).sum()})")


# ══════════════════════════════════════════════════════════════════
# PARTIE 1F — ENTRAÎNEMENT RANDOM FOREST
# ══════════════════════════════════════════════════════════════════
print("\n▶  1F — Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=4,
    min_samples_split=10,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Validation temporelle (TimeSeriesSplit respecte l'ordre chronologique)
tscv = TimeSeriesSplit(n_splits=5)
rf_auc  = cross_val_score(rf, X_scaled, y, cv=tscv, scoring='roc_auc')
rf_f1   = cross_val_score(rf, X_scaled, y, cv=tscv, scoring='f1')
rf_prec = cross_val_score(rf, X_scaled, y, cv=tscv, scoring='precision')
rf_rec  = cross_val_score(rf, X_scaled, y, cv=tscv, scoring='recall')

print(f"   TimeSeriesSplit (5 folds) :")
print(f"   AUC-ROC   : {rf_auc.mean():.4f} ± {rf_auc.std():.4f}")
print(f"   F1        : {rf_f1.mean():.4f} ± {rf_f1.std():.4f}")
print(f"   Précision : {rf_prec.mean():.4f} ± {rf_prec.std():.4f}")
print(f"   Rappel    : {rf_rec.mean():.4f} ± {rf_rec.std():.4f}")

# Entraînement final
rf.fit(X_smote, y_smote)
joblib.dump(rf, "models/rf_model.pkl")
print("   ✓ models/rf_model.pkl sauvegardé")


# ══════════════════════════════════════════════════════════════════
# PARTIE 1G — ENTRAÎNEMENT XGBOOST
# ══════════════════════════════════════════════════════════════════
print("\n▶  1G — XGBoost...")

ratio = float((y == 0).sum()) / float((y == 1).sum())
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=ratio,   # gestion du déséquilibre
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

xgb_auc = cross_val_score(xgb, X_scaled, y, cv=tscv, scoring='roc_auc')
xgb_f1  = cross_val_score(xgb, X_scaled, y, cv=tscv, scoring='f1')

print(f"   TimeSeriesSplit (5 folds) :")
print(f"   AUC-ROC : {xgb_auc.mean():.4f} ± {xgb_auc.std():.4f}")
print(f"   F1      : {xgb_f1.mean():.4f} ± {xgb_f1.std():.4f}")

xgb.fit(X_scaled, y)
joblib.dump(xgb, "models/xgb_model.pkl")
print("   ✓ models/xgb_model.pkl sauvegardé")





# ══════════════════════════════════════════════════════════════════
# PARTIE 1I — VISUALISATION 3 : IMPORTANCE DES VARIABLES
# ══════════════════════════════════════════════════════════════════
print("\n▶  1I — Importance des variables...")

rf_imp  = pd.Series(rf.feature_importances_,  index=FEATURES).sort_values(ascending=False)
xgb_imp = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values(ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
fig.patch.set_facecolor(PALETTE['dark'])

for ax, imp, title, color in [
    (ax1, rf_imp.head(20),  "Random Forest — Top 20", PALETTE['normal']),
    (ax2, xgb_imp.head(20), "XGBoost — Top 20",       PALETTE['ducting']),
]:
    ax.set_facecolor(PALETTE['mid'])
    bars = ax.barh(range(20), imp.values[::-1], color=color, alpha=0.85, edgecolor='none')

    # Dégradé de couleur selon l'importance
    max_imp = imp.values[0]
    for i, (bar, val) in enumerate(zip(bars, imp.values[::-1])):
        alpha = 0.4 + 0.6 * (val / max_imp)
        bar.set_alpha(alpha)

    ax.set_yticks(range(20))
    ax.set_yticklabels(imp.index[::-1], fontsize=9, color='white')
    ax.set_title(title, fontsize=13, color='white', fontweight='bold')
    ax.set_xlabel("Importance", color='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#555')
    ax.spines['left'].set_color('#555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Valeurs sur les barres
    for i, (bar, val) in enumerate(zip(bars, imp.values[::-1])):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', ha='left', color='white', fontsize=8)

fig.suptitle("IMPORTANCE DES VARIABLES — RandomForest vs XGBoost",
             fontsize=14, color='white', fontweight='bold')
plt.tight_layout()
plt.savefig("figures/fig3_importance_variables.png", dpi=150, bbox_inches='tight',
            facecolor=PALETTE['dark'])
plt.close()
print("   ✓ figures/fig3_importance_variables.png")

print("\n   Top 10 variables (Random Forest) :")
for feat, imp_val in rf_imp.head(10).items():
    bar = "█" * int(imp_val * 400)
    print(f"   {feat:<35} {imp_val:.4f}  {bar}")


# ══════════════════════════════════════════════════════════════════
# PARTIE 1J — VISUALISATION 4 : COURBES ROC ET PRECISION-RAPPEL
# ══════════════════════════════════════════════════════════════════
print("\n▶  1J — Courbes ROC et Précision-Rappel...")

y_proba_rf  = rf.predict_proba(X_scaled)[:, 1]
y_proba_xgb = xgb.predict_proba(X_scaled)[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(PALETTE['dark'])

for ax in axes:
    ax.set_facecolor(PALETTE['mid'])
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#555')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

# ROC
fpr_rf, tpr_rf, _ = roc_curve(y, y_proba_rf)
fpr_xg, tpr_xg, _ = roc_curve(y, y_proba_xgb)
auc_rf  = roc_auc_score(y, y_proba_rf)
auc_xg  = roc_auc_score(y, y_proba_xgb)

axes[0].plot(fpr_rf, tpr_rf, color=PALETTE['normal'],  lw=2.5, label=f'Random Forest (AUC={auc_rf:.3f})')
axes[0].plot(fpr_xg, tpr_xg, color=PALETTE['ducting'], lw=2.5, label=f'XGBoost (AUC={auc_xg:.3f})')
axes[0].plot([0,1],[0,1], 'w--', lw=1, alpha=0.4, label='Aléatoire (AUC=0.5)')
axes[0].fill_between(fpr_rf, tpr_rf, alpha=0.1, color=PALETTE['normal'])
axes[0].fill_between(fpr_xg, tpr_xg, alpha=0.1, color=PALETTE['ducting'])
axes[0].set_xlabel("Taux Faux Positifs"); axes[0].set_ylabel("Taux Vrais Positifs")
axes[0].set_title("Courbe ROC"); axes[0].legend(facecolor='#333', labelcolor='white')

# Précision-Rappel
prec_rf, rec_rf, _ = precision_recall_curve(y, y_proba_rf)
prec_xg, rec_xg, _ = precision_recall_curve(y, y_proba_xgb)
ap_rf  = average_precision_score(y, y_proba_rf)
ap_xg  = average_precision_score(y, y_proba_xgb)

axes[1].plot(rec_rf, prec_rf, color=PALETTE['normal'],  lw=2.5, label=f'Random Forest (AP={ap_rf:.3f})')
axes[1].plot(rec_xg, prec_xg, color=PALETTE['ducting'], lw=2.5, label=f'XGBoost (AP={ap_xg:.3f})')
axes[1].axhline(y.mean(), color='white', lw=1.5, ls='--', alpha=0.5, label=f'Baseline ({y.mean():.2f})')
axes[1].set_xlabel("Rappel"); axes[1].set_ylabel("Précision")
axes[1].set_title("Courbe Précision-Rappel"); axes[1].legend(facecolor='#333', labelcolor='white')

fig.suptitle("PERFORMANCES DES MODÈLES — Courbes ROC et PR",
             fontsize=14, color='white', fontweight='bold')
plt.tight_layout()
plt.savefig("figures/fig4_roc_pr.png", dpi=150, bbox_inches='tight', facecolor=PALETTE['dark'])
plt.close()
print("   ✓ figures/fig4_roc_pr.png")


# ══════════════════════════════════════════════════════════════════
# PARTIE 1K — VISUALISATION 5 : SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════
print("\n▶  1K — SHAP Explainability (XGBoost)...")

# Sous-échantillon pour SHAP (pour rapidité)
idx_sample = np.random.RandomState(42).choice(len(X_scaled), min(800, len(X_scaled)), replace=False)
X_shap = X_scaled[idx_sample]

explainer   = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_shap)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor(PALETTE['dark'])

# Beeswarm SHAP
plt.sca(axes[0])
shap.summary_plot(shap_values, X_shap, feature_names=FEATURES,
                  show=False, plot_size=None, max_display=15)
axes[0].set_facecolor(PALETTE['mid'])
axes[0].set_title("SHAP — Impact sur la prédiction", color='white', fontsize=12, fontweight='bold')
axes[0].tick_params(colors='white')
for spine in axes[0].spines.values(): spine.set_color('#555')

# Bar plot SHAP
shap_mean = np.abs(shap_values).mean(0)
shap_series = pd.Series(shap_mean, index=FEATURES).sort_values(ascending=False).head(15)
axes[1].set_facecolor(PALETTE['mid'])
axes[1].barh(range(15), shap_series.values[::-1], color=PALETTE['accent'], alpha=0.85)
axes[1].set_yticks(range(15))
axes[1].set_yticklabels(shap_series.index[::-1], fontsize=9, color='white')
axes[1].set_title("SHAP — Importance moyenne |SHAP|", color='white', fontsize=12, fontweight='bold')
axes[1].set_xlabel("Mean |SHAP value|", color='white')
axes[1].tick_params(colors='white')
for spine in axes[1].spines.values(): spine.set_color('#555')

fig.suptitle("SHAP EXPLAINABILITY — XGBoost",
             fontsize=14, color='white', fontweight='bold')
plt.tight_layout()
plt.savefig("figures/fig5_shap.png", dpi=150, bbox_inches='tight', facecolor=PALETTE['dark'])
plt.close()
print("   ✓ figures/fig5_shap.png")


# ══════════════════════════════════════════════════════════════════
# PARTIE 1L — VISUALISATION 6 : ARBRE DE DÉCISION
# ══════════════════════════════════════════════════════════════════
print("\n▶  1L — Arbre de décision simplifié...")

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
dt.fit(X_scaled, y)

fig, ax = plt.subplots(figsize=(22, 10))
fig.patch.set_facecolor(PALETTE['dark'])
ax.set_facecolor(PALETTE['dark'])

plot_tree(dt, feature_names=FEATURES, class_names=['Normal', 'Ducting'],
          filled=True, rounded=True, fontsize=8, ax=ax,
          impurity=True, proportion=True,
          precision=3)

# Changer les couleurs
for text in ax.texts:
    text.set_color('black')

ax.set_title("ARBRE DE DÉCISION (profondeur=4) — Règles de détection du Ducting",
             fontsize=14, color='white', fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("figures/fig6_arbre_decision.png", dpi=120, bbox_inches='tight',
            facecolor=PALETTE['dark'])
plt.close()
print("   ✓ figures/fig6_arbre_decision.png")


# ══════════════════════════════════════════════════════════════════
# PARTIE 1M — VISUALISATION 7 : SCORE TEMPOREL PAR ZONE
# ══════════════════════════════════════════════════════════════════
print("\n▶  1M — Évolution temporelle du score de ducting...")

df['proba_rf']  = rf.predict_proba(scaler.transform(df[FEATURES].fillna(0).values))[:, 1]
df['proba_xgb'] = xgb.predict_proba(scaler.transform(df[FEATURES].fillna(0).values))[:, 1]
df['proba_ensemble'] = 0.5 * df['proba_rf'] + 0.5 * df['proba_xgb']

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor(PALETTE['dark'])
zones = sorted(df['zone'].unique())
n_zones = len(zones)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.3)

for i, zone in enumerate(zones):
    ax = fig.add_subplot(gs[i//2, i%2])
    ax.set_facecolor(PALETTE['mid'])
    dz = df[df['zone'] == zone].sort_values('time')

    ax.fill_between(dz['time'], dz['proba_ensemble'],
                    alpha=0.3, color=PALETTE['normal'])
    ax.plot(dz['time'], dz['proba_ensemble'],
            color=PALETTE['normal'], lw=1.2, alpha=0.8)

    # Marquer les jours de ducting réel
    duct_times = dz[dz['duction'] == 1]
    ax.scatter(duct_times['time'], duct_times['proba_ensemble'],
               color=PALETTE['ducting'], s=20, zorder=5, alpha=0.9)

    ax.axhline(0.35, color='yellow', lw=1, ls='--', alpha=0.7)
    ax.set_title(zone, color='white', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.tick_params(colors='white', labelsize=7)
    ax.set_xlabel("", color='white')
    for sp in ax.spines.values(): sp.set_color('#444')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

fig.suptitle("PROBABILITÉ DE DUCTING — Évolution temporelle par zone\n"
             "● Points rouges = ducting réel  |  Ligne jaune = seuil 0.35",
             fontsize=13, color='white', fontweight='bold')
plt.savefig("figures/fig7_evolution_temporelle.png", dpi=150, bbox_inches='tight',
            facecolor=PALETTE['dark'])
plt.close()
print("   ✓ figures/fig7_evolution_temporelle.png")


# ══════════════════════════════════════════════════════════════════
# PARTIE 1N — VISUALISATION 8 : MATRICE DE CONFUSION
# ══════════════════════════════════════════════════════════════════
print("\n▶  1N — Matrices de confusion...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(PALETTE['dark'])

for ax, model, proba, name, color in [
    (axes[0], rf,  y_proba_rf,  "Random Forest", PALETTE['normal']),
    (axes[1], xgb, y_proba_xgb, "XGBoost",       PALETTE['ducting']),
]:
    y_pred = (proba >= 0.35).astype(int)
    cm = confusion_matrix(y, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    ax.set_facecolor(PALETTE['mid'])
    sns.heatmap(cm, annot=False, fmt='', cmap='Blues', ax=ax,
                linewidths=2, linecolor='#333',
                cbar_kws={'shrink': 0.7})

    labels = [['VP\n(Correct\nNormal)', 'FP\n(Fausse\nAlarme)'],
              ['FN\n(Ducting\nRaté)', 'VN\n(Correct\nDucting)']]
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.4, f'{cm[i,j]}',
                    ha='center', fontsize=16, fontweight='bold',
                    color='white' if cm[i,j] < cm.max()/2 else 'black')
            ax.text(j + 0.5, i + 0.65, f'({cm_pct[i,j]:.1f}%)',
                    ha='center', fontsize=10, color='#ccc')
            ax.text(j + 0.5, i + 0.85, labels[i][j],
                    ha='center', fontsize=7, color='#aaa')

    ax.set_title(f"Matrice de Confusion — {name}\nAUC={roc_auc_score(y, proba):.3f}",
                 color='white', fontsize=12, fontweight='bold')
    ax.set_xticklabels(['Normal', 'Ducting'], color='white', fontsize=11)
    ax.set_yticklabels(['Normal', 'Ducting'], color='white', fontsize=11, rotation=0)
    ax.set_xlabel("Prédit", color='white', fontsize=11)
    ax.set_ylabel("Réel",   color='white', fontsize=11)

fig.suptitle("MATRICES DE CONFUSION (seuil = 0.35)",
             fontsize=14, color='white', fontweight='bold')
plt.tight_layout()
plt.savefig("figures/fig8_confusion.png", dpi=150, bbox_inches='tight', facecolor=PALETTE['dark'])
plt.close()
print("   ✓ figures/fig8_confusion.png")


# ══════════════════════════════════════════════════════════════════
# PARTIE 1O — SAUVEGARDE FINALE
# ══════════════════════════════════════════════════════════════════
print("\n▶  1O — Sauvegarde des fichiers modèles...")

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le,     "models/label_encoder.pkl")

meta = {
    "features": FEATURES,
    "zones": list(le.classes_),
    "seuil_decision": 0.35,
    "chemin_excel": CHEMIN_EXCEL,
    "performances": {
        "rf_auc":  round(float(rf_auc.mean()), 4),
        "rf_f1":   round(float(rf_f1.mean()),  4),
        "xgb_auc": round(float(xgb_auc.mean()), 4),
        "xgb_f1":  round(float(xgb_f1.mean()),  4),
    }
}
with open("models/meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("   ✓ models/scaler.pkl")
print("   ✓ models/label_encoder.pkl")
print("   ✓ models/meta.json")

print("\n" + "=" * 65)
print("  PARTIE 1 TERMINÉE")
print("=" * 65)
print(f"""
  Fichiers générés :
    models/rf_model.pkl      ← Random Forest
    models/xgb_model.pkl     ← XGBoost
    models/iso_model.pkl     ← IsolationForest
    models/scaler.pkl        ← Normaliseur
    models/meta.json         ← Métadonnées
    figures/fig1_analyse_exploratoire.png
    figures/fig2_correlation.png
    figures/fig3_importance_variables.png
    figures/fig4_roc_pr.png
    figures/fig5_shap.png
    figures/fig6_arbre_decision.png
    figures/fig7_evolution_temporelle.png
    figures/fig8_confusion.png

  → Lancer ensuite : python partie2_prediction.py
""")
