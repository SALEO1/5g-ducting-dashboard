# dashboard.py
import json
import os
import warnings
import base64
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

FEEDBACK_FILE = "feedback_ducting.csv"
TRAINING_DATASET_FILE = "dataset_reentrainement.csv"

# ------------------------------------------------------------------
# CONFIGURATION STREAMLIT
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Prédiction ducting atmosphérique",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------------
# PERSISTANCE FEEDBACK & DONNÉES DE RÉENTRAÎNEMENT
# ------------------------------------------------------------------
def charger_feedback():
    if os.path.exists(FEEDBACK_FILE):
        try:
            return pd.read_csv(FEEDBACK_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=["Horodateur_Saisie", "Zone", "Date", "Heure", "Score_Modele", "Statut_Modele", "Impact_Reel", "Commentaire"])

def enregistrer_feedback_et_features(df_feedback, df_features_row):
    # 1. Sauvegarde dans l'historique visuel du NOC
    df_existant = charger_feedback()
    df_combine = pd.concat([df_feedback, df_existant], ignore_index=True)
    df_combine = df_combine.drop_duplicates(subset=["Zone", "Date", "Heure"], keep="first")
    df_combine.to_csv(FEEDBACK_FILE, index=False)

    # 2. Sauvegarde des variables météo (Features) + Target (Impact_Reel: OUI=1, NON=0) pour le réentraînement
    if os.path.exists(TRAINING_DATASET_FILE):
        df_train_exist = pd.read_csv(TRAINING_DATASET_FILE)
        df_train_combine = pd.concat([df_features_row, df_train_exist], ignore_index=True)
        df_train_combine = df_train_combine.drop_duplicates(subset=["zone", "time"], keep="first")
        df_train_combine.to_csv(TRAINING_DATASET_FILE, index=False)
    else:
        df_features_row.to_csv(TRAINING_DATASET_FILE, index=False)

# ------------------------------------------------------------------
# MOTEUR DE RÉENTRAÎNEMENT IA EN CONTINU
# ------------------------------------------------------------------
def reentrainer_modele():
    if not os.path.exists(TRAINING_DATASET_FILE):
        return False, "Aucune donnée de validation terrain disponible pour le réentraînement."

    df_train = pd.read_csv(TRAINING_DATASET_FILE)
    if len(df_train) < 5:
        return False, f"Pas assez d'échantillons valides ({len(df_train)}/5 minimum requis pour réentraîner)."

    try:
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.preprocessing import StandardScaler

        X = df_train[FEATURES]
        y = df_train['target_ducting'].astype(int)

        # Réentraînement et ajustement
        scaler_new = StandardScaler()
        X_scaled = scaler_new.fit_transform(X)

        rf_new = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf_new.fit(X_scaled, y)

        xgb_new = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42)
        xgb_new.fit(X_scaled, y)

        # Sauvegarde des modèles mis à jour
        os.makedirs("models", exist_ok=True)
        joblib.dump(rf_new, "models/rf_model.pkl")
        joblib.dump(xgb_new, "models/xgb_model.pkl")
        joblib.dump(scaler_new, "models/scaler.pkl")

        # Invalidation du cache Streamlit pour charger les nouveaux modèles
        st.cache_resource.clear()
        return True, f"Modèle réentraîné avec succès sur {len(df_train)} observations réelles !"
    except Exception as e:
        return False, f"Erreur lors du réentraînement : {str(e)}"

# ------------------------------------------------------------------
# CHARGEMENT DE L'ARRIÈRE-PLAN
# ------------------------------------------------------------------
def get_base64_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded_string}"
    return None

bg_image_base64 = get_base64_background("arriere plan.png")

if bg_image_base64:
    # Voile sombre sur l'image pour garantir la lisibilité des données
    bg_css = f"""
    [data-testid="stAppViewContainer"] {{
        background-image: linear-gradient(180deg, rgba(6, 11, 22, 0.90), rgba(6, 11, 22, 0.96)), url('{bg_image_base64}') !important;
        background-size: cover !important;
        background-position: center center !important;
        background-attachment: fixed !important;
    }}
    """
else:
    # Fond type papier millimétré, très discret
    bg_css = """
    [data-testid="stAppViewContainer"] {
        background-color: #070D18 !important;
        background-image:
            radial-gradient(1100px 480px at 12% -8%, rgba(59, 196, 232, 0.10), transparent 60%),
            linear-gradient(rgba(255, 255, 255, 0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.025) 1px, transparent 1px) !important;
        background-size: auto, 36px 36px, 36px 36px !important;
        background-attachment: fixed !important;
    }
    """

# ------------------------------------------------------------------
# STYLE CSS PROFESSIONNEL HAUTE LISIBILITÉ
# ------------------------------------------------------------------
FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');"

CSS_PRINCIPAL = """
:root {
    --bg: #070D18;
    --surface: #0E182A;
    --surface-2: #15233A;
    --line: #22344F;
    --text: #E8EEF8;
    --muted: #8A9CB8;
    --accent: #3BC4E8;
    --ok: #2FBF8F;
    --warn: #E8B04A;
    --high: #F28C3A;
    --crit: #EF5A5A;
    --mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
}

/* ---------- Base ---------- */
.stApp, .stApp button, .stApp input, .stApp textarea {
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp label {
    color: var(--text) !important;
}

.stDeployButton,
[data-testid="stHeader"] > div:first-child,
button[title="Deploy"],
#MainMenu,
[data-testid="stActionButtonIcon"] {
    visibility: hidden !important;
    display: none !important;
}
[data-testid="stHeader"], footer { background: transparent !important; }
.block-container { padding-top: 1.8rem !important; max-width: 94% !important; }

/* ---------- En-tête ---------- */
.top-header-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    flex-wrap: wrap;
    background: linear-gradient(90deg, rgba(21, 35, 58, 0.95), rgba(14, 24, 42, 0.92));
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 18px 24px;
    margin-bottom: 22px;
    backdrop-filter: blur(10px);
}
.top-header-left { display: flex; align-items: center; gap: 16px; }
.top-header-title {
    color: var(--text);
    font-weight: 600;
    font-size: 24px;
    letter-spacing: -0.3px;
    line-height: 1.2;
}
.top-header-subtitle { color: var(--muted); font-size: 13px; margin-top: 2px; }
.top-header-chips { display: flex; gap: 8px; flex-wrap: wrap; }
.chip-info {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--muted);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 6px 12px;
    background: var(--bg);
}
.chip-info b { color: var(--accent); font-weight: 600; }

/* ---------- Boutons ---------- */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    background-color: var(--surface-2) !important;
    color: var(--text) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border: 1px solid var(--line) !important;
    border-radius: 6px !important;
    white-space: nowrap !important;
    padding: 9px 16px !important;
    min-height: 40px;
    transition: border-color 0.15s ease, color 0.15s ease;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background-color: var(--surface-2) !important;
}
.stButton > button:focus-visible, [data-testid="stFormSubmitButton"] > button:focus-visible {
    outline: 2px solid var(--accent) !important;
    outline-offset: 2px;
}
.stButton > button p, [data-testid="stFormSubmitButton"] > button p { color: inherit !important; }
[data-testid="stFormSubmitButton"] > button {
    background-color: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #05202B !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #66D4F1 !important;
    border-color: #66D4F1 !important;
    color: #05202B !important;
}

/* ---------- Titres de section ---------- */
.section-title {
    color: var(--text);
    font-weight: 600;
    font-size: 15px;
    margin: 6px 0 12px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
}
.section-title .side { color: var(--muted); font-weight: 400; font-size: 12px; font-family: var(--mono); }
.section-rule { border-top: 1px solid var(--line); margin: 26px 0 18px 0; }

.period-box {
    display: inline-flex;
    gap: 10px;
    align-items: baseline;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 9px 14px;
    margin-bottom: 14px;
    font-size: 13px;
}
.period-box .lbl { color: var(--muted); }
.period-box .val { color: var(--text); font-weight: 600; font-family: var(--mono); font-size: 12.5px; }

/* ---------- Bandeaux ---------- */
.status-banner-nominal, .status-banner-alert {
    display: flex;
    align-items: center;
    gap: 12px;
    border-radius: 8px;
    padding: 12px 18px;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 14px;
    color: var(--text);
}
.status-banner-nominal {
    background: rgba(47, 191, 143, 0.09);
    border: 1px solid rgba(47, 191, 143, 0.40);
}
.status-banner-alert {
    background: rgba(239, 90, 90, 0.11);
    border: 1px solid rgba(239, 90, 90, 0.50);
}
.status-dot { width: 10px; height: 10px; border-radius: 50%; flex: none; }
.status-banner-nominal .status-dot { background: var(--ok); }
.status-banner-alert .status-dot { background: var(--crit); animation: pulse-dot 1.8s ease-out infinite; }
@keyframes pulse-dot {
    0%   { box-shadow: 0 0 0 0 rgba(239, 90, 90, 0.55); }
    100% { box-shadow: 0 0 0 10px rgba(239, 90, 90, 0); }
}

.mode-info-banner {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 12px;
}

/* ---------- Indicateurs clés ---------- */
.kpi-row { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 26px; }
.kpi {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 14px 18px;
}
.kpi-l { color: var(--muted); font-size: 12.5px; font-weight: 500; }
.kpi-v { font-family: var(--mono); font-size: 28px; font-weight: 600; line-height: 1.2; margin-top: 4px; color: var(--text); }
.kpi-s { color: var(--muted); font-size: 12px; margin-top: 2px; font-family: var(--mono); }
@media (max-width: 900px) { .kpi-row { grid-template-columns: repeat(2, minmax(0, 1fr)); } }

/* ---------- Cartes de zones ---------- */
.zone-card {
    border: 1px solid var(--line);
    border-left-width: 4px;
    border-radius: 8px;
    padding: 14px 16px 12px 16px;
    margin-bottom: 8px;
}
.zone-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
.zone-name { color: var(--text); font-weight: 600; font-size: 15px; }
.zone-score {
    font-family: var(--mono);
    font-size: 36px;
    font-weight: 600;
    line-height: 1;
    margin-top: 12px;
}
.zone-unit { font-size: 16px; font-weight: 500; margin-left: 2px; opacity: 0.75; }
.zone-spark { margin-top: 10px; border-top: 1px solid rgba(255, 255, 255, 0.06); padding-top: 8px; }
.zone-spark svg { display: block; height: 46px; width: 100%; }
.zone-foot { color: var(--muted); font-size: 12px; font-family: var(--mono); margin-top: 6px; }

.level-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11.5px;
    font-weight: 600;
    border-radius: 999px;
    padding: 3px 10px;
}
.level-badge i { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }

/* ---------- Matrice zone x temps ---------- */
.hm-wrap {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 16px 18px 12px 18px;
    overflow-x: auto;
}
.hm { display: grid; gap: 2px; align-items: center; min-width: 560px; }
.hm-day {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    padding: 0 0 6px 3px;
    border-left: 1px solid var(--line);
    overflow: hidden;
    white-space: nowrap;
}
.hm-zone { font-size: 13px; font-weight: 500; color: var(--text); padding-right: 10px; white-space: nowrap; }
.hm-c { display: block; height: 22px; border-radius: 2px; }
.hm-c.d0 { margin-left: 3px; }
.hm-legend { display: flex; gap: 18px; flex-wrap: wrap; margin-top: 14px; font-size: 12px; color: var(--muted); }
.hm-legend span { display: inline-flex; align-items: center; gap: 6px; }
.hm-legend i { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }

/* ---------- Détail secteur ---------- */
.sector-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    margin: 0 0 12px 0;
}
.sector-name { color: var(--text); font-weight: 600; font-size: 20px; }
.sector-count { color: var(--muted); font-size: 13px; font-family: var(--mono); }
.sector-count strong { color: var(--crit); font-weight: 600; }

.chart-box {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 14px 16px 6px 12px;
    margin-bottom: 16px;
}
.chart-box svg { display: block; width: 100%; height: auto; }

.day-head {
    font-family: var(--mono);
    font-size: 12.5px;
    font-weight: 600;
    color: var(--muted);
    margin: 14px 0 8px 0;
}
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.hchip {
    display: inline-flex;
    align-items: baseline;
    gap: 10px;
    font-family: var(--mono);
    font-size: 13px;
    border-radius: 6px;
    padding: 6px 12px;
}
.hchip b { font-weight: 600; }

/* ---------- Onglets ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 28px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 10px 2px !important;
    color: var(--muted) !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab"] p { color: inherit !important; font-size: 14px; }
.stTabs [aria-selected="true"] { color: var(--text) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent) !important; height: 2px !important; }
.stTabs [data-baseweb="tab-border"] { background-color: transparent !important; }

/* ---------- Champs de formulaire ---------- */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="base-input"],
.stDateInput input, .stTextInput input {
    background-color: var(--surface-2) !important;
    border-color: var(--line) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
}
[data-baseweb="popover"] ul, [data-baseweb="menu"] { background-color: var(--surface) !important; }
[data-baseweb="select"] *, .stDateInput input, .stTextInput input { color: var(--text) !important; }
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { font-size: 13px !important; font-weight: 500 !important; color: var(--muted) !important; }
[data-testid="stForm"] {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    padding: 20px !important;
}

/* ---------- Validation terrain ---------- */
.pred-box {
    background: var(--surface-2);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 9px 14px;
    margin-top: 4px;
}
.pred-box .lbl { font-size: 12px; color: var(--muted); }
.pred-box strong { font-family: var(--mono); font-size: 15px; font-weight: 600; }

.stat-box {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 13px;
    color: var(--muted);
}
.stat-box strong { color: var(--accent); font-family: var(--mono); font-weight: 600; }

/* ---------- Messages & tableaux ---------- */
[data-testid="stAlert"] { border-radius: 6px !important; border: 1px solid var(--line) !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }

/* ---------- Pied de page ---------- */
.custom-footer {
    text-align: left;
    margin-top: 44px;
    padding-top: 14px;
    border-top: 1px solid var(--line);
    font-size: 12px;
    color: var(--muted) !important;
}

@media (prefers-reduced-motion: reduce) {
    .status-banner-alert .status-dot { animation: none; }
}
"""

st.markdown("<style>" + FONT_IMPORT + bg_css + CSS_PRINCIPAL + "</style>", unsafe_allow_html=True)

if 'active_mode' not in st.session_state:
    st.session_state['active_mode'] = None
if 'df_proc_cached' not in st.session_state:
    st.session_state['df_proc_cached'] = None

# ------------------------------------------------------------------
# CHARGEMENT MOTEUR DE PRÉDICTION
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def charger_moteur():
    try:
        rf = joblib.load("models/rf_model.pkl")
        xgb = joblib.load("models/xgb_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        le = joblib.load("models/label_encoder.pkl")
        with open("models/meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        return rf, xgb, scaler, le, meta
    except:
        return None, None, None, None, {"features": []}

rf, xgb, scaler, le, meta = charger_moteur()
FEATURES = meta.get("features", [])
SEUIL_ALERTE = meta.get("seuil_decision", 0.35)
POIDS_XGB = meta.get("poids_xgb", 0.55)
POIDS_RF = meta.get("poids_rf", 0.45)

# ------------------------------------------------------------------
# EN-TÊTE PERMANENT
# ------------------------------------------------------------------
st.markdown(
    '<div class="top-header-container">'
    '<div class="top-header-left">'
    '<svg width="42" height="42" viewBox="0 0 34 34" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<rect width="34" height="34" rx="8" fill="#0E182A" stroke="#22344F"/>'
    '<path d="M6 21c3-7 6-7 9 0s6 7 9 0c1-2 2-3 4-4" stroke="#3BC4E8" stroke-width="2" stroke-linecap="round"/>'
    '<path d="M6 13c3-4 6-4 9 0s6 4 9 0" stroke="#3BC4E8" stroke-opacity="0.45" stroke-width="2" stroke-linecap="round"/>'
    '</svg>'
    '<div>'
    '<div class="top-header-title">Prédiction ducting atmosphérique</div>'
    '<div class="top-header-subtitle">Réseau 5G TDD</div>'
    '</div>'
    '</div>'
    '<div class="top-header-chips">'
    '<span class="chip-info">Zones <b>8</b></span>'
    f'<span class="chip-info">Seuil d\'alerte <b>{int(SEUIL_ALERTE * 100)} %</b></span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)

# Mise en évidence du mode actif (affichage uniquement)
_mode_actif = st.session_state.get('active_mode')
if _mode_actif in ('custom', 'weekly'):
    st.markdown(
        f"<style>.st-key-mode_{_mode_actif} button {{ border-color: #3BC4E8 !important; color: #3BC4E8 !important; background-color: rgba(59, 196, 232, 0.10) !important; }}</style>",
        unsafe_allow_html=True
    )

# ------------------------------------------------------------------
# CALCUL DES FEATURES ATMOSPHÉRIQUES
# ------------------------------------------------------------------
def calculer_features(df):
    df = df.copy()
    if "sst (°C)" in df.columns:
        df = df.rename(columns={"sst (°C)": "sst"})
    df["time"] = pd.to_datetime(df["time"])

    if 'zone' in df.columns:
        df['zone'] = df['zone'].astype(str).str.strip()

    df = df.sort_values(["zone", "time"]).reset_index(drop=True)
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df.groupby("zone")[col].transform(lambda x: x.fillna(x.median()))

    df['inversion_basse'] = df['temperature_925hPa (°C)'] - df['temperature_1000hPa (°C)']
    df['inversion_moyenne'] = df['temperature_900hPa (°C)'] - df['temperature_925hPa (°C)']
    df['inversion_totale'] = df['temperature_900hPa (°C)'] - df['temperature_1000hPa (°C)']
    df['lapse_rate_bas'] = df['inversion_basse'] / 0.75
    df['lapse_rate_haut'] = df['inversion_moyenne'] / 0.25
    df['diff_air_sst'] = df['temperature_2m (°C)'] - df['sst']
    df['grad_hum_bas'] = df['relative_humidity_2m (%)'] - df['relative_humidity_1000hPa (%)']
    df['grad_hum_moyen'] = df['relative_humidity_1000hPa (%)'] - df['relative_humidity_950hPa (%)']
    df['hum_total_gradient'] = df['relative_humidity_2m (%)'] - df['relative_humidity_900hPa (%)']

    T_K = df['temperature_2m (°C)'] + 273.15
    e_s = 6.112 * np.exp(17.67 * df['temperature_2m (°C)'] / (df['temperature_2m (°C)'] + 243.5))
    e = (df['relative_humidity_2m (%)'] / 100.0) * e_s
    df['refractivite_N'] = (77.6 * df['pressure_msl (hPa)'] / T_K) + (373e3 * e / T_K**2)

    def norm(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 0 else s * 0

    df['score_inversion'] = norm(df['inversion_basse'].clip(lower=0))
    df['score_humidite'] = norm(df['relative_humidity_2m (%)'])
    df['score_vent_calme'] = norm(1 / df['wind_speed_10m (km/h)'].clip(lower=0.5))
    df['score_pression'] = norm(df['pressure_msl (hPa)'])
    df['score_sst_diff'] = norm(df['diff_air_sst'].clip(lower=0))
    df['ducting_score'] = (0.35 * df['score_inversion'] + 0.20 * df['score_humidite'] + 0.20 * df['score_vent_calme'] + 0.15 * df['score_pression'] + 0.10 * df['score_sst_diff'])
    df['heure'] = df['time'].dt.hour
    df['mois'] = df['time'].dt.month
    df['saison'] = df['mois'].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})

    if 'zone' in df.columns and le is not None:
        try:
            df['zone_code'] = le.transform(df['zone'])
        except ValueError:
            classes_dict = {str(c).strip().lower(): i for i, c in enumerate(le.classes_)}
            df['zone_code'] = df['zone'].str.lower().map(classes_dict).fillna(0).astype(int)

    df['is_inversion'] = (df['inversion_basse'] > 1.0).astype(int)
    df['is_vent_calme'] = (df['wind_speed_10m (km/h)'] < 15).astype(int)
    df['is_anticyclone'] = (df['pressure_msl (hPa)'] > 1020).astype(int)
    df['is_mer_froide'] = (df['diff_air_sst'] > 3).astype(int)
    return df

# ------------------------------------------------------------------
# COLLECTE MÉTÉO API
# ------------------------------------------------------------------
def recuperer_donnees_open_meteo(start_date, end_date):
    villes = {
        "Tangier": {"lat": 35.7696, "lon": -5.8034},
        "Asilah": {"lat": 35.4650, "lon": -6.0348},
        "Larache": {"lat": 35.1932, "lon": -6.1563},
        "Kenitra": {"lat": 34.2610, "lon": -6.5802},
        "Sidi Yahya": {"lat": 34.3049, "lon": -6.3046},
        "Sidi Slimane": {"lat": 34.2648, "lon": -5.9250},
        "Mohammedia": {"lat": 33.6861, "lon": -7.3829},
        "Casablanca": {"lat": 33.5731, "lon": -7.5898}
    }

    noms_villes = list(villes.keys())
    latitudes = [str(villes[v]["lat"]) for v in noms_villes]
    longitudes = [str(villes[v]["lon"]) for v in noms_villes]

    base_url = "https://api.open-meteo.com/v1/forecast"
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1.5, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    liste_variables = [
        "temperature_2m", "relative_humidity_2m", "pressure_msl", "wind_speed_10m",
        "temperature_1000hPa", "temperature_925hPa", "temperature_900hPa",
        "relative_humidity_1000hPa", "relative_humidity_950hPa", "relative_humidity_900hPa",
        "temperature_80m", "temperature_975hPa", "temperature_950hPa",
        "relative_humidity_975hPa", "surface_pressure", "wind_speed_80m",
        "wind_speed_1000hPa", "cloud_cover", "cloud_cover_low",
        "visibility", "vapour_pressure_deficit"
    ]

    params = {
        "latitude": ",".join(latitudes),
        "longitude": ",".join(longitudes),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ",".join(liste_variables),
        "timezone": "Africa/Casablanca"
    }

    response = session.get(base_url, params=params, timeout=30)
    if response.status_code != 200:
        raise Exception(f"Erreur API Open-Meteo ({response.status_code})")

    data = response.json()
    responses_list = data if isinstance(data, list) else [data]

    all_dfs = []
    for idx, res in enumerate(responses_list):
        hourly_data = res.get("hourly", {})
        if not hourly_data:
            continue

        df_ville = pd.DataFrame({
            "time": pd.to_datetime(hourly_data["time"]),
            "zone": noms_villes[idx].strip(),
            "temperature_2m (°C)": hourly_data["temperature_2m"],
            "relative_humidity_2m (%)": hourly_data["relative_humidity_2m"],
            "pressure_msl (hPa)": hourly_data["pressure_msl"],
            "wind_speed_10m (km/h)": hourly_data["wind_speed_10m"],
            "temperature_1000hPa (°C)": hourly_data["temperature_1000hPa"],
            "temperature_925hPa (°C)": hourly_data["temperature_925hPa"],
            "temperature_900hPa (°C)": hourly_data["temperature_900hPa"],
            "relative_humidity_1000hPa (%)": hourly_data["relative_humidity_1000hPa"],
            "relative_humidity_950hPa (%)": hourly_data["relative_humidity_950hPa"],
            "relative_humidity_900hPa (%)": hourly_data["relative_humidity_900hPa"],
            "temperature_80m (°C)": hourly_data["temperature_80m"],
            "temperature_975hPa (°C)": hourly_data["temperature_975hPa"],
            "temperature_950hPa (°C)": hourly_data["temperature_950hPa"],
            "relative_humidity_975hPa (%)": hourly_data["relative_humidity_975hPa"],
            "surface_pressure (hPa)": hourly_data["surface_pressure"],
            "wind_speed_80m (km/h)": hourly_data["wind_speed_80m"],
            "wind_speed_1000hPa (km/h)": hourly_data["wind_speed_1000hPa"],
            "cloud_cover (%)": hourly_data["cloud_cover"],
            "cloud_cover_low (%)": hourly_data["cloud_cover_low"],
            "visibility (m)": hourly_data["visibility"],
            "vapour_pressure_deficit (kPa)": hourly_data["vapour_pressure_deficit"]
        })
        df_ville["sst"] = df_ville["temperature_2m (°C)"] - 1.5
        all_dfs.append(df_ville)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def obtenir_prochain_dimanche(ref_date=None):
    if ref_date is None:
        ref_date = datetime.now().date()
    delta = (6 - ref_date.weekday()) % 7
    return ref_date + timedelta(days=delta)

# ------------------------------------------------------------------
# BOUTONS SELECTION MODE
# ------------------------------------------------------------------
col_m1, col_m2 = st.columns(2)

with col_m1:
    if st.button("Analyse personnalisée", key="mode_custom", use_container_width=True):
        st.session_state['active_mode'] = 'custom'
        st.session_state['df_proc_cached'] = None
        st.rerun()

with col_m2:
    if st.button("Prédiction hebdomadaire", key="mode_weekly", use_container_width=True):
        st.session_state['active_mode'] = 'weekly'
        st.session_state['df_proc_cached'] = None
        st.rerun()

st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

target_start, target_end = None, None
mode_label = ""
do_execute = False

if st.session_state['active_mode'] == 'custom':
    c_d1, c_d2, c_act, c_reset = st.columns([2, 2, 2, 1.2])
    start_input = c_d1.date_input("Date de début", datetime.now().date())
    end_input = c_d2.date_input("Date de fin", datetime.now().date() + timedelta(days=2))

    # Alignement des boutons avec les champs de saisie
    c_act.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    c_reset.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    if c_act.button("Lancer l'analyse", use_container_width=True):
        if start_input > end_input:
            st.error("Date de début supérieure à la date de fin.")
        else:
            target_start, target_end = start_input, end_input
            mode_label = f"Analyse personnalisée  |  {target_start.strftime('%d/%m/%Y')} au {target_end.strftime('%d/%m/%Y')}"
            do_execute = True

    if c_reset.button("Accueil", use_container_width=True):
        st.session_state['active_mode'] = None
        st.session_state['df_proc_cached'] = None
        st.rerun()

elif st.session_state['active_mode'] == 'weekly':
    dim_prochain = obtenir_prochain_dimanche()
    sam_prochain = dim_prochain + timedelta(days=6)

    st.markdown(
        '<div class="period-box">'
        '<span class="lbl">Période</span>'
        f'<span class="val">{dim_prochain.strftime("%d/%m/%Y")} au {sam_prochain.strftime("%d/%m/%Y")}</span>'
        '</div>',
        unsafe_allow_html=True
    )

    c_act, c_reset, _ = st.columns([2.5, 1.2, 3])
    if c_act.button("Lancer la prédiction", use_container_width=True):
        target_start, target_end = dim_prochain, sam_prochain
        mode_label = f"Prédiction hebdomadaire  |  {target_start.strftime('%d/%m/%Y')} au {target_end.strftime('%d/%m/%Y')}"
        do_execute = True

    if c_reset.button("Accueil", use_container_width=True):
        st.session_state['active_mode'] = None
        st.session_state['df_proc_cached'] = None
        st.rerun()

# ------------------------------------------------------------------
# CALCUL DES PRÉDICTIONS
# ------------------------------------------------------------------
if do_execute:
    with st.spinner("Collecte et analyse météo en cours..."):
        try:
            df_raw = recuperer_donnees_open_meteo(target_start, target_end)
            if not df_raw.empty:
                df_proc = calculer_features(df_raw)
                X = df_proc[FEATURES]
                X_scaled = scaler.transform(X)
                proba_rf = rf.predict_proba(X_scaled)[:, 1]
                proba_xgb = xgb.predict_proba(X_scaled)[:, 1]
                df_proc["Score_Ducting"] = np.round(POIDS_XGB * proba_xgb + POIDS_RF * proba_rf, 3)
                df_proc["Statut"] = np.where(df_proc["Score_Ducting"] >= SEUIL_ALERTE, "CRITIQUE", "NOMINAL")

                st.session_state['df_proc_cached'] = df_proc
                st.session_state['selected_city'] = None
                st.session_state['mode_label'] = mode_label
        except Exception as e:
            st.error(f"Erreur lors de l'exécution : {e}")

# ------------------------------------------------------------------
# HELPERS D'AFFICHAGE (aucun impact sur les prédictions)
# ------------------------------------------------------------------
def Obtenir_Couleur_Niveau(score_max):
    if score_max < 0.35:
        return "#2FBF8F"
    elif score_max < 0.50:
        return "#E8B04A"
    elif score_max < 0.70:
        return "#F28C3A"
    return "#EF5A5A"

def Obtenir_Style_Bloc(score_max):
    couleur = Obtenir_Couleur_Niveau(score_max)
    if score_max < 0.35:
        libelle = "Nominal"
        bg, border = "rgba(14, 24, 42, 0.95)", "#22344F"
    elif score_max < 0.50:
        libelle = "Modéré"
        bg, border = "rgba(14, 24, 42, 0.95)", couleur
    elif score_max < 0.70:
        libelle = "Élevé"
        bg, border = "rgba(14, 24, 42, 0.95)", couleur
    else:
        libelle = "Critique"
        bg, border = "rgba(239, 90, 90, 0.09)", couleur
    badge = (
        f"<span class='level-badge' style='color:{couleur}; background:{couleur}22; border:1px solid {couleur}66;'>"
        f"<i style='background:{couleur};'></i>{libelle}</span>"
    )
    return badge, bg, border

def couleur_cellule(score):
    if score < 0.35:
        return f"rgba(47,191,143,{0.08 + 0.22 * score / 0.35:.2f})"
    elif score < 0.50:
        return "rgba(232,176,74,0.85)"
    elif score < 0.70:
        return "rgba(242,140,58,0.90)"
    return "rgba(239,90,90,0.95)"

def construire_svg_courbe(times, scores, detail=False):
    """Courbe du score de ducting. detail=False : mini courbe de carte, detail=True : graphique complet."""
    n = len(scores)
    if n < 2:
        return ""
    if detail:
        W, H, L, R, T, B = 900, 230, 46, 14, 14, 30
    else:
        W, H, L, R, T, B = 200, 46, 0, 0, 4, 4
    pw, ph = W - L - R, H - T - B

    def X(i):
        return L + pw * i / (n - 1)

    def Y(s):
        return T + ph * (1 - min(max(float(s), 0.0), 1.0))

    pts = " ".join(f"{X(i):.1f},{Y(s):.1f}" for i, s in enumerate(scores))
    couleur = "#3BC4E8" if detail else Obtenir_Couleur_Niveau(max(scores))
    base = T + ph
    y_seuil = Y(SEUIL_ALERTE)

    if detail:
        svg = [f"<svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg'>"]
        for v in (0, 0.25, 0.5, 0.75, 1):
            y = Y(v)
            svg.append(f"<line x1='{L}' x2='{W - R}' y1='{y:.1f}' y2='{y:.1f}' stroke='rgba(255,255,255,0.07)' stroke-width='1'/>")
            svg.append(f"<text x='{L - 8}' y='{y + 3.5:.1f}' text-anchor='end' font-size='10' fill='#8A9CB8' font-family='IBM Plex Mono, monospace'>{int(v * 100)}</text>")
        for i, t in enumerate(times):
            if t.hour == 0:
                svg.append(f"<line x1='{X(i):.1f}' x2='{X(i):.1f}' y1='{T}' y2='{base}' stroke='rgba(255,255,255,0.10)' stroke-width='1'/>")
                svg.append(f"<text x='{X(i) + 4:.1f}' y='{base + 18}' font-size='10' fill='#8A9CB8' font-family='IBM Plex Mono, monospace'>{t.strftime('%d/%m')}</text>")
    else:
        svg = [f"<svg viewBox='0 0 {W} {H}' preserveAspectRatio='none' xmlns='http://www.w3.org/2000/svg'>"]

    svg.append(f"<polygon points='{L},{base} {pts} {L + pw},{base}' fill='{couleur}' fill-opacity='0.14'/>")
    svg.append(f"<line x1='{L}' x2='{W - R}' y1='{y_seuil:.1f}' y2='{y_seuil:.1f}' stroke='#EF5A5A' stroke-width='1' stroke-dasharray='4 4' stroke-opacity='0.85' vector-effect='non-scaling-stroke'/>")
    if detail:
        svg.append(f"<text x='{W - R - 4}' y='{y_seuil - 5:.1f}' text-anchor='end' font-size='10' fill='#EF5A5A' font-family='IBM Plex Mono, monospace'>Seuil {int(SEUIL_ALERTE * 100)} %</text>")
    svg.append(f"<polyline points='{pts}' fill='none' stroke='{couleur}' stroke-width='{1.8 if detail else 1.6}' stroke-linejoin='round' vector-effect='non-scaling-stroke'/>")
    if detail and n <= 200:
        for i, s in enumerate(scores):
            if s >= SEUIL_ALERTE:
                svg.append(f"<circle cx='{X(i):.1f}' cy='{Y(s):.1f}' r='3' fill='#EF5A5A'/>")
    svg.append("</svg>")
    return "".join(svg)

def construire_matrice(df, villes):
    """Matrice zone x temps : une cellule par heure, colorée selon le niveau de risque."""
    temps = [pd.Timestamp(t) for t in sorted(df["time"].unique())]
    piv = df.pivot_table(index="zone", columns="time", values="Score_Ducting", aggfunc="max")
    n = len(temps)

    jours = {}
    for t in temps:
        jours[t.date()] = jours.get(t.date(), 0) + 1

    html = [f"<div class='hm' style='grid-template-columns: 110px repeat({n}, minmax(0, 1fr));'>", "<div></div>"]
    for d, k in jours.items():
        html.append(f"<div class='hm-day' style='grid-column: span {k};'>{d.strftime('%d/%m')}</div>")

    for ville in villes:
        if ville not in piv.index:
            continue
        html.append(f"<div class='hm-zone'>{ville}</div>")
        jour_prec = None
        for t in temps:
            s = piv.loc[ville, t] if t in piv.columns else 0.0
            if pd.isna(s):
                s = 0.0
            cls = "hm-c d0" if t.date() != jour_prec else "hm-c"
            jour_prec = t.date()
            html.append(f"<i class='{cls}' style='background:{couleur_cellule(s)};' title='{ville} {t.strftime('%d/%m %H:00')} : {int(s * 100)} %'></i>")
    html.append("</div>")
    html.append(
        "<div class='hm-legend'>"
        "<span><i style='background:rgba(47,191,143,0.30);'></i>Nominal</span>"
        "<span><i style='background:rgba(232,176,74,0.85);'></i>Modéré 35-50 %</span>"
        "<span><i style='background:rgba(242,140,58,0.90);'></i>Élevé 50-70 %</span>"
        "<span><i style='background:rgba(239,90,90,0.95);'></i>Critique 70 % et plus</span>"
        "</div>"
    )
    return "".join(html)

def construire_kpi(label, valeur, sous_texte="", couleur=None):
    style = f" style='color:{couleur};'" if couleur else ""
    return (
        f"<div class='kpi'><div class='kpi-l'>{label}</div>"
        f"<div class='kpi-v'{style}>{valeur}</div>"
        f"<div class='kpi-s'>{sous_texte}</div></div>"
    )

# ------------------------------------------------------------------
# TABLEAU DE BORD (PRÉVISION + FEEDBACK CONTINU)
# ------------------------------------------------------------------
if 'df_proc_cached' in st.session_state and st.session_state['df_proc_cached'] is not None:
    df_proc = st.session_state['df_proc_cached']

    if 'mode_label' in st.session_state:
        st.markdown(f'<div class="mode-info-banner">{st.session_state["mode_label"]}</div>', unsafe_allow_html=True)

    tab_previsions, tab_validation = st.tabs([
        "Prévisions",
        "Validation terrain"
    ])

    with tab_previsions:
        df_alertes = df_proc[df_proc["Statut"] == "CRITIQUE"].sort_values("time")
        if df_alertes.empty:
            st.markdown('<div class="status-banner-nominal"><span class="status-dot"></span>Propagation nominale</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-banner-alert"><span class="status-dot"></span>Alerte ducting sur le littoral</div>', unsafe_allow_html=True)

        # Indicateurs clés
        ligne_pic = df_proc.loc[df_proc["Score_Ducting"].idxmax()]
        nb_zones_total = df_proc["zone"].nunique()
        nb_zones_alerte = df_alertes["zone"].nunique()
        couleur_alerte = "#EF5A5A" if nb_zones_alerte > 0 else "#2FBF8F"
        st.markdown(
            '<div class="kpi-row">'
            + construire_kpi("Zones en alerte", f"{nb_zones_alerte}/{nb_zones_total}", "", couleur_alerte)
            + construire_kpi("Créneaux critiques", f"{len(df_alertes)}", "heures au-dessus du seuil", couleur_alerte)
            + construire_kpi("Pic de risque", f"{int(ligne_pic['Score_Ducting'] * 100)} %", f"{ligne_pic['zone']}, {ligne_pic['time'].strftime('%d/%m %H:00')}", Obtenir_Couleur_Niveau(ligne_pic['Score_Ducting']))
            + construire_kpi("Période", f"{df_proc['time'].min().strftime('%d/%m')} au {df_proc['time'].max().strftime('%d/%m')}", f"{df_proc['time'].nunique()} heures analysées")
            + '</div>',
            unsafe_allow_html=True
        )

        # Cartes par zone
        st.markdown('<div class="section-title">Zones</div>', unsafe_allow_html=True)

        toutes_villes = ["Tangier", "Asilah", "Larache", "Kenitra", "Sidi Yahya", "Sidi Slimane", "Mohammedia", "Casablanca"]
        lignes_villes = [toutes_villes[0:4], toutes_villes[4:8]]

        for ligne in lignes_villes:
            cols = st.columns(4)
            for idx, ville in enumerate(ligne):
                df_ville_all = df_proc[df_proc["zone"] == ville].sort_values("time")
                max_score = df_ville_all["Score_Ducting"].max() if not df_ville_all.empty else 0.0
                badge_html, bg_card, border_card = Obtenir_Style_Bloc(max_score)
                couleur_niveau = Obtenir_Couleur_Niveau(max_score)
                pct_score = int(max_score * 100)

                if not df_ville_all.empty:
                    scores_ville = df_ville_all["Score_Ducting"].tolist()
                    temps_ville = df_ville_all["time"].tolist()
                    spark_html = construire_svg_courbe(temps_ville, scores_ville)
                    heure_pic = temps_ville[int(np.argmax(scores_ville))].strftime('%d/%m %H:00')
                    pied = f"Pic {heure_pic}"
                else:
                    spark_html, pied = "", "Aucune donnée"

                selectionnee = st.session_state.get('selected_city') == ville
                bordure = "#3BC4E8" if selectionnee else "#22344F"
                halo = "box-shadow: 0 0 0 1px #3BC4E8, 0 0 22px rgba(59,196,232,0.18);" if selectionnee else ""

                with cols[idx]:
                    st.markdown(
                        f'<div class="zone-card" style="background:{bg_card}; border-color:{bordure}; border-left-color:{couleur_niveau}; {halo}">'
                        f'<div class="zone-head"><span class="zone-name">{ville}</span>{badge_html}</div>'
                        f'<div class="zone-score" style="color:{couleur_niveau};">{pct_score}<span class="zone-unit">%</span></div>'
                        f'<div class="zone-spark">{spark_html}</div>'
                        f'<div class="zone-foot">{pied}</div>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                    if st.button("Détails", key=f"btn_{ville}", use_container_width=True):
                        st.session_state['selected_city'] = ville
                        st.rerun()

        # Matrice zone x temps
        st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Matrice zone × temps<span class="side">1 cellule = 1 heure</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hm-wrap">{construire_matrice(df_proc, toutes_villes)}</div>', unsafe_allow_html=True)

        # Détail du secteur
        selected_ville = st.session_state.get('selected_city')
        if selected_ville is not None:
            df_ville_data = df_proc[df_proc["zone"] == selected_ville].sort_values("time")
            df_ville_alertes = df_ville_data[df_ville_data["Statut"] == "CRITIQUE"]

            st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="sector-bar">'
                f'<div class="sector-name">{selected_ville}</div>'
                f'<div class="sector-count"><strong>{len(df_ville_alertes)}</strong> créneaux critiques</div>'
                '</div>',
                unsafe_allow_html=True
            )

            courbe_detail = construire_svg_courbe(df_ville_data["time"].tolist(), df_ville_data["Score_Ducting"].tolist(), detail=True)
            if courbe_detail:
                st.markdown(f'<div class="chart-box">{courbe_detail}</div>', unsafe_allow_html=True)

            if df_ville_alertes.empty:
                st.info(f"Aucun dépassement du seuil critique pour {selected_ville}.")
            else:
                df_ville_alertes['Date_Jour'] = df_ville_alertes['time'].dt.strftime('%d/%m/%Y')
                df_ville_alertes['Heure'] = df_ville_alertes['time'].dt.strftime('%H:00')
                for date_str, group in df_ville_alertes.groupby('Date_Jour', sort=False):
                    puces = ""
                    for _, r in group.iterrows():
                        c = Obtenir_Couleur_Niveau(r['Score_Ducting'])
                        puces += (
                            f"<span class='hchip' style='background:{c}1F; border:1px solid {c}66; color:{c};'>"
                            f"{r['Heure']}<b>{int(r['Score_Ducting'] * 100)} %</b></span>"
                        )
                    st.markdown(f"<div class='day-head'>{date_str}</div><div class='chips'>{puces}</div>", unsafe_allow_html=True)

    # ==================================================================
    # ONGLET 2 : FEEDBACK & BOUTON RÉENTRAÎNEMENT CONTINU
    # ==================================================================
    with tab_validation:
        st.markdown('<div class="section-title" style="margin-top:18px;">Validation terrain</div>', unsafe_allow_html=True)

        villes_disponibles = sorted(df_proc["zone"].unique().tolist())
        dates_disponibles = sorted(df_proc["time"].dt.strftime('%Y-%m-%d').unique().tolist())

        col_v1, col_v2 = st.columns(2)
        val_zone = col_v1.selectbox("Zone", villes_disponibles)
        val_date = col_v2.selectbox("Date", dates_disponibles)

        df_sub = df_proc[(df_proc["zone"] == val_zone) & (df_proc["time"].dt.strftime('%Y-%m-%d') == val_date)].copy()
        df_sub['Heure'] = df_sub['time'].dt.strftime('%H:00')
        df_sub = df_sub.sort_values("time")

        with st.form("form_validation_duct"):
            heures_dispo = df_sub["Heure"].tolist()
            c_h1, c_h2, c_h3 = st.columns([1.5, 2, 2.5])
            val_heure = c_h1.selectbox("Heure", heures_dispo)

            row_heure = df_sub[df_sub["Heure"] == val_heure].iloc[0]
            pred_score = row_heure["Score_Ducting"]
            pred_statut = row_heure["Statut"]

            c_h2.markdown(
                '<div class="pred-box">'
                '<span class="lbl">Prédiction du modèle</span><br/>'
                f'<strong style="color:{"#EF5A5A" if pred_statut == "CRITIQUE" else "#2FBF8F"}">{pred_statut.capitalize()}  {int(pred_score*100)} %</strong>'
                '</div>',
                unsafe_allow_html=True
            )

            impact_observe = c_h3.radio(
                "Impact réel sur la 5G TDD",
                options=["OUI - Perturbation constatée", "NON - Aucun impact constaté"]
            )

            commentaire_noc = st.text_input("Commentaire / réf. ticket NOC (optionnel)")
            btn_sauvegarder = st.form_submit_button("Enregistrer")

            if btn_sauvegarder:
                impact_bool = "OUI" if "OUI" in impact_observe else "NON"

                # 1. Enregistrement visuel
                df_feedback_new = pd.DataFrame([{
                    "Horodateur_Saisie": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Zone": val_zone,
                    "Date": val_date,
                    "Heure": val_heure,
                    "Score_Modele": pred_score,
                    "Statut_Modele": pred_statut,
                    "Impact_Reel": impact_bool,
                    "Commentaire": commentaire_noc
                }])

                # 2. Extrait des features météo exactes + Label réel (OUI=1, NON=0)
                df_feature_row = row_heure[FEATURES + ['zone', 'time']].to_frame().T
                df_feature_row['target_ducting'] = 1 if impact_bool == "OUI" else 0

                enregistrer_feedback_et_features(df_feedback_new, df_feature_row)
                st.success("Validation enregistrée.")

        # SECTION DE RÉENTRAÎNEMENT DU MODÈLE EN UN CLIC
        st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Réentraînement du modèle</div>', unsafe_allow_html=True)

        col_retrain1, col_retrain2 = st.columns([2, 3])

        with col_retrain1:
            if st.button("Réentraîner", use_container_width=True):
                with st.spinner("Réentraînement des modèles Random Forest & XGBoost en cours..."):
                    succes, msg = reentrainer_modele()
                    if succes:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.warning(msg)

        with col_retrain2:
            nb_samples = len(pd.read_csv(TRAINING_DATASET_FILE)) if os.path.exists(TRAINING_DATASET_FILE) else 0
            st.markdown(
                '<div class="stat-box">'
                f'Échantillons terrain : <strong>{nb_samples}</strong>'
                '</div>',
                unsafe_allow_html=True
            )

        # HISTORIQUE ENREGISTRÉ
        st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Historique</div>', unsafe_allow_html=True)

        df_historique = charger_feedback()
        if not df_historique.empty:
            st.dataframe(df_historique.sort_values("Horodateur_Saisie", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("Aucune validation enregistrée.")

st.markdown('<div class="custom-footer">Prédiction ducting atmosphérique, NOC Infrastructure</div>', unsafe_allow_html=True)