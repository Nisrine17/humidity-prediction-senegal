"""
Application Streamlit — Estimation de l'humidité au Sénégal.

Lancement :
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "humidity_pipeline.joblib"
META_PATH = ROOT / "models" / "metadata.json"
DATA_PATH = ROOT / "data" / "meteo_departements_Senegal.csv"
FIG_DIR = ROOT / "reports" / "figures"

WEATHER_FR = {
    "clear sky": "Ciel dégagé",
    "few clouds": "Quelques nuages",
    "scattered clouds": "Nuages épars",
    "broken clouds": "Nuages fragmentés",
    "overcast clouds": "Ciel couvert",
    "light rain": "Pluie légère",
    "moderate rain": "Pluie modérée",
    "thunderstorm with rain": "Orage avec pluie",
}

st.set_page_config(page_title="Humidité au Sénégal", page_icon="💧", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    return json.loads(META_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).drop_duplicates()
    return df


def interpret(h: float) -> str:
    if h < 30:
        return "Air très sec"
    if h < 50:
        return "Air plutôt sec"
    if h < 70:
        return "Humidité modérée"
    if h < 85:
        return "Air humide"
    return "Air très humide"


# --------------------------------------------------------------------------- #
if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error("Modèle introuvable. Lancez d'abord `python train.py` pour le générer.")
    st.stop()

pipe = load_model()
meta = load_metadata()
df = load_dataset()

lodo = meta["validation"]["leave_one_date_out"]
typical_mae = lodo["MAE_mean"]
typical_rmse = lodo["RMSE_mean"]

st.title("💧 Estimation de l'humidité au Sénégal")
st.caption(
    "Projet Data Science — Master 2. Modèle entraîné sur 5 journées d'octobre 2025 "
    "(45 départements, 14 régions)."
)

tab_pred, tab_data, tab_model = st.tabs(["Estimation", "Données", "Modèle et limites"])

# ------------------------------- Estimation -------------------------------- #
with tab_pred:
    st.subheader("Renseignez les conditions observées")
    col1, col2 = st.columns(2)

    with col1:
        region = st.selectbox("Région", sorted(meta["region_to_departements"]))
        departement = st.selectbox("Département", meta["region_to_departements"][region])
        weather_key = st.selectbox(
            "Conditions météorologiques",
            meta["weather_categories"],
            format_func=lambda k: WEATHER_FR.get(k, k),
        )

    t_min, t_max = meta["ranges"]["temperature"]
    w_min, w_max = meta["ranges"]["wind_speed"]
    with col2:
        temperature = st.number_input(
            "Température (°C)", min_value=10.0, max_value=50.0, value=30.0, step=0.5,
            help=f"Plage observée dans les données : {t_min:.1f} – {t_max:.1f} °C",
        )
        wind_speed = st.number_input(
            "Vitesse du vent (m/s)", min_value=0.0, max_value=30.0, value=2.5, step=0.1,
            help=f"Plage observée dans les données : {w_min:.1f} – {w_max:.1f} m/s",
        )

    out_of_range = []
    if not (t_min <= temperature <= t_max):
        out_of_range.append(f"température ({t_min:.1f} – {t_max:.1f} °C)")
    if not (w_min <= wind_speed <= w_max):
        out_of_range.append(f"vitesse du vent ({w_min:.1f} – {w_max:.1f} m/s)")
    if out_of_range:
        st.warning(
            "Valeur(s) hors de la plage vue à l'entraînement : "
            + ", ".join(out_of_range)
            + ". L'estimation est une extrapolation, à interpréter avec une grande prudence."
        )

    if st.button("Estimer l'humidité", type="primary"):
        X_new = pd.DataFrame(
            [{
                "region": region,
                "departement": departement,
                "weather": weather_key,
                "temperature": temperature,
                "wind_speed": wind_speed,
            }]
        )[meta["features"]]
        raw = float(pipe.predict(X_new)[0])
        pred = min(max(raw, 0.0), 100.0)  # l'humidité relative est bornée à [0, 100]

        c1, c2 = st.columns([1, 2])
        c1.metric("Humidité estimée", f"{pred:.0f} %")
        c2.markdown(
            f"**{interpret(pred)}.**  \n"
            f"Ordre de grandeur de l'erreur : environ ± {typical_mae:.0f} points "
            f"(erreur absolue moyenne en validation par date ; RMSE ≈ {typical_rmse:.1f}). "
            f"L'erreur réelle peut être plus grande selon la région et la journée."
        )
        if raw != pred:
            st.info(f"Estimation brute du modèle ({raw:.1f} %) ramenée dans l'intervalle [0, 100].")

    st.info(
        "**À retenir :** il s'agit d'une *estimation à partir de mesures fournies*, "
        "pas d'une mesure météorologique réelle ni d'une prévision du temps futur."
    )

# --------------------------------- Données --------------------------------- #
with tab_data:
    if df is None:
        st.info(
            "Le fichier de données n'est pas présent dans ce dépôt (`data/`). "
            "Les visualisations descriptives sont donc indisponibles."
        )
    else:
        st.subheader("Aperçu descriptif du jeu de données")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Observations", len(df))
        m2.metric("Régions", df["region"].nunique())
        m3.metric("Départements", df["departement"].nunique())
        m4.metric("Dates distinctes", df["date"].dt.normalize().nunique())

        st.markdown("**Température et humidité** (corrélation observée : "
                    f"{df['temperature'].corr(df['humidity']):.2f} — un constat descriptif, pas une preuve de causalité)")
        st.scatter_chart(df, x="temperature", y="humidity", height=350)

        st.markdown("**Humidité moyenne par région**")
        st.bar_chart(df.groupby("region")["humidity"].mean().sort_values(), horizontal=True)

        st.markdown("**Humidité moyenne par date**")
        by_date = df.assign(jour=df["date"].dt.date.astype(str)).groupby("jour")["humidity"].mean()
        st.bar_chart(by_date)

# ---------------------------- Modèle et limites ----------------------------- #
with tab_model:
    st.subheader("Modèle retenu")
    st.markdown(
        f"**{meta['model_name']}** (pipeline scikit-learn : `StandardScaler` + `OneHotEncoder` + modèle), "
        f"variables : {', '.join(meta['features'])}."
    )
    st.markdown(
        "Le modèle a été choisi par **validation « une date laissée de côté »** : "
        "chaque date est tour à tour le jeu de test, le modèle étant entraîné sur les quatre autres."
    )
    v1, v2, v3 = st.columns(3)
    v1.metric("MAE moyenne", f"{lodo['MAE_mean']:.2f}")
    v2.metric("RMSE moyen", f"{lodo['RMSE_mean']:.2f}")
    v3.metric("R² moyen", f"{lodo['R2_mean']:.2f}")

    for name, caption in [
        ("rmse_comparison.png", "RMSE selon le protocole de validation"),
        ("actual_vs_predicted.png", "Humidité réelle vs prédite (validation par date)"),
        ("permutation_importance.png", "Importance par permutation — une association, pas une cause"),
    ]:
        if (FIG_DIR / name).exists():
            st.image(str(FIG_DIR / name), caption=caption)

    st.subheader("Limites")
    st.markdown(
        "- Seulement **5 journées** d'octobre 2025 : aucune information sur les autres saisons ou années.\n"
        "- La validation par date repose sur **5 plis seulement** : les écarts entre modèles proches "
        "(ex. Ridge et régression linéaire) ne sont pas significatifs.\n"
        "- Les performances élevées s'expliquent en grande partie par le lien température–humidité.\n"
        "- Ce n'est **pas** un système de prévision météorologique.\n"
        "- Régions et départements hors du jeu d'entraînement ne sont pas pris en charge."
    )
