"""
Prédiction de l'humidité au Sénégal — script d'entraînement et de validation.

Ce script :
  1. charge et nettoie les données (suppression des doublons exacts) ;
  2. construit des pipelines scikit-learn (prétraitement + modèle) ;
  3. compare plusieurs modèles avec trois protocoles de validation :
       - séparation aléatoire 80/20 (référence "optimiste") ;
       - séparation temporelle (test sur le 26 octobre 2025) ;
       - validation "une date laissée de côté" (leave-one-date-out) ;
  4. choisit le modèle sur la base de la validation par date (jamais sur un
     jeu de test unique) ;
  5. ré-entraîne le pipeline retenu sur toutes les données et le sauvegarde ;
  6. produit les graphiques et tableaux de résultats.

Usage :
    python train.py
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # pas d'affichage : on enregistre les figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "meteo_departements_Senegal.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
for d in (MODELS_DIR, REPORTS_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET = "humidity"
CAT_FEATURES = ["region", "departement", "weather"]
NUM_FEATURES_WITH_HOUR = ["temperature", "wind_speed", "hour"]
NUM_FEATURES_NO_HOUR = ["temperature", "wind_speed"]
TEST_DATE = pd.Timestamp("2025-10-26")

# Notes de conception :
# - `month` est constant (tout est en octobre) -> aucune information, exclu.
# - `day` identifie la date de collecte : il ne peut pas se généraliser à une
#   date jamais vue (et fuit l'identité de la date en validation aléatoire),
#   donc exclu du modèle final.
# - `hour` est testé : avec et sans.


# --------------------------------------------------------------------------- #
# Données
# --------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Lignes brutes : {n_raw} | après suppression des doublons : {len(df)}")
    df["hour"] = df["date"].dt.hour
    df["obs_date"] = df["date"].dt.normalize()
    return df


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #
def make_preprocessor(num_features, scale=True):
    """Prétraitement ajusté uniquement sur l'entraînement (via le Pipeline)."""
    num_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("num", num_step, num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ]
    )


def make_models():
    """Hyperparamètres fixés a priori (pas de réglage sur le test)."""
    return {
        "Baseline (moyenne)": (DummyRegressor(strategy="mean"), True),
        "Régression linéaire": (LinearRegression(), True),
        "Ridge": (Ridge(alpha=1.0), True),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1
            ),
            False,
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED
            ),
            False,
        ),
        "XGBoost": (
            XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED,
                n_jobs=-1,
                verbosity=0,
            ),
            False,
        ),
    }


def build_pipeline(model, scale, num_features):
    return Pipeline(
        [("prep", make_preprocessor(num_features, scale=scale)), ("model", model)]
    )


def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# Protocoles de validation
# --------------------------------------------------------------------------- #
def eval_random(df, num_features):
    feats = CAT_FEATURES + num_features
    X, y = df[feats], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    rows = []
    for name, (model, scale) in make_models().items():
        pipe = build_pipeline(model, scale, num_features).fit(X_tr, y_tr)
        rows.append({"model": name, **metrics(y_te, pipe.predict(X_te))})
    return pd.DataFrame(rows)


def eval_temporal(df, num_features):
    feats = CAT_FEATURES + num_features
    tr = df[df["obs_date"] < TEST_DATE]
    te = df[df["obs_date"] == TEST_DATE]
    rows = []
    for name, (model, scale) in make_models().items():
        pipe = build_pipeline(model, scale, num_features).fit(tr[feats], tr[TARGET])
        rows.append({"model": name, **metrics(te[TARGET], pipe.predict(te[feats]))})
    return pd.DataFrame(rows)


def eval_leave_one_date_out(df, num_features, keep_predictions_for=None):
    """Chaque date est tour à tour le jeu de test ; on entraîne sur les 4 autres."""
    feats = CAT_FEATURES + num_features
    X, y = df[feats], df[TARGET]
    groups = df["obs_date"]
    logo = LeaveOneGroupOut()
    fold_rows, oof = [], None
    for name, (model, scale) in make_models().items():
        preds = pd.Series(index=df.index, dtype=float)
        for tr_idx, te_idx in logo.split(X, y, groups):
            pipe = build_pipeline(model, scale, num_features).fit(
                X.iloc[tr_idx], y.iloc[tr_idx]
            )
            preds.iloc[te_idx] = pipe.predict(X.iloc[te_idx])
            held_out = groups.iloc[te_idx].iloc[0].date()
            fold_rows.append(
                {
                    "model": name,
                    "held_out_date": str(held_out),
                    **metrics(y.iloc[te_idx], preds.iloc[te_idx]),
                }
            )
        if name == keep_predictions_for:
            oof = preds.copy()
    return pd.DataFrame(fold_rows), oof


def summarize_folds(folds: pd.DataFrame) -> pd.DataFrame:
    return (
        folds.groupby("model")[["MAE", "RMSE", "R2"]]
        .agg(["mean", "std"])
        .pipe(lambda t: t.set_axis([f"{a}_{b}" for a, b in t.columns], axis=1))
        .sort_values("RMSE_mean")
        .reset_index()
    )


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_rmse_comparison(rand_df, temp_df, logo_summary):
    names = list(logo_summary["model"])
    x = np.arange(len(names))
    w = 0.27
    r = rand_df.set_index("model").loc[names, "RMSE"]
    t = temp_df.set_index("model").loc[names, "RMSE"]
    l = logo_summary.set_index("model").loc[names, "RMSE_mean"]
    lerr = logo_summary.set_index("model").loc[names, "RMSE_std"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, r, w, label="Aléatoire 80/20", color="#9ecae1")
    ax.bar(x, t, w, label="Temporelle (test 26 oct.)", color="#3182bd")
    ax.bar(x + w, l, w, yerr=lerr, capsize=3, label="Une date laissée de côté (moy. ± écart-type)", color="#08519c")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("RMSE (points de % d'humidité)")
    ax.set_title("RMSE selon le protocole de validation (plus bas = mieux)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rmse_comparison.png", dpi=150)
    plt.close(fig)


def plot_actual_vs_pred(y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=14, alpha=0.6, color="#3182bd")
    lim = [min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2]
    ax.plot(lim, lim, "--", color="grey", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Humidité réelle (%)")
    ax.set_ylabel("Humidité prédite (%)")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(y_true, y_pred, title, path):
    res = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_pred, res, s=14, alpha=0.6, color="#3182bd")
    axes[0].axhline(0, color="grey", ls="--", lw=1)
    axes[0].set_xlabel("Humidité prédite (%)")
    axes[0].set_ylabel("Résidu (réel − prédit)")
    axes[0].set_title("Résidus vs prédictions")
    axes[1].hist(res, bins=30, color="#9ecae1", edgecolor="white")
    axes[1].set_xlabel("Résidu (réel − prédit)")
    axes[1].set_title("Distribution des résidus")
    fig.suptitle(title)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_by_region(df, y_pred, path):
    tmp = df[["region", TARGET]].copy()
    tmp["abs_err"] = (df[TARGET] - y_pred).abs()
    by_reg = tmp.groupby("region")["abs_err"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(by_reg.index, by_reg.values, color="#3182bd")
    ax.set_xlabel("MAE (points de %)")
    ax.set_title("Erreur absolue moyenne par région (prédictions hors échantillon)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    by_reg.rename("MAE").to_csv(REPORTS_DIR / "mae_by_region.csv")


def plot_permutation_importance(imp: pd.DataFrame, path):
    imp = imp.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp["feature"], imp["importance_mean"], xerr=imp["importance_std"],
            color="#3182bd", capsize=3)
    ax.set_xlabel("Hausse de RMSE quand la variable est mélangée")
    ax.set_title("Importance par permutation (test : 26 octobre)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Programme principal
# --------------------------------------------------------------------------- #
def main():
    df = load_data()
    print("Dates distinctes :", sorted(df["obs_date"].dt.date.unique()))

    # ---- 1. Quelles variables numériques ? (avec / sans heure) -------------
    results_by_featureset = {}
    for label, num_feats in [("avec_heure", NUM_FEATURES_WITH_HOUR),
                             ("sans_heure", NUM_FEATURES_NO_HOUR)]:
        rand_df = eval_random(df, num_feats)
        temp_df = eval_temporal(df, num_feats)
        folds, _ = eval_leave_one_date_out(df, num_feats)
        summary = summarize_folds(folds)
        results_by_featureset[label] = (rand_df, temp_df, folds, summary)
        print(f"\n===== Jeu de variables : {label} =====")
        print("\n[Aléatoire 80/20]\n", rand_df.round(3).to_string(index=False))
        print("\n[Temporelle - test 26 oct.]\n", temp_df.round(3).to_string(index=False))
        print("\n[Une date laissée de côté]\n", summary.round(3).to_string(index=False))

    # ---- 2. Sélection (sur la validation par date, sans toucher à un test final)
    best_choice = None  # (rmse_mean, featureset, model_name)
    for label, (_, _, _, summary) in results_by_featureset.items():
        # on exclut la baseline de la sélection
        cand = summary[summary["model"] != "Baseline (moyenne)"].iloc[0]
        if best_choice is None or cand["RMSE_mean"] < best_choice[0]:
            best_choice = (cand["RMSE_mean"], label, cand["model"])
    _, best_set, best_name = best_choice
    num_feats = NUM_FEATURES_WITH_HOUR if best_set == "avec_heure" else NUM_FEATURES_NO_HOUR
    print(f"\n>>> Sélection : {best_name} ({best_set}) — meilleur RMSE moyen en validation par date")

    rand_df, temp_df, folds, summary = results_by_featureset[best_set]
    rand_df.to_csv(REPORTS_DIR / "results_random_split.csv", index=False)
    temp_df.to_csv(REPORTS_DIR / "results_temporal_split.csv", index=False)
    folds.to_csv(REPORTS_DIR / "results_leave_one_date_out_folds.csv", index=False)
    summary.to_csv(REPORTS_DIR / "results_leave_one_date_out_summary.csv", index=False)
    for label, (r, t, f, s) in results_by_featureset.items():
        s.assign(feature_set=label).to_csv(
            REPORTS_DIR / f"summary_lodo_{label}.csv", index=False
        )

    # ---- 3. Figures et analyse d'erreurs pour le modèle retenu ---------------
    plot_rmse_comparison(rand_df, temp_df, summary)
    _, oof = eval_leave_one_date_out(df, num_feats, keep_predictions_for=best_name)
    plot_actual_vs_pred(df[TARGET], oof, f"{best_name} — réel vs prédit (validation par date)",
                        FIG_DIR / "actual_vs_predicted.png")
    plot_residuals(df[TARGET], oof, f"{best_name} — analyse des résidus (validation par date)",
                   FIG_DIR / "residuals.png")
    plot_error_by_region(df, oof, FIG_DIR / "error_by_region.png")

    # ---- 4. Diagnostics de fuite / dépendance à la température --------------
    # a) un modèle qui n'utilise QUE la température
    temp_only = Pipeline([("prep", ColumnTransformer([("num", StandardScaler(), ["temperature"])])),
                          ("model", LinearRegression())])
    oof_t = pd.Series(index=df.index, dtype=float)
    for tr_idx, te_idx in LeaveOneGroupOut().split(df, df[TARGET], df["obs_date"]):
        temp_only.fit(df.iloc[tr_idx][["temperature"]], df.iloc[tr_idx][TARGET])
        oof_t.iloc[te_idx] = temp_only.predict(df.iloc[te_idx][["temperature"]])
    temp_only_metrics = metrics(df[TARGET], oof_t)
    print("\n[Diagnostic] Régression sur la température seule (validation par date) :",
          {k: round(v, 3) for k, v in temp_only_metrics.items()})

    # b) importance par permutation sur la séparation temporelle
    feats = CAT_FEATURES + num_feats
    model, scale = make_models()[best_name]
    tr, te = df[df["obs_date"] < TEST_DATE], df[df["obs_date"] == TEST_DATE]
    pipe_tmp = build_pipeline(model, scale, num_feats).fit(tr[feats], tr[TARGET])
    pi = permutation_importance(pipe_tmp, te[feats], te[TARGET], n_repeats=30,
                                random_state=SEED, scoring="neg_root_mean_squared_error")
    imp = pd.DataFrame({"feature": feats, "importance_mean": pi.importances_mean,
                        "importance_std": pi.importances_std})
    imp.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)
    plot_permutation_importance(imp, FIG_DIR / "permutation_importance.png")
    print("\n[Importance par permutation]\n", imp.sort_values("importance_mean", ascending=False).round(3).to_string(index=False))

    # ---- 5. Ré-entraînement final sur TOUTES les données et sauvegarde ------
    model, scale = make_models()[best_name]
    final_pipe = build_pipeline(model, scale, num_feats).fit(df[feats], df[TARGET])
    joblib.dump(final_pipe, MODELS_DIR / "humidity_pipeline.joblib")

    region_to_dep = (df.groupby("region")["departement"]
                       .apply(lambda s: sorted(s.unique())).to_dict())
    metadata = {
        "model_name": best_name,
        "feature_set": best_set,
        "features": feats,
        "numeric_features": num_feats,
        "categorical_features": CAT_FEATURES,
        "target": TARGET,
        "region_to_departements": region_to_dep,
        "weather_categories": sorted(df["weather"].unique()),
        "ranges": {
            "temperature": [float(df.temperature.min()), float(df.temperature.max())],
            "wind_speed": [float(df.wind_speed.min()), float(df.wind_speed.max())],
            "hour": [int(df.hour.min()), int(df.hour.max())],
            "humidity": [float(df.humidity.min()), float(df.humidity.max())],
        },
        "observed_hours": sorted(int(h) for h in df.hour.unique()),
        "n_observations": int(len(df)),
        "n_dates": int(df.obs_date.nunique()),
        "dates": [str(d.date()) for d in sorted(df.obs_date.unique())],
        "validation": {
            "leave_one_date_out": summary.set_index("model").loc[best_name].round(4).to_dict(),
            "temporal_split": temp_df.set_index("model").loc[best_name].round(4).to_dict(),
            "random_split": rand_df.set_index("model").loc[best_name].round(4).to_dict(),
            "temperature_only_linear_lodo": {k: round(v, 4) for k, v in temp_only_metrics.items()},
        },
        "library_versions": {"scikit-learn": sklearn.__version__},
        "note": ("Pipeline ré-entraîné sur l'ensemble des données (5 dates, octobre 2025). "
                 "Estimation d'humidité à partir de mesures fournies, pas une prévision météo."),
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nPipeline sauvegardé : {MODELS_DIR / 'humidity_pipeline.joblib'}")
    print(f"Métadonnées        : {MODELS_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
