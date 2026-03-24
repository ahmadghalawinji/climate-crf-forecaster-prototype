import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

PROC   = Path("data/processed")
MODELS = Path("models")
OUT    = Path("outputs")
OUT.mkdir(exist_ok=True)

FEATURES = [
    "avg_temp",
    "heat_stress",
    "avg_precip_mm",
    "pa_policy_score",
    "gdp_per_capita",
    "urban_pop_pct",
]

def load_model(name="random_forest"):
    return joblib.load(MODELS / f"{name}.pkl")

def simulate_climate_scenarios(df: pd.DataFrame, model) -> pd.DataFrame:
    scenarios = {"Baseline": 0, "+1°C": 1, "+2°C": 2, "+3°C": 3}
    records = []
    for label, delta_t in scenarios.items():
        X = df[FEATURES].copy()
        X["avg_temp"]   = X["avg_temp"]   + delta_t
        X["heat_stress"] = (X["heat_stress"] + delta_t * 0.05).clip(0, 1)
        pred = model.predict(X)
        records.append({
            "scenario":      label,
            "mean_vo2max":   round(pred.mean(), 2),
            "median_vo2max": round(float(np.median(pred)), 2),
        })
    return pd.DataFrame(records)


def simulate_mitigation(df: pd.DataFrame, model,
                        temp_increase: float = 2.0,
                        pa_boost: float = 0.20) -> pd.DataFrame:
    baseline_pred = model.predict(df[FEATURES]).mean()
    scenarios = {
        f"+{temp_increase}°C, current PA":          0.0,
        f"+{temp_increase}°C, PA +{int(pa_boost*100)}%": pa_boost,
    }
    records = []
    for label, pa_delta in scenarios.items():
        X = df[FEATURES].copy()
        X["avg_temp"]    = X["avg_temp"]    + temp_increase
        X["heat_stress"] = (X["heat_stress"] + temp_increase * 0.05).clip(0, 1)
        X["pa_policy_score"] = (X["pa_policy_score"] * (1 + pa_delta))
        pred = model.predict(X)
        records.append({
            "scenario":          label,
            "mean_vo2max":       round(pred.mean(), 2),
            "delta_vs_baseline": round(pred.mean() - baseline_pred, 2),
        })
    return pd.DataFrame(records)

def plot_climate_scenarios(results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#1d9e75", "#fac775", "#ef9f27", "#d85a30"]
    bars = ax.bar(results["scenario"], results["mean_vo2max"],
                  color=colors, width=0.5, edgecolor="none")
    ax.set_ylabel("Mean predicted VO₂max (ml/kg/min)", fontsize=11)
    ax.set_title("Predicted youth CRF under climate scenarios", fontsize=12)
    ax.set_ylim(results["mean_vo2max"].min() - 2, results["mean_vo2max"].max() + 2)
    for bar, val in zip(bars, results["mean_vo2max"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / "climate_scenarios.png", dpi=150)
    plt.show()

def plot_mitigation(results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#ef9f27", "#1d9e75"]
    bars = ax.bar(results["scenario"], results["mean_vo2max"],
                  color=colors, width=0.4, edgecolor="none")
    ax.set_ylabel("Mean predicted VO₂max (ml/kg/min)", fontsize=11)
    ax.set_title("Physical activity as climate mitigation", fontsize=12)
    ax.set_ylim(results["mean_vo2max"].min() - 2, results["mean_vo2max"].max() + 2)
    for bar, val in zip(bars, results["mean_vo2max"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / "mitigation_analysis.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv(PROC / "dataset_with_crf.csv").dropna(subset=FEATURES)
    model = load_model("random_forest")

    climate_results = simulate_climate_scenarios(df, model)
    print("\nClimate scenarios:")
    print(climate_results)
    plot_climate_scenarios(climate_results)

    mitigation_results = simulate_mitigation(df, model)
    print("\nMitigation analysis:")
    print(mitigation_results)
    plot_mitigation(mitigation_results)