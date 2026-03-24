import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

PROC    = Path("data/processed")
MODELS  = Path("models")
OUTPUTS = Path("outputs")
MODELS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

TARGET = "vo2max"

FEATURES = [
    "avg_temp",
    "heat_stress",
    "avg_precip_mm",
    "pa_policy_score",
    "gdp_per_capita",
    "urban_pop_pct",
]


def train():
    df = pd.read_csv(PROC / "dataset_with_crf.csv")

    # Keep only features that exist in the dataset
    features = [f for f in FEATURES if f in df.columns]
    print(f"Features used: {features}")

    df = df.dropna(subset=features + [TARGET])
    print(f"Rows after dropna: {len(df)}")

    X, y = df[features], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    models = {
        "linear_regression": (LinearRegression(),                        True),
        "random_forest":     (RandomForestRegressor(
                                  n_estimators=300, max_depth=6,
                                  min_samples_leaf=2, random_state=42),  False),
        "xgboost":           (XGBRegressor(
                                  n_estimators=300, max_depth=4,
                                  learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, random_state=42,
                                  verbosity=0),                           False),
    }

    results = {}
    for name, (model, use_scaled) in models.items():
        Xtr = X_tr_s if use_scaled else X_train
        Xte = X_te_s if use_scaled else X_test
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        r2   = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results[name] = {"r2": round(r2, 3), "rmse": round(rmse, 3)}
        print(f"  {name:25s}  R²={r2:.3f}  RMSE={rmse:.3f} ml/kg/min")
        joblib.dump(model, MODELS / f"{name}.pkl")

    joblib.dump(scaler,   MODELS / "scaler.pkl")
    joblib.dump(features, MODELS / "features.pkl")

    # Feature importance plot
    rf = models["random_forest"][0]
    imp = pd.Series(rf.feature_importances_, index=features).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    imp.plot.barh(ax=ax, color="#1d9e75", edgecolor="none")
    ax.set_title("Feature importance — Random Forest", fontsize=11)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "feature_importance.png", dpi=150)
    plt.close()
    print(f"\nFeature importance plot saved to outputs/feature_importance.png")

    pd.DataFrame(results).T.to_csv(OUTPUTS / "model_evaluation.csv")
    print(f"Model evaluation saved to outputs/model_evaluation.csv")
    return results, features


if __name__ == "__main__":
    train()