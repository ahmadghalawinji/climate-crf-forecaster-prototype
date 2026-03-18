import pandas as pd
import numpy as np
from pathlib import Path

PROC = Path("data/processed")


def norm(series):
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def generate_crf(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    temp_norm  = norm(df.get("avg_temp",        pd.Series(10.0,    index=df.index)))
    wbgt_norm  = norm(df.get("heat_stress",      pd.Series(0.0,    index=df.index)))
    rh_norm    = norm(df.get("avg_rh_pct",       pd.Series(70.0,   index=df.index)))
    pa_norm    = norm(df.get("pa_policy_score",  pd.Series(0.0,    index=df.index)))
    gdp_norm   = norm(df.get("gdp_per_capita",   pd.Series(20000.0, index=df.index)))
    urban_norm = norm(df.get("urban_pop_pct",    pd.Series(60.0,   index=df.index)))

    base_vo2 = (
        42.0
        + 8.0 * pa_norm
        + 3.0 * gdp_norm
        - 3.5 * temp_norm
        - 4.5 * wbgt_norm
        - 1.5 * rh_norm
        - 2.0 * urban_norm
    )

    noise  = rng.normal(0, 2.5, size=len(df))
    vo2max = np.clip(base_vo2 + noise, 30, 60)

    crf_score  = np.interp(vo2max, [30, 60], [25, 92])
    crf_score += rng.normal(0, 1.5, len(df))
    crf_score  = np.clip(crf_score, 10, 100)

    shuttle  = np.interp(vo2max, [30, 60], [22, 105])
    shuttle += rng.normal(0, 2.5, len(df))
    shuttle  = np.clip(shuttle, 10, 120)

    df = df.copy()
    df["vo2max"]            = vo2max.round(2)
    df["crf_score"]         = crf_score.round(2)
    df["shuttle_run_equiv"] = shuttle.round(1)

    print(f"\nCRF generation complete:")
    print(f"  VO2max    — mean: {vo2max.mean():.1f}  std: {vo2max.std():.1f}  "
          f"range: [{vo2max.min():.1f}, {vo2max.max():.1f}]")
    print(f"  CRF score — mean: {crf_score.mean():.1f}  "
          f"range: [{crf_score.min():.1f}, {crf_score.max():.1f}]")
    return df


if __name__ == "__main__":
    df = pd.read_csv(PROC / "integrated.csv")
    df = generate_crf(df)
    df.to_csv(PROC / "dataset_with_crf.csv", index=False)
    print(f"\nSaved dataset_with_crf.csv: {df.shape}")

    show_cols = ["country", "year"]
    for candidate in ["pa_policy_score", "avg_temp", "heat_stress",
                      "vo2max", "crf_score", "shuttle_run_equiv"]:
        if candidate in df.columns:
            show_cols.append(candidate)
    print(df[show_cols].head(10).to_string())