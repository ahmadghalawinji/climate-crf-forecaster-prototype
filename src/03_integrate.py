"""
Integration: merge ERA5 + HEPA + GDP + Urban → single flat dataset.
All sources are aligned on (country, year).
ERA5 uses country names from bounding box dict.
World Bank uses full country names.
HEPA uses ISO3 → converted to country names via utils.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import normalize_country_name

PROC = Path("data/processed")


def integrate() -> pd.DataFrame:
    # ── Load preprocessed files ───────────────────────────────────────────────
    era5  = pd.read_csv(PROC / "era5_yearly.csv")
    hepa  = pd.read_csv(PROC / "hepa_clean.csv")
    gdp   = pd.read_csv(PROC / "gdp_clean.csv")
    urban = pd.read_csv(PROC / "urban_clean.csv")

    # Normalize country names everywhere
    for df in [era5, hepa, gdp, urban]:
        df["country"] = df["country"].apply(normalize_country_name)

    print("ERA5 countries:", sorted(era5["country"].unique()))
    print("HEPA countries:", sorted(hepa["country"].unique()))

    # ── Identify ERA5 climate columns ─────────────────────────────────────────
    era5_numeric = [c for c in era5.columns
                    if c not in ("country", "year")
                    and pd.api.types.is_numeric_dtype(era5[c])]
    print(f"\nERA5 numeric columns: {era5_numeric}")

    # Build a clean climate summary
    # Rename the first mean temperature column found as avg_temp
    # and the first precipitation column as avg_precip
    era5_clean = era5[["country", "year"]].copy()

    temp_cols   = [c for c in era5_numeric if "t2m" in c or "temp" in c]
    precip_cols = [c for c in era5_numeric if "tp" in c or "precip" in c]

    if temp_cols:
        era5_clean["avg_temp"] = era5[temp_cols[0]]
        # Heat stress: proportion of months where mean temp > 25°C
        # Proxy: (avg_temp - 15) / 20, clipped 0–1
        era5_clean["heat_stress"] = ((era5[temp_cols[0]] - 15) / 20).clip(0, 1)
    else:
        # ERA5 files contain only precipitation — generate temperature proxy
        # from latitude-derived climatology (used only if no t2m available)
        print("  [Warning] No temperature column found in ERA5. "
              "Using latitude-based proxy.")
        # We'll fill avg_temp after merge using a country lookup
        era5_clean["avg_temp"]   = np.nan
        era5_clean["heat_stress"] = np.nan

    if precip_cols:
        # tp in ERA5 is in metres per timestep — convert to mm/year proxy
        era5_clean["avg_precip_mm"] = era5[precip_cols[0]] * 1000

    era5_clean = era5_clean.dropna(subset=["country", "year"])

    # ── Fallback: typical mean annual temperatures per country ────────────────
    # Used only if no ERA5 temperature variable is available
    COUNTRY_TEMP = {
        "Austria": 7.5, "Belgium": 10.5, "Bulgaria": 10.5, "Croatia": 11.5,
        "Czechia": 8.5, "Denmark": 8.0, "Estonia": 5.5, "Finland": 3.0,
        "France": 11.0, "Germany": 9.0, "Greece": 15.5, "Hungary": 10.5,
        "Iceland": 3.0, "Ireland": 9.5, "Italy": 13.5, "Latvia": 6.0,
        "Lithuania": 6.5, "Luxembourg": 9.0, "Netherlands": 10.0, "Norway": 3.5,
        "Poland": 8.5, "Portugal": 15.5, "Romania": 9.5, "Serbia": 11.0,
        "Slovakia": 8.5, "Slovenia": 9.5, "Spain": 14.0, "Sweden": 4.5,
        "Switzerland": 8.5, "Turkey": 12.0, "Ukraine": 8.0, "United Kingdom": 9.5,
        "Albania": 12.5, "Belarus": 6.5, "Bosnia and Herzegovina": 10.0,
        "Georgia": 11.0, "Moldova": 9.5, "Montenegro": 11.5,
        "North Macedonia": 11.0, "Russia": 0.0,
    }
    if era5_clean["avg_temp"].isna().all():
        era5_clean["avg_temp"]    = era5_clean["country"].map(COUNTRY_TEMP)
        era5_clean["heat_stress"] = ((era5_clean["avg_temp"] - 15) / 20).clip(0, 1)

    # ── Merge all datasets ────────────────────────────────────────────────────
    # Start from HEPA (European countries, 2015–2024)
    merged = hepa[["country", "year", "pa_policy_score"]].copy()
    merged = merged.merge(era5_clean, on=["country", "year"], how="left")
    merged = merged.merge(gdp,        on=["country", "year"], how="left")
    merged = merged.merge(urban,      on=["country", "year"], how="left")

    # ── Fill remaining gaps ───────────────────────────────────────────────────
    # Fill numeric columns with per-country medians, then global median
    fill_cols = ["avg_temp", "heat_stress", "gdp_per_capita",
                 "urban_pop_pct", "avg_precip_mm"]
    fill_cols = [c for c in fill_cols if c in merged.columns]

    for col in fill_cols:
        merged[col] = merged.groupby("country")[col].transform(
            lambda x: x.fillna(x.median())
        )
        merged[col] = merged[col].fillna(merged[col].median())

    # Drop rows still missing essential features
    essential = ["pa_policy_score", "avg_temp", "gdp_per_capita"]
    essential = [c for c in essential if c in merged.columns]
    merged = merged.dropna(subset=essential)

    merged = merged.reset_index(drop=True)
    merged.to_csv(PROC / "integrated.csv", index=False)
    print(f"\n── Integrated dataset ──")
    print(f"   Shape: {merged.shape}")
    print(f"   Countries: {merged['country'].nunique()}")
    print(f"   Years: {sorted(merged['year'].unique())}")
    print(f"   Columns: {list(merged.columns)}")
    print(merged.head(5).to_string())
    return merged


if __name__ == "__main__":
    integrate()