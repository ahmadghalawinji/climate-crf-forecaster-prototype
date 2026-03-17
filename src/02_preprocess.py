"""
Preprocessing — precise column handling for both ERA5 files:

  avgad file: valid_time, latitude, longitude, tp, number, expver
              tp = total precipitation (metres per timestep)

  avgua file: valid_time, latitude, longitude, d2m, t2m, number, expver
              t2m  = 2-metre air temperature (Kelvin)
              d2m  = 2-metre dewpoint temperature (Kelvin)

  Derived variables:
    avg_temp_c   = mean annual 2m air temperature (°C)
    avg_precip_mm= mean annual total precipitation (mm)
    rh_pct       = relative humidity (%) via Magnus formula
    heat_stress  = WBGT proxy (0–1 normalised), physiologically meaningful
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW  = Path("data/raw")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

# ── Country bounding boxes (lat_min, lat_max, lon_min, lon_max) ───────────────
# Covers all HEPA-reporting European countries
COUNTRY_BOXES = {
    "Albania":                   (39.6, 42.7,  19.3, 21.1),
    "Austria":                   (46.4, 49.0,   9.5, 17.2),
    "Belarus":                   (51.2, 56.2,  23.2, 32.8),
    "Belgium":                   (49.5, 51.5,   2.5,  6.4),
    "Bosnia and Herzegovina":    (42.6, 45.3,  15.7, 19.6),
    "Bulgaria":                  (41.2, 44.2,  22.4, 28.6),
    "Croatia":                   (42.4, 46.6,  13.5, 19.4),
    "Czechia":                   (48.5, 51.1,  12.1, 18.9),
    "Denmark":                   (54.6, 57.8,   8.1, 15.2),
    "Estonia":                   (57.5, 59.7,  21.8, 28.2),
    "Finland":                   (59.8, 70.1,  20.0, 31.6),
    "France":                    (41.3, 51.1,  -5.1,  9.6),
    "Georgia":                   (41.0, 43.6,  40.0, 46.7),
    "Germany":                   (47.3, 55.1,   5.9, 15.0),
    "Greece":                    (35.0, 42.0,  20.0, 28.2),
    "Hungary":                   (45.7, 48.6,  16.1, 22.9),
    "Iceland":                   (63.4, 66.5, -24.5,-13.5),
    "Ireland":                   (51.4, 55.4, -10.5, -6.0),
    "Italy":                     (36.6, 47.1,   6.6, 18.5),
    "Latvia":                    (55.7, 58.1,  21.0, 28.2),
    "Lithuania":                 (53.9, 56.5,  21.0, 26.8),
    "Luxembourg":                (49.4, 50.2,   5.7,  6.5),
    "Moldova":                   (45.5, 48.5,  26.6, 30.1),
    "Montenegro":                (41.8, 43.6,  18.4, 20.4),
    "Netherlands":               (50.8, 53.6,   3.4,  7.2),
    "North Macedonia":           (41.1, 42.4,  20.4, 23.0),
    "Norway":                    (57.9, 71.2,   4.5, 31.1),
    "Poland":                    (49.0, 54.9,  14.1, 24.2),
    "Portugal":                  (37.0, 42.2,  -9.5, -6.2),
    "Romania":                   (43.6, 48.3,  22.0, 30.0),
    "Russia":                    (50.0, 70.0,  30.0, 60.0),
    "Serbia":                    (42.2, 46.2,  18.8, 23.0),
    "Slovakia":                  (47.7, 49.6,  16.8, 22.6),
    "Slovenia":                  (45.4, 46.9,  13.4, 16.6),
    "Spain":                     (36.0, 43.8,  -9.3,  4.3),
    "Sweden":                    (55.3, 69.1,  11.1, 24.2),
    "Switzerland":               (45.8, 47.8,   5.9, 10.5),
    "Turkey":                    (36.0, 42.1,  26.0, 45.0),
    "Ukraine":                   (44.4, 52.4,  22.1, 40.2),
    "United Kingdom":            (49.9, 58.7,  -8.2,  2.0),
}


# ── Physics helpers ────────────────────────────────────────────────────────────

def kelvin_to_celsius(k: pd.Series) -> pd.Series:
    return k - 273.15


def relative_humidity(t_c: pd.Series, td_c: pd.Series) -> pd.Series:
    """
    Relative humidity (%) using the Magnus approximation.
    RH = 100 × exp(17.625 × Td / (243.04 + Td))
             / exp(17.625 × T  / (243.04 + T ))
    Valid for T and Td in °C.
    Result clamped to [0, 100].
    """
    num = np.exp(17.625 * td_c / (243.04 + td_c))
    den = np.exp(17.625 * t_c  / (243.04 + t_c))
    rh  = 100.0 * num / den
    return rh.clip(0, 100)


def wbgt_proxy(t_c: pd.Series, rh: pd.Series) -> pd.Series:
    """
    Simplified indoor WBGT proxy (Liljegren approximation, no solar load):
        WBGT ≈ 0.567 × T + 0.393 × e + 3.94
    where e = vapour pressure (hPa) = RH/100 × 6.105 × exp(17.27×T/(237.7+T))
    Returns WBGT in °C.
    Reference: Bernard & Pourmoghani (1999), used widely in exercise physiology.
    """
    e = (rh / 100.0) * 6.105 * np.exp(17.27 * t_c / (237.7 + t_c))
    wbgt = 0.567 * t_c + 0.393 * e + 3.94
    return wbgt


def normalise_01(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.0, index=series.index)
    return (series - mn) / (mx - mn)


# ── ERA5 spatial aggregation ───────────────────────────────────────────────────

def _aggregate_era5(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """
    Clip a raw ERA5 grid DataFrame to each country's bounding box
    and compute yearly means (and max for WBGT-related cols).
    Returns a long DataFrame: country, year, + aggregated columns.
    """
    records = []
    for country, (lat_min, lat_max, lon_min, lon_max) in COUNTRY_BOXES.items():
        mask = (
            (df["latitude"]  >= lat_min) & (df["latitude"]  <= lat_max) &
            (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max)
        )
        subset = df.loc[mask]
        if subset.empty:
            continue
        for year, grp in subset.groupby("year"):
            row = {"country": country, "year": int(year)}
            for col in value_cols:
                row[f"{col}_mean"] = grp[col].mean()
                row[f"{col}_max"]  = grp[col].max()
            records.append(row)
    return pd.DataFrame(records)


# ── File-specific processors ───────────────────────────────────────────────────

def process_era5_precip(filename: str = "data_stream-moda_stepType-avgad.csv") -> pd.DataFrame:
    """
    avgad file — columns: valid_time, latitude, longitude, tp, number, expver
    tp = total precipitation in metres per timestep.
    Outputs: country, year, avg_precip_mm (mean × 1000 to convert m → mm).
    """
    print(f"\nLoading ERA5 precipitation: {filename}")
    df = pd.read_csv(RAW / filename)
    print(f"  Shape: {df.shape}  |  Columns: {list(df.columns)}")

    df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce")
    df = df.dropna(subset=["valid_time", "latitude", "longitude", "tp"])
    df["year"] = df["valid_time"].dt.year

    # tp is in metres → convert to mm
    df["tp_mm"] = df["tp"] * 1000.0

    agg = _aggregate_era5(df, ["tp_mm"])

    # Keep only the annual mean precipitation (mm)
    result = agg[["country", "year", "tp_mm_mean"]].rename(
        columns={"tp_mm_mean": "avg_precip_mm"}
    )
    print(f"  → Aggregated: {result.shape}")
    return result


def process_era5_temp(filename: str = "data_stream-moda_stepType-avgua.csv") -> pd.DataFrame:
    """
    avgua file — columns: valid_time, latitude, longitude, d2m, t2m, number, expver
    t2m = 2m air temperature (Kelvin)
    d2m = 2m dewpoint temperature (Kelvin)

    Derived outputs per country-year:
      avg_temp_c    — mean annual 2m temperature (°C)
      max_temp_c    — warmest timestep mean (°C), useful for heat extremes
      avg_rh_pct    — mean relative humidity (%)
      avg_wbgt      — mean WBGT proxy (°C), physiological heat stress
      max_wbgt      — peak WBGT proxy (°C)
      heat_stress   — normalised [0–1] heat stress from avg_wbgt
    """
    print(f"\nLoading ERA5 temperature: {filename}")
    df = pd.read_csv(RAW / filename)
    print(f"  Shape: {df.shape}  |  Columns: {list(df.columns)}")

    # Parse datetime
    df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce")
    df = df.dropna(subset=["valid_time", "latitude", "longitude", "t2m", "d2m"])
    df["year"] = df["valid_time"].dt.year

    # Convert Kelvin → Celsius
    df["t2m_c"] = kelvin_to_celsius(df["t2m"])
    df["d2m_c"] = kelvin_to_celsius(df["d2m"])

    # Sanity check: dewpoint must be ≤ air temperature
    invalid = (df["d2m_c"] > df["t2m_c"] + 0.5).sum()
    if invalid > 0:
        print(f"  [Warning] {invalid} rows where d2m > t2m — clamping dewpoint")
        df["d2m_c"] = df[["d2m_c", "t2m_c"]].min(axis=1)

    # Derived variables
    df["rh_pct"] = relative_humidity(df["t2m_c"], df["d2m_c"])
    df["wbgt"]   = wbgt_proxy(df["t2m_c"], df["rh_pct"])

    print(f"  t2m_c range : {df['t2m_c'].min():.1f} – {df['t2m_c'].max():.1f} °C")
    print(f"  rh_pct range: {df['rh_pct'].min():.1f} – {df['rh_pct'].max():.1f} %")
    print(f"  wbgt range  : {df['wbgt'].min():.1f} – {df['wbgt'].max():.1f} °C")

    # Spatial + temporal aggregation
    agg = _aggregate_era5(df, ["t2m_c", "rh_pct", "wbgt"])

    result = agg[[
        "country", "year",
        "t2m_c_mean", "t2m_c_max",
        "rh_pct_mean",
        "wbgt_mean", "wbgt_max",
    ]].rename(columns={
        "t2m_c_mean": "avg_temp_c",
        "t2m_c_max":  "max_temp_c",
        "rh_pct_mean":"avg_rh_pct",
        "wbgt_mean":  "avg_wbgt",
        "wbgt_max":   "max_wbgt",
    })

    # Normalise WBGT to [0, 1] heat_stress feature
    # WBGT thresholds from exercise physiology:
    #   < 18°C  → no risk
    #   18–23°C → low risk
    #   23–28°C → moderate risk
    #   > 28°C  → high risk (exercise restrictions)
    result["heat_stress"] = ((result["avg_wbgt"] - 10) / 25).clip(0, 1)

    print(f"  → Aggregated: {result.shape}")
    print(result.describe().round(2))
    return result


def process_worldbank(filename: str, value_col: str) -> pd.DataFrame:
    """Standard World Bank wide-format CSV → long (country, year, value)."""
    df = pd.read_csv(RAW / filename, skiprows=4)
    df = df.rename(columns={"Country Name": "country", "Country Code": "code"})
    year_cols = [c for c in df.columns if str(c).isdigit()]
    long = df[["country", "code"] + year_cols].melt(
        id_vars=["country", "code"], var_name="year", value_name=value_col
    )
    long["year"] = long["year"].astype(int)
    long[value_col] = pd.to_numeric(long[value_col], errors="coerce")
    long = long.dropna(subset=[value_col])
    print(f"WorldBank {value_col}: {long.shape}")
    return long[["country", "year", value_col]]


def process_hepa() -> pd.DataFrame:
    """
    HEPA table data:
      Measure code, YES_NO, COUNTRY_REGION (ISO3), YEAR, VALUE
    Returns pivoted DataFrame with one row per country-year
    and a pa_policy_score column.
    """
    for fname in ("HEPA Data (table).csv", "HEPA Data (pivoted).csv"):
        path = RAW / fname
        if path.exists():
            df = pd.read_csv(path)
            print(f"\nLoaded HEPA from: {fname}")
            break
    else:
        raise FileNotFoundError("No HEPA file found in data/raw/")

    # Normalise column names
    df.columns = [
        c.strip().lstrip('\ufeff').replace(' ', '_').upper()
        for c in df.columns
    ]
    print(f"  Columns after normalisation: {list(df.columns)}")

    # Rename to standard names
    col_map = {}
    for col in df.columns:
        if "MEASURE" in col:        col_map[col] = "measure"
        elif col in ("YES_NO",):    col_map[col] = "yes_no"
        elif "COUNTRY" in col or "REGION" in col: col_map[col] = "country_iso3"
        elif "YEAR" in col:         col_map[col] = "year"
        elif "VALUE" in col:        col_map[col] = "value"
    df = df.rename(columns=col_map)

    from src.utils import iso3_to_country, normalize_country_name
    df["country"] = df["country_iso3"].apply(iso3_to_country).apply(normalize_country_name)
    df["year"]    = pd.to_numeric(df["year"],  errors="coerce")
    df["value"]   = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["country", "year"])

    # Pivot: one column per measure × yes_no
    df["feature"] = df["measure"] + "_" + df.get("yes_no", pd.Series("", index=df.index)).str.upper()
    pivoted = df.pivot_table(
        index=["country", "year"],
        columns="feature",
        values="value",
        aggfunc="first"
    ).reset_index()
    pivoted.columns.name = None

    # Composite score: count of YES policy indicators
    yes_cols = [c for c in pivoted.columns if c.endswith("_YES")]
    pivoted["pa_policy_score"] = pivoted[yes_cols].sum(axis=1).fillna(0) if yes_cols else 0
    print(f"  → HEPA pivoted: {pivoted.shape}, YES columns: {yes_cols}")
    return pivoted


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ERA5 — precipitation
    precip = process_era5_precip("data_stream-moda_stepType-avgad.csv")
    precip.to_csv(PROC / "era5_precip.csv", index=False)

    # ERA5 — temperature + humidity + WBGT heat stress
    temp = process_era5_temp("data_stream-moda_stepType-avgua.csv")
    temp.to_csv(PROC / "era5_temp.csv", index=False)

    # Merge both ERA5 files
    era5 = temp.merge(precip, on=["country", "year"], how="outer")
    era5.to_csv(PROC / "era5_yearly.csv", index=False)
    print(f"\nMerged ERA5: {era5.shape}")
    print(era5.head(3).to_string())

    # World Bank
    gdp   = process_worldbank(
        "API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_35.csv", "gdp_per_capita"
    )
    urban = process_worldbank(
        "API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_249.csv", "urban_pop_pct"
    )
    gdp.to_csv(PROC   / "gdp_clean.csv",   index=False)
    urban.to_csv(PROC / "urban_clean.csv", index=False)

    # HEPA
    hepa = process_hepa()
    hepa.to_csv(PROC / "hepa_clean.csv", index=False)

    print("\nAll preprocessing complete. Files in data/processed/:")
    for f in sorted(PROC.iterdir()):
        print(f"  {f.name}")