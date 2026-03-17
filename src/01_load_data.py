import pandas as pd
from pathlib import Path

RAW = Path("data/raw")

def inspect_all():
    files = {
        "ERA5 (avgad)": "data_stream-moda_stepType-avgad.csv",
        "ERA5 (avgua)": "data_stream-moda_stepType-avgua.csv",
        "GDP":          "API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_35.csv",
        "Urban pop":    "API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_249.csv",
        "HEPA":         "HEPA Data (pivoted).csv",
    }
    for label, fname in files.items():
        path = RAW / fname
        if not path.exists():
            print(f"[MISSING] {label}: {fname}")
            continue
        df = pd.read_csv(path, nrows=3, skiprows=(4 if "API_" in fname else 0))
        print(f"\n── {label} ──")
        print(f"   Columns: {list(df.columns)}")
        print(df.head(2).to_string())

if __name__ == "__main__":
    inspect_all()