# Data Sources

Download the following files and place them in this folder (`data/raw/`) 
before running the pipeline.

---

## 1. ERA5 Climate Data (Copernicus Climate Data Store)

**Both files are required.**

### How to download

1. Create a free account at https://cds.climate.copernicus.eu
2. Go to: Datasets → ERA5 Monthly Averaged Data on Single Levels
   https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means
3. Configure the request form as follows:

**For the temperature + dewpoint file (avgua):**

| Field | Value |
|---|---|
| Product type | Monthly averaged reanalysis |
| Variable | 2m temperature, 2m dewpoint temperature |
| Year | 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024 |
| Month | All (January – December) |
| Time | 00:00 |
| Geographic area | Europe (North: 72, West: -25, East: 45, South: 35) |
| Format | CSV |

Save the downloaded file as:
```
data_stream-moda_stepType-avgua.csv
```

**For the precipitation file (avgad):**

| Field | Value |
|---|---|
| Product type | Monthly averaged reanalysis by hour of day |
| Variable | Total precipitation |
| Year | 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024 |
| Month | All (January – December) |
| Time | 06:00 |
| Geographic area | Europe (North: 72, West: -25, East: 45, South: 35) |
| Format | CSV |

Save the downloaded file as:
```
data_stream-moda_stepType-avgad.csv
```

### Notes
- Downloads are free but require registration
- File size is approximately 50–200 MB depending on resolution
- Processing can take a few minutes on the CDS queue
- Licence: Copernicus Licence Agreement
  https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf

---

## 2. GDP per capita PPP (World Bank)

**Direct download link:**
https://api.worldbank.org/v2/en/indicator/NY.GDP.PCAP.PP.CD?downloadformat=csv

Or manually:
1. Go to https://data.worldbank.org/indicator/NY.GDP.PCAP.PP.CD
2. Click **Download** → **CSV**
3. Extract the zip file

Save the file as:
```
API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_35.csv
```

- Licence: Creative Commons Attribution 4.0
- No registration required

---

## 3. Urban Population % (World Bank)

**Direct download link:**
https://api.worldbank.org/v2/en/indicator/SP.URB.TOTL.IN.ZS?downloadformat=csv

Or manually:
1. Go to https://data.worldbank.org/indicator/SP.URB.TOTL.IN.ZS
2. Click **Download** → **CSV**
3. Extract the zip file

Save the file as:
```
API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_249.csv
```

- Licence: Creative Commons Attribution 4.0
- No registration required

---

## 4. HEPA Physical Activity Data (WHO Europe)

1. Go to https://www.euro.who.int/en/health-topics/disease-prevention/
   physical-activity/data-and-statistics/hepa-europe-scorecards
2. Download the HEPA scorecard data tables
3. Save the two files as:
```
HEPA Data (table).csv
HEPA Data (pivoted).csv
```

- Licence: WHO open data licence
- No registration required

---

## Expected folder contents after download
```
data/raw/
├── data_stream-moda_stepType-avgad.csv
├── data_stream-moda_stepType-avgua.csv
├── API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_35.csv
├── API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_249.csv
├── HEPA Data (table).csv
└── HEPA Data (pivoted).csv
```

Once all files are in place, return to the root README and follow
the **Running the pipeline** instructions.