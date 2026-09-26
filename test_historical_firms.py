import requests
import pandas as pd
from io import StringIO

MAP_KEY = "ef621531ef96b89b454cc5e349fe68be"

AREA = "77.0,15.5,81.5,20.0"
SOURCE = "VIIRS_NOAA20_SP"
DAY_RANGE = 1
DATE = "2025-05-01"

url = (
    f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    f"{MAP_KEY}/{SOURCE}/{AREA}/{DAY_RANGE}/{DATE}"
)

print("Downloading historical FIRMS data...")
print("Date:", DATE)
print("Source:", SOURCE)

response = requests.get(url, timeout=60)

print("\nStatus code:", response.status_code)

if response.status_code != 200:
    print("\nERROR:")
    print(response.text[:2000])
    raise SystemExit

df = pd.read_csv(StringIO(response.text))

print("\nDownload successful!")
print("Number of records:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 records:")
print(df.head(10))

df.to_csv("telangana_fire_2025_05_01.csv", index=False)

print("\nSaved as:")
print("telangana_fire_2025_05_01.csv")