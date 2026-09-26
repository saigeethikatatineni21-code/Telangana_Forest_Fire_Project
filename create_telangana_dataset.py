import requests
import pandas as pd
from io import StringIO

MAP_KEY = "ef621531ef96b89b454cc5e349fe68be"

# Telangana bounding box
AREA = "77.0,15.5,81.5,20.0"

# Use Standard Processing for historical data
SOURCE = "VIIRS_NOAA20_SP"

# One day only
DAY_RANGE = 1

url = (
    f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    f"{MAP_KEY}/{SOURCE}/{AREA}/{DAY_RANGE}"
)

print("Downloading FIRMS data...")
print("Status URL request started...")

response = requests.get(url, timeout=60)

print("Status code:", response.status_code)

if response.status_code != 200:
    print("\nERROR:")
    print(response.text[:2000])
    raise SystemExit

# Read CSV returned by NASA
df = pd.read_csv(StringIO(response.text))

print("\nDownload successful!")
print("Number of records:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

# Save
df.to_csv("telangana_fire_test.csv", index=False)

print("\nSaved successfully:")
print("telangana_fire_test.csv")