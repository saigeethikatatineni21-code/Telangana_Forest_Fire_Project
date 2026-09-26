import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ============================================================
# FILES
# ============================================================

INPUT_FILE = "telangana_firms_2020_2025_raw.csv"
BOUNDARY_FILE = "india_adm1.geojson"
OUTPUT_FILE = "telangana_forest_fire_dataset.csv"

print("=" * 60)
print("TELANGANA FOREST FIRE DATA PROCESSING")
print("=" * 60)

# ============================================================
# 1. LOAD RAW DATA
# ============================================================

print("\n[1] Loading raw FIRMS dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Original records: {len(df):,}")

# ============================================================
# 2. BASIC CLEANING
# ============================================================

print("\n[2] Cleaning data...")

# Remove completely empty rows
df = df.dropna(how="all")

# Convert numeric columns
numeric_columns = [
    "latitude",
    "longitude",
    "bright_ti4",
    "bright_ti5",
    "scan",
    "track",
    "frp"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert date
df["acq_date"] = pd.to_datetime(
    df["acq_date"],
    errors="coerce"
)

# Remove invalid coordinates
df = df.dropna(subset=["latitude", "longitude"])

df = df[
    (df["latitude"] >= -90) &
    (df["latitude"] <= 90) &
    (df["longitude"] >= -180) &
    (df["longitude"] <= 180)
]

# Remove invalid dates
df = df.dropna(subset=["acq_date"])

print(f"Records after basic cleaning: {len(df):,}")

# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates()

duplicates_removed = before_duplicates - len(df)

print(f"Exact duplicates removed: {duplicates_removed:,}")

# ============================================================
# 4. LOAD TELANGANA BOUNDARY
# ============================================================

print("\n[3] Loading Telangana boundary...")

gdf_states = gpd.read_file(BOUNDARY_FILE)

print("Available state names:")
print(gdf_states["shapeName"].unique())

# Find Telangana
telangana = gdf_states[
    gdf_states["shapeName"]
    .astype(str)
    .str.lower()
    .str.contains("telang")
]

if telangana.empty:
    raise ValueError("Telangana boundary not found.")

print("Telangana boundary found successfully.")

# ============================================================
# 5. CONVERT FIRE LOCATIONS TO GEOMETRY
# ============================================================

print("\n[4] Creating geographic points...")

geometry = [
    Point(lon, lat)
    for lon, lat in zip(
        df["longitude"],
        df["latitude"]
    )
]

fires = gpd.GeoDataFrame(
    df,
    geometry=geometry,
    crs="EPSG:4326"
)

# Make sure boundary uses same CRS
telangana = telangana.to_crs("EPSG:4326")

# ============================================================
# 6. FILTER ONLY ACTUAL TELANGANA LOCATIONS
# ============================================================

print("\n[5] Filtering points inside Telangana...")

telangana_geometry = telangana.geometry.union_all()

inside = fires.geometry.within(telangana_geometry)

fires = fires[inside].copy()

print(
    f"Records inside Telangana: {len(fires):,}"
)

# Remove geometry column
fires = pd.DataFrame(fires.drop(columns="geometry"))

# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

print("\n[6] Creating features...")

fires["year"] = fires["acq_date"].dt.year
fires["month"] = fires["acq_date"].dt.month
fires["day"] = fires["acq_date"].dt.day
fires["day_of_year"] = fires["acq_date"].dt.dayofyear

# Season
def get_season(month):

    if month in [12, 1, 2]:
        return "Winter"

    elif month in [3, 4, 5]:
        return "Summer"

    elif month in [6, 7, 8, 9]:
        return "Monsoon"

    else:
        return "Post-Monsoon"


fires["season"] = fires["month"].apply(get_season)

# ============================================================
# 8. FIRE OCCURRENCE
# ============================================================

# FIRMS records represent satellite active-fire detections
fires["fire_occurred"] = 1

# ============================================================
# 9. FIRE SEVERITY PROXY
# ============================================================

def classify_severity(frp):

    if pd.isna(frp):
        return "Unknown"

    elif frp <= 5:
        return "Low"

    elif frp <= 20:
        return "Moderate"

    elif frp <= 50:
        return "High"

    else:
        return "Very High"


fires["fire_severity"] = fires["frp"].apply(
    classify_severity
)

# ============================================================
# 10. CREATE FIRE ID
# ============================================================

fires.insert(
    0,
    "fire_id",
    [
        f"TSF_{i:06d}"
        for i in range(1, len(fires) + 1)
    ]
)

# ============================================================
# 11. REORDER COLUMNS
# ============================================================

preferred_columns = [
    "fire_id",
    "acq_date",
    "acq_time",
    "year",
    "month",
    "day",
    "day_of_year",
    "season",
    "latitude",
    "longitude",
    "satellite",
    "instrument",
    "confidence",
    "bright_ti4",
    "bright_ti5",
    "frp",
    "scan",
    "track",
    "daynight",
    "type",
    "version",
    "fire_occurred",
    "fire_severity"
]

existing_columns = [
    col for col in preferred_columns
    if col in fires.columns
]

fires = fires[existing_columns]

# ============================================================
# 12. SAVE FINAL DATASET
# ============================================================

fires.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# 13. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PROCESSING COMPLETED")
print("=" * 60)

print(f"Original raw records     : {len(df):,}")
print(f"Final Telangana records  : {len(fires):,}")
print(f"Total columns             : {len(fires.columns)}")

print("\nYear distribution:")
print(fires["year"].value_counts().sort_index())

print("\nSeason distribution:")
print(fires["season"].value_counts())

print("\nFire severity distribution:")
print(fires["fire_severity"].value_counts())

print("\nOutput file:")
print(OUTPUT_FILE)

print("=" * 60)