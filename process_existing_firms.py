import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = "TELANGANA_FIRE_FOREST.CSV"
BOUNDARY_FILE = "india_adm1.geojson"
OUTPUT_FILE = "telangana_forest_fire_dataset.csv"


# ============================================================
# 1. CHECK INPUT FILES
# ============================================================

if not os.path.exists(INPUT_FILE):
    print(f"ERROR: {INPUT_FILE} not found.")
    raise SystemExit

if not os.path.exists(BOUNDARY_FILE):
    print(f"ERROR: {BOUNDARY_FILE} not found.")
    raise SystemExit


# ============================================================
# 2. LOAD FIRMS DATA
# ============================================================

print("Loading FIRMS data...")

df = pd.read_csv(INPUT_FILE)

print(f"Original records: {len(df)}")

print("\nOriginal columns:")
print(df.columns.tolist())


# ============================================================
# 3. BASIC DATA CLEANING
# ============================================================

# Remove completely empty rows
df = df.dropna(how="all").copy()

# Convert latitude and longitude to numeric
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

# Convert FRP and brightness values to numeric
numeric_columns = [
    "bright_ti4",
    "bright_ti5",
    "scan",
    "track",
    "frp"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove rows without coordinates
df = df.dropna(subset=["latitude", "longitude"]).copy()

# Convert acquisition date
df["acq_date"] = pd.to_datetime(
    df["acq_date"],
    errors="coerce"
)

# Remove rows with invalid dates
df = df.dropna(subset=["acq_date"]).copy()

print(f"\nRecords after basic cleaning: {len(df)}")


# ============================================================
# 4. REMOVE EXACT DUPLICATES
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates().copy()

after_duplicates = len(df)

print(
    f"Exact duplicates removed: "
    f"{before_duplicates - after_duplicates}"
)


# ============================================================
# 5. LOAD INDIA STATE BOUNDARIES
# ============================================================

print("\nLoading state boundaries...")

states = gpd.read_file(BOUNDARY_FILE)

print("Boundary columns:")
print(states.columns.tolist())


# ============================================================
# 6. FIND STATE NAME COLUMN
# ============================================================

state_col = "shapeName"

print("State-name column:", state_col)


# ============================================================
# 7. FIND TELANGANA
# ============================================================

# geoBoundaries uses "Telangāna"
telangana = states[
    states[state_col]
    .astype(str)
    .str.strip()
    .str.lower()
    == "telangāna"
].copy()


if telangana.empty:

    print("\nERROR: Telangana boundary not found.")

    print("\nAvailable state names:")
    print(states[state_col].unique())

    raise SystemExit


print("Telangana boundary found.")


# ============================================================
# 8. CREATE GEOMETRY FROM LATITUDE/LONGITUDE
# ============================================================

print("\nCreating geographic points...")

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


# ============================================================
# 9. MAKE SURE BOUNDARY USES SAME CRS
# ============================================================

telangana = telangana.to_crs("EPSG:4326")


# ============================================================
# 10. FILTER ONLY POINTS INSIDE TELANGANA
# ============================================================

print("Filtering fire detections to Telangana...")

telangana_geometry = telangana.geometry.union_all()

inside_mask = fires.geometry.within(
    telangana_geometry
)

fires = fires[inside_mask].copy()

print(
    f"Records inside Telangana boundary: "
    f"{len(fires)}"
)


# ============================================================
# 11. REMOVE GEOMETRY BEFORE SAVING CSV
# ============================================================

fires = fires.drop(columns=["geometry"])


# ============================================================
# 12. CREATE DATE FEATURES
# ============================================================

print("\nCreating date features...")

fires["year"] = fires["acq_date"].dt.year
fires["month"] = fires["acq_date"].dt.month
fires["day"] = fires["acq_date"].dt.day
fires["day_of_year"] = fires["acq_date"].dt.dayofyear


# ============================================================
# 13. CREATE SEASON
# ============================================================

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
# 14. FIRE OCCURRED LABEL
# ============================================================

# FIRMS records represent satellite active-fire detections.
# Therefore, every record in this positive detection dataset
# receives fire_occurred = 1.
#
# This does NOT create negative/background samples.

fires["fire_occurred"] = 1


# ============================================================
# 15. FIRE SEVERITY PROXY USING FRP
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
# 16. CREATE FIRE ID
# ============================================================

fires = fires.reset_index(drop=True)

fires["fire_id"] = [
    f"TSF_{i:06d}"
    for i in range(1, len(fires) + 1)
]


# ============================================================
# 17. REORDER IMPORTANT COLUMNS
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


# Keep only columns that actually exist
existing_columns = [
    column
    for column in preferred_columns
    if column in fires.columns
]

remaining_columns = [
    column
    for column in fires.columns
    if column not in existing_columns
]

fires = fires[
    existing_columns + remaining_columns
]


# ============================================================
# 18. SAVE FINAL DATASET
# ============================================================

fires.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 19. DISPLAY RESULTS
# ============================================================

print("\n==============================================")
print("DATASET CREATION COMPLETED")
print("==============================================")

print(f"Final records: {len(fires)}")

print(f"Output file: {OUTPUT_FILE}")

print("\nFinal columns:")
print(fires.columns.tolist())

print("\nSeason distribution:")
print(fires["season"].value_counts())

print("\nFire severity distribution:")
print(fires["fire_severity"].value_counts())

print("\nFirst 5 records:")
print(fires.head())

print("\n==============================================")
print("SUCCESS")
print("==============================================")