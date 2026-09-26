import os
from getpass import getpass
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# ============================================================
# CONFIGURATION
# ============================================================

# Telangana bounding box: west,south,east,north
TELANGANA_BBOX = "77.0,15.5,81.5,20.0"

# Current NASA FIRMS Near-Real-Time sensors
SOURCES = [
    "VIIRS_NOAA21_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_SNPP_NRT",
]

# FIRMS Area API allows 1 to 5 days per request
DAY_RANGE = 5

OUTPUT_DIR = Path("firms_raw")
OUTPUT_FILE = OUTPUT_DIR / "telangana_2026_latest_nrt.csv"

TIMEOUT = 120


# ============================================================
# MAP KEY
# ============================================================

def get_map_key():
    # First try an environment variable.
    map_key = os.getenv("FIRMS_MAP_KEY", "").strip()

    if map_key:
        print("Using FIRMS MAP_KEY from environment variable.")
        return map_key

    # Otherwise securely ask for the existing key.
    print()
    print("=" * 60)
    print("NASA FIRMS MAP_KEY required")
    print("=" * 60)
    print("Your key will not be displayed while typing.")
    print()

    map_key = getpass(
        "Enter your existing FIRMS MAP_KEY: "
    ).strip()

    if not map_key:
        raise RuntimeError("No MAP_KEY was provided.")

    return map_key


# ============================================================
# CHECK MAP KEY
# ============================================================

def check_map_key(map_key):
    url = (
        "https://firms.modaps.eosdis.nasa.gov/"
        "mapserver/mapkey_status/"
        f"?MAP_KEY={map_key}"
    )

    print("Checking NASA FIRMS MAP_KEY...")

    response = requests.get(
        url,
        timeout=TIMEOUT
    )

    if response.status_code != 200:
        print("MAP_KEY check failed.")
        print("HTTP status:", response.status_code)
        print(response.text[:1000])
        raise RuntimeError(
            "The FIRMS MAP_KEY could not be verified."
        )

    data = response.json()

    print("MAP_KEY is valid.")
    print("Transaction limit:", data.get("transaction_limit"))
    print("Current transactions:", data.get("current_transactions"))


# ============================================================
# DOWNLOAD ONE SENSOR
# ============================================================

def download_sensor(map_key, source):
    url = (
        "https://firms.modaps.eosdis.nasa.gov/"
        "api/area/csv/"
        f"{map_key}/{source}/{TELANGANA_BBOX}/{DAY_RANGE}"
    )

    print()
    print("-" * 60)
    print("Downloading:", source)
    print("Area: Telangana")
    print("Days requested:", DAY_RANGE)

    response = requests.get(
        url,
        timeout=TIMEOUT
    )

    print("HTTP status:", response.status_code)

    if response.status_code != 200:
        print("NASA FIRMS returned an error:")
        print(response.text[:1500])
        return pd.DataFrame()

    if not response.text.strip():
        print("No data returned.")
        return pd.DataFrame()

    try:
        data = pd.read_csv(StringIO(response.text))
    except Exception as error:
        print("Could not read FIRMS CSV:", error)
        print(response.text[:1000])
        return pd.DataFrame()

    if data.empty:
        print("No detections returned.")
        return data

    data["source"] = source

    print("Records downloaded:", f"{len(data):,}")

    return data


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 70)
    print("TELANGANA 2026 NASA FIRMS NRT DOWNLOADER")
    print("=" * 70)

    map_key = get_map_key()
    check_map_key(map_key)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_data = []

    for source in SOURCES:
        try:
            data = download_sensor(
                map_key,
                source
            )

            if not data.empty:
                all_data.append(data)

        except requests.RequestException as error:
            print("Network error for", source)
            print(error)

        except Exception as error:
            print("Error for", source)
            print(error)

    if not all_data:
        print()
        print("=" * 70)
        print("NO FIRMS DATA WAS DOWNLOADED")
        print("=" * 70)
        print("Possible reasons:")
        print("1. No recent detections in the Telangana bounding box.")
        print("2. An NRT product has a temporary data delay.")
        print("3. FIRMS has a temporary service/data issue.")
        return

    # Combine all sensors.
    df = pd.concat(
        all_data,
        ignore_index=True
    )

    # Standardize date.
    df["acq_date"] = pd.to_datetime(
        df["acq_date"],
        errors="coerce"
    )

    # Keep only 2026.
    df = df[
        df["acq_date"].dt.year == 2026
    ].copy()

    # Numeric columns.
    numeric_columns = [
        "latitude",
        "longitude",
        "frp",
        "scan",
        "track",
        "bright_ti4",
        "bright_ti5",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Remove exact duplicates.
    before = len(df)
    df = df.drop_duplicates()
    duplicates_removed = before - len(df)

    # Sort records.
    sort_columns = [
        column
        for column in [
            "acq_date",
            "latitude",
            "longitude",
            "source"
        ]
        if column in df.columns
    ]

    if sort_columns:
        df = df.sort_values(sort_columns)

    # Save.
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # Summary.
    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETED")
    print("=" * 70)
    print("Output file:", OUTPUT_FILE)
    print("Total 2026 records:", f"{len(df):,}")
    print("Exact duplicates removed:", f"{duplicates_removed:,}")

    if not df.empty:
        print(
            "Earliest observation:",
            df["acq_date"].min().strftime("%Y-%m-%d")
        )
        print(
            "Latest observation:",
            df["acq_date"].max().strftime("%Y-%m-%d")
        )

        print()
        print("Records by sensor:")
        print(
            df["source"].value_counts().to_string()
        )

        print()
        print("Records by date:")
        print(
            df["acq_date"]
            .dt.strftime("%Y-%m-%d")
            .value_counts()
            .sort_index()
            .to_string()
        )

        print()
        print("First 5 records:")
        print(
            df.head().to_string(index=False)
        )

    else:
        print()
        print(
            "The API responded successfully, but no 2026 "
            "detections were returned."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
