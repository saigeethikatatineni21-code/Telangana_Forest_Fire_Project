import os
import requests
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
from tqdm import tqdm


# ============================================================
# SETTINGS
# ============================================================

SOURCE = "VIIRS_NOAA20_SP"

# Approximate Telangana bounding box
# west, south, east, north
AREA = "77.0,15.5,81.5,20.0"

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

CHUNK_DAYS = 5

OUTPUT_DIR = "firms_raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# GET MAP KEY
# ============================================================

MAP_KEY = os.environ.get("FIRMS_MAP_KEY")

if not MAP_KEY:
    print("ERROR: FIRMS_MAP_KEY is not set.")
    print()
    print("In PowerShell, run:")
    print('$env:FIRMS_MAP_KEY="YOUR_KEY_HERE"')
    print()
    print("Then run this program again.")
    raise SystemExit


# ============================================================
# DATE RANGE
# ============================================================

start = datetime.strptime(
    START_DATE,
    "%Y-%m-%d"
)

end = datetime.strptime(
    END_DATE,
    "%Y-%m-%d"
)

current = start

all_files = []


# ============================================================
# DOWNLOAD DATA IN 5-DAY CHUNKS
# ============================================================

print("Starting NASA FIRMS historical download...")
print(f"Source: {SOURCE}")
print(f"Period: {START_DATE} to {END_DATE}")
print()


while current <= end:

    chunk_end = min(
        current + timedelta(days=CHUNK_DAYS - 1),
        end
    )

    date_text = current.strftime("%Y-%m-%d")

    filename = (
        f"{OUTPUT_DIR}/"
        f"firms_{current.strftime('%Y%m%d')}_"
        f"{chunk_end.strftime('%Y%m%d')}.csv"
    )

    # Skip already downloaded chunks
    if os.path.exists(filename):

        print(
            f"Already exists: "
            f"{current.strftime('%Y-%m-%d')} → "
            f"{chunk_end.strftime('%Y-%m-%d')}"
        )

        all_files.append(filename)

        current = chunk_end + timedelta(days=1)

        continue


    url = (
        f"https://firms.modaps.eosdis.nasa.gov/"
        f"api/area/csv/"
        f"{MAP_KEY}/"
        f"{SOURCE}/"
        f"{AREA}/"
        f"{CHUNK_DAYS}/"
        f"{date_text}"
    )

    print(
        f"Downloading "
        f"{current.strftime('%Y-%m-%d')} → "
        f"{chunk_end.strftime('%Y-%m-%d')}..."
    )

    try:

        response = requests.get(
            url,
            timeout=60
        )

        if response.status_code != 200:

            print(
                f"ERROR {response.status_code}: "
                f"{response.text[:200]}"
            )

            current = chunk_end + timedelta(days=1)

            continue


        # Read CSV returned by NASA
        data = pd.read_csv(
            StringIO(response.text)
        )

        data.to_csv(
            filename,
            index=False
        )

        print(
            f"  Records downloaded: {len(data)}"
        )

        all_files.append(filename)


    except Exception as e:

        print(
            f"Download failed: {e}"
        )


    current = chunk_end + timedelta(days=1)


# ============================================================
# COMBINE ALL RAW FILES
# ============================================================

print()
print("Combining downloaded files...")

if not all_files:

    print("ERROR: No FIRMS files were downloaded.")
    raise SystemExit


frames = []

for file in tqdm(all_files):

    try:

        data = pd.read_csv(file)

        if len(data) > 0:
            frames.append(data)

    except Exception as e:

        print(
            f"Could not read {file}: {e}"
        )


if not frames:

    print("ERROR: Downloaded files contain no records.")
    raise SystemExit


combined = pd.concat(
    frames,
    ignore_index=True
)


# ============================================================
# REMOVE EXACT DUPLICATES
# ============================================================

before = len(combined)

combined = combined.drop_duplicates()

after = len(combined)

print()
print(f"Total raw records: {before}")
print(f"Exact duplicates removed: {before - after}")
print(f"Final combined records: {after}")


# ============================================================
# SAVE COMBINED RAW DATA
# ============================================================

raw_output = "telangana_firms_2020_2025_raw.csv"

combined.to_csv(
    raw_output,
    index=False
)


print()
print("==============================================")
print("HISTORICAL DOWNLOAD COMPLETED")
print("==============================================")
print(f"Output file: {raw_output}")
print(f"Records: {len(combined)}")
print("==============================================")