from pathlib import Path

import geopandas as gpd
import pandas as pd


# ============================================================
# FILES
# ============================================================

FIRMS_FILE = Path(
    "firms_raw/telangana_2026_latest_nrt.csv"
)

BOUNDARY_FILE = Path(
    "india_adm1.geojson"
)

EXISTING_FILE = Path(
    "telangana_forest_fire_dataset.csv"
)

OUTPUT_FILE = Path(
    "telangana_forest_fire_dataset_updated.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MERGING RECENT 2026 NASA FIRMS DATA")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for file_path in [
        FIRMS_FILE,
        BOUNDARY_FILE,
        EXISTING_FILE,
    ]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    # --------------------------------------------------------
    # Read FIRMS data
    # --------------------------------------------------------

    print("\nReading recent FIRMS data...")

    firms = pd.read_csv(FIRMS_FILE)

    print(
        "FIRMS records loaded:",
        len(firms)
    )

    # --------------------------------------------------------
    # Read Telangana boundary
    # --------------------------------------------------------

    print("\nReading India boundary file...")

    boundary = gpd.read_file(BOUNDARY_FILE)

    print("Boundary columns:")
    print(boundary.columns.tolist())

    # Your boundary file previously contained Telangāna
    # in the shapeName column.
    state_column = "shapeName"

    if state_column not in boundary.columns:
        raise ValueError(
            f"'{state_column}' column was not found."
        )

    # Normalize the Telangana name.
    state_names = (
        boundary[state_column]
        .astype(str)
        .str.lower()
        .str.replace("ā", "a", regex=False)
        .str.strip()
    )

    telangana_boundary = boundary[
        state_names == "telangana"
    ].copy()

    if telangana_boundary.empty:
        raise ValueError(
            "Telangana boundary was not found."
        )

    print("Telangana boundary found.")

    # --------------------------------------------------------
    # Prepare FIRMS coordinates
    # --------------------------------------------------------

    firms["latitude"] = pd.to_numeric(
        firms["latitude"],
        errors="coerce"
    )

    firms["longitude"] = pd.to_numeric(
        firms["longitude"],
        errors="coerce"
    )

    firms = firms.dropna(
        subset=["latitude", "longitude"]
    ).copy()

    # --------------------------------------------------------
    # Convert FIRMS records to geographic points
    # --------------------------------------------------------

    firms_gdf = gpd.GeoDataFrame(
        firms,
        geometry=gpd.points_from_xy(
            firms["longitude"],
            firms["latitude"]
        ),
        crs="EPSG:4326"
    )

    # Make sure Telangana boundary uses the same CRS.
    if telangana_boundary.crs is None:
        telangana_boundary = (
            telangana_boundary.set_crs("EPSG:4326")
        )
    else:
        telangana_boundary = (
            telangana_boundary.to_crs("EPSG:4326")
        )

    # --------------------------------------------------------
    # Keep only points inside Telangana
    # --------------------------------------------------------

    print(
        "\nChecking which FIRMS points are inside Telangana..."
    )

    joined = gpd.sjoin(
        firms_gdf,
        telangana_boundary[["geometry"]],
        how="inner",
        predicate="within"
    )

    telangana_firms = pd.DataFrame(
        joined.drop(
            columns=["geometry", "index_right"],
            errors="ignore"
        )
    )

    print(
        "Records inside Telangana:",
        len(telangana_firms)
    )

    print(
        "Records outside Telangana:",
        len(firms) - len(telangana_firms)
    )

    if telangana_firms.empty:
        raise RuntimeError(
            "No FIRMS records were found inside Telangana."
        )

    # --------------------------------------------------------
    # Read existing dataset
    # --------------------------------------------------------

    print("\nReading existing Telangana dataset...")

    existing = pd.read_csv(EXISTING_FILE)

    print(
        "Existing records:",
        f"{len(existing):,}"
    )

    print(
        "Existing columns:",
        existing.columns.tolist()
    )

    # --------------------------------------------------------
    # Standardize FIRMS date
    # --------------------------------------------------------

    if "acq_date" in telangana_firms.columns:

        dates = pd.to_datetime(
            telangana_firms["acq_date"],
            errors="coerce"
        )

        telangana_firms["acq_date"] = (
            dates.dt.strftime("%Y-%m-%d")
        )

    # --------------------------------------------------------
    # Rename FIRMS columns where the existing dataset
    # uses different names.
    # --------------------------------------------------------

    rename_map = {}

    if (
        "acq_date" in telangana_firms.columns
        and "date" in existing.columns
    ):
        rename_map["acq_date"] = "date"

    if (
        "frp" in telangana_firms.columns
        and "fire_radiative_power" in existing.columns
    ):
        rename_map["frp"] = "fire_radiative_power"

    if (
        "bright_ti4" in telangana_firms.columns
        and "brightness" in existing.columns
    ):
        rename_map["bright_ti4"] = "brightness"

    if (
        "source" in telangana_firms.columns
        and "satellite_source" in existing.columns
    ):
        rename_map["source"] = "satellite_source"

    telangana_firms = telangana_firms.rename(
        columns=rename_map
    )

    # --------------------------------------------------------
    # Add columns that exist in the main dataset but not
    # in the new FIRMS records.
    #
    # We leave them empty rather than inventing values.
    # --------------------------------------------------------

    for column in existing.columns:

        if column not in telangana_firms.columns:

            telangana_firms[column] = pd.NA

    # Keep exactly the same column structure.
    new_rows = telangana_firms[
        existing.columns
    ].copy()

    # --------------------------------------------------------
    # Add year/month if those columns exist
    # --------------------------------------------------------

    if "date" in new_rows.columns:

        dates = pd.to_datetime(
            new_rows["date"],
            errors="coerce"
        )

        if "year" in new_rows.columns:
            new_rows["year"] = dates.dt.year

        if "month" in new_rows.columns:
            new_rows["month"] = dates.dt.month

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    print("\nCombining datasets...")

    combined = pd.concat(
        [
            existing,
            new_rows
        ],
        ignore_index=True
    )

    before_duplicates = len(combined)

    combined = combined.drop_duplicates()

    duplicates_removed = (
        before_duplicates - len(combined)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    combined.to_csv(
        OUTPUT_FILE,
        index=False
    )

    added_records = (
        len(combined) - len(existing)
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MERGE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        "Original records:",
        f"{len(existing):,}"
    )

    print(
        "FIRMS records inside Telangana:",
        f"{len(new_rows):,}"
    )

    print(
        "New records actually added:",
        f"{added_records:,}"
    )

    print(
        "Duplicates removed:",
        f"{duplicates_removed:,}"
    )

    print(
        "Updated total records:",
        f"{len(combined):,}"
    )

    print()
    print(
        "Created:",
        OUTPUT_FILE
    )

    print()
    print(
        "Original dataset was NOT changed."
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()