import pandas as pd
import numpy as np

# ============================================================
# FILES
# ============================================================

RAW_FILE = "telangana_firms_2020_2025_raw.csv"
CLEAN_FILE = "telangana_forest_fire_dataset.csv"

print("=" * 70)
print("TELANGANA FOREST FIRE - DATA QUALITY ANALYSIS")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1] Loading datasets...")

raw = pd.read_csv(RAW_FILE)
clean = pd.read_csv(CLEAN_FILE)

print(f"Raw dataset records   : {len(raw):,}")
print(f"Clean dataset records : {len(clean):,}")

print(f"Raw dataset columns   : {len(raw.columns)}")
print(f"Clean dataset columns : {len(clean.columns)}")


# ============================================================
# 2. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

raw_missing = raw.isnull().sum()
clean_missing = clean.isnull().sum()

missing_table = pd.DataFrame({
    "Column": raw.columns,
    "Raw_Missing": raw_missing.values,
    "Clean_Missing": [
        clean_missing.get(col, 0)
        for col in raw.columns
    ]
})

missing_table["Raw_Missing_%"] = (
    missing_table["Raw_Missing"] /
    len(raw) * 100
).round(2)

missing_table["Clean_Missing_%"] = (
    missing_table["Clean_Missing"] /
    len(clean) * 100
).round(2)

print(missing_table.to_string(index=False))


# ============================================================
# 3. DUPLICATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE ANALYSIS")
print("=" * 70)

raw_duplicates = raw.duplicated().sum()
clean_duplicates = clean.duplicated().sum()

print(f"Raw duplicate records   : {raw_duplicates:,}")
print(f"Clean duplicate records : {clean_duplicates:,}")


# ============================================================
# 4. INVALID COORDINATES
# ============================================================

print("\n" + "=" * 70)
print("COORDINATE VALIDATION")
print("=" * 70)

raw_invalid_coords = (
    (raw["latitude"] < -90) |
    (raw["latitude"] > 90) |
    (raw["longitude"] < -180) |
    (raw["longitude"] > 180)
).sum()

clean_invalid_coords = (
    (clean["latitude"] < -90) |
    (clean["latitude"] > 90) |
    (clean["longitude"] < -180) |
    (clean["longitude"] > 180)
).sum()

print(f"Raw invalid coordinates   : {raw_invalid_coords:,}")
print(f"Clean invalid coordinates : {clean_invalid_coords:,}")


# ============================================================
# 5. DATE VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("DATE VALIDATION")
print("=" * 70)

raw_dates = pd.to_datetime(
    raw["acq_date"],
    errors="coerce"
)

clean_dates = pd.to_datetime(
    clean["acq_date"],
    errors="coerce"
)

print(
    f"Raw invalid dates   : {raw_dates.isna().sum():,}"
)

print(
    f"Clean invalid dates : {clean_dates.isna().sum():,}"
)


# ============================================================
# 6. DATA COMPLETENESS
# ============================================================

print("\n" + "=" * 70)
print("DATA COMPLETENESS")
print("=" * 70)

raw_total_cells = raw.shape[0] * raw.shape[1]
clean_total_cells = clean.shape[0] * clean.shape[1]

raw_missing_cells = raw.isnull().sum().sum()
clean_missing_cells = clean.isnull().sum().sum()

raw_completeness = (
    1 - raw_missing_cells / raw_total_cells
) * 100

clean_completeness = (
    1 - clean_missing_cells / clean_total_cells
) * 100

print(f"Raw data completeness   : {raw_completeness:.2f}%")
print(f"Clean data completeness : {clean_completeness:.2f}%")


# ============================================================
# 7. TELANGANA FILTERING
# ============================================================

print("\n" + "=" * 70)
print("TELANGANA BOUNDARY FILTERING")
print("=" * 70)

records_removed = len(raw) - len(clean)

retention_percentage = (
    len(clean) / len(raw)
) * 100

print(f"Raw records                 : {len(raw):,}")
print(f"Records after Telangana filter: {len(clean):,}")
print(f"Records removed              : {records_removed:,}")
print(f"Records retained             : {retention_percentage:.2f}%")


# ============================================================
# 8. DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(clean.dtypes.to_string())


# ============================================================
# 9. NUMERICAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("NUMERICAL DATA SUMMARY")
print("=" * 70)

numeric_columns = clean.select_dtypes(
    include=np.number
).columns

print(
    clean[numeric_columns]
    .describe()
    .round(2)
    .to_string()
)


# ============================================================
# 10. YEAR DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("YEAR DISTRIBUTION")
print("=" * 70)

print(
    clean["year"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 11. SEASON DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("SEASON DISTRIBUTION")
print("=" * 70)

print(
    clean["season"]
    .value_counts()
)


# ============================================================
# 12. FIRE SEVERITY
# ============================================================

print("\n" + "=" * 70)
print("FIRE SEVERITY DISTRIBUTION")
print("=" * 70)

print(
    clean["fire_severity"]
    .value_counts()
)


# ============================================================
# 13. SAVE QUALITY REPORT
# ============================================================

report = {
    "Raw Records": len(raw),
    "Clean Records": len(clean),
    "Records Removed": records_removed,
    "Retention Percentage": round(retention_percentage, 2),
    "Raw Duplicate Records": raw_duplicates,
    "Clean Duplicate Records": clean_duplicates,
    "Raw Invalid Coordinates": raw_invalid_coords,
    "Clean Invalid Coordinates": clean_invalid_coords,
    "Raw Invalid Dates": raw_dates.isna().sum(),
    "Clean Invalid Dates": clean_dates.isna().sum(),
    "Raw Completeness (%)": round(raw_completeness, 2),
    "Clean Completeness (%)": round(clean_completeness, 2)
}

report_df = pd.DataFrame(
    list(report.items()),
    columns=["Metric", "Value"]
)

report_df.to_csv(
    "data_quality_report.csv",
    index=False
)

missing_table.to_csv(
    "missing_value_analysis.csv",
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("DATA QUALITY ANALYSIS COMPLETED")
print("=" * 70)

print("\nFiles created:")
print("1. data_quality_report.csv")
print("2. missing_value_analysis.csv")

print("\nYou can now use these results in your PPT/report.")

print("=" * 70)