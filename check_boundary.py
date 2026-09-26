import geopandas as gpd

print("Loading boundary file...")

gdf = gpd.read_file("india_adm1.geojson")

print("\nState names available in the file:\n")

print(gdf["shapeName"].unique())