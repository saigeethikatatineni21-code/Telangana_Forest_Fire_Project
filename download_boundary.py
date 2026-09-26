import requests

url = "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM1/"

response = requests.get(url)

print("Status:", response.status_code)

if response.status_code == 200:
    data = response.json()

    geojson_url = data["gjDownloadURL"]

    print("Downloading boundary file...")

    boundary = requests.get(geojson_url)

    if boundary.status_code == 200:
        with open("india_adm1.geojson", "wb") as f:
            f.write(boundary.content)

        print("SUCCESS!")
        print("Created: india_adm1.geojson")
    else:
        print("Could not download GeoJSON.")
        print("Status:", boundary.status_code)

else:
    print("Could not access geoBoundaries API.")