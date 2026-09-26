import requests

MAP_KEY = "ef621531ef96b89b454cc5e349fe68be"

url = (
    "https://firms.modaps.eosdis.nasa.gov/"
    "api/data_availability/"
)

response = requests.get(url, timeout=60)

print("Status code:", response.status_code)
print(response.text[:5000])