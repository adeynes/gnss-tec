import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import contextily as ctx  # Importer les tuiles OpenStreetMap
from pyproj import Proj, transform  # Conversion en coordonnées projetées
import ast
import folium

# WGS84 vers Web Mercator (EPSG:3857) pour `contextily`
wgs84 = Proj(init="epsg:4326")  # Latitude/Longitude
mercator = Proj(init="epsg:3857")  # Web Mercator

# Constantes WGS84
a = 6378.137  # Demi-grand axe en km
e2 = 0.00669437999014  # Excentricité au carré

# Fonction de conversion ECEF → WGS84
def ecef_to_wgs84(x, y, z):
    lon = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(z, rho * (1 - e2))
    for _ in range(10):
        N = a / np.sqrt(1 - e2 * np.sin(phi)**2)
        h = rho / np.cos(phi) - N
        phi_new = np.arctan2(z + e2 * N * np.sin(phi), rho)
        if abs(phi_new - phi) < 1e-10:
            break
        phi = phi_new
    lat = np.degrees(phi)
    lon = np.degrees(lon)
    return lat, lon

print("Chargement des données")
# Récupération des positions en ECEF
fichier = "gauthier_pos_20250312.txt"
with open(fichier, 'r') as f:
    data = ast.literal_eval(f.read())  # sécurise l'évaluation

positions = np.array(data)[::100]  # ndarray shape (N, 3)
print("Données chargées")

# Conversion en (latitude, longitude)
print("Conversion en latitude, longitude")
lat_lon = np.array([ecef_to_wgs84(*p) for p in positions])

# Convertir en coordonnées Web Mercator
print("Conversion en Web Mercantour")
x_merc, y_merc = transform(wgs84, mercator, lat_lon[:, 1], lat_lon[:, 0])

# Création de la figure
print("Création de la figure")
fig, ax = plt.subplots(figsize=(10, 6))
sns.kdeplot(x=x_merc, y=y_merc, cmap="Reds", fill=True, levels=100, alpha=0.8, ax=ax)

# Ajouter le fond de carte OpenStreetMap
print("Ajout du fond de carte")
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=1)

# Ajustement des axes
ax.set_xlabel("Longitude (°)")
ax.set_ylabel("Latitude (°)")
plt.title("Heatmap des positions GPS avec fond de carte")
plt.grid(True)

print("Affichage !")

plt.show()

# Carte centrée sur le centre des positions
center = [lat_lon[:, 0].mean(), lat_lon[:, 1].mean()]
m = folium.Map(location=center, zoom_start=12)

# Ajout des points
for lat, lon in lat_lon:
    folium.CircleMarker(location=[lat, lon], radius=2, color="red", fill=True).add_to(m)

# Export ou affichage
m.save("carte_positions.html")