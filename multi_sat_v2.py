import georinex as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr


# def elevation_angle(x_s, y_s, z_s, x_r, y_r, z_r):      #Calcule l'élévation du satellite s par rapport au récepteur r
#     vec = np.array([x_s - x_r, y_s - y_r, z_s - z_r])
#     norm_vec = np.linalg.norm(vec)
#     vec_unit = vec / norm_vec
#
#     r_unit = np.array([x_r, y_r, z_r]) / np.linalg.norm([x_r, y_r, z_r])
#     cos_el = np.dot(vec_unit, r_unit)
#     el_rad = np.pi/2 - np.arccos(cos_el)
#     return np.degrees(el_rad)


#Chargement des fichiers RINEX
fichiers_rinex = ["sirt1203.25o", "sirt1303.25o", "sirt1403.25o", "sirt1503.25o", "sirt1603.25o", "sirt1703.25o",
                  "sirt1803.25o", "sirt1903.25o"]

datasets = []
index = 1
for fichier in fichiers_rinex:
    print(f"Chargement de : {fichier}  -- Fichier {index} sur {fichiers_rinex.__len__()}")
    ds = gr.load("Fichiers_RINEX_SIRTA_1203_1903/" + fichier)
    print(f"{fichier} :  de {ds.time.values[0]} à {ds.time.values[-1]}")
    datasets.append(ds)
    index += 1

data = xr.concat(datasets, dim='time')

#Chargement des fichiers de nav

fichiers_nav = ["sirt1203.25n","sirt1303.25n","sirt1403.25n","sirt1503.25n",
                "sirt1603.25n","sirt1703.25n","sirt1803.25n","sirt1903.25n"]

datasets_nav = [gr.load("Fichiers_RINEX_SIRTA_1203_1903/" + f) for f in fichiers_nav]
nav = xr.concat(datasets_nav, dim='time')

# Fréquences GPS L1 et L2
f1 = 1575.42e6
f2 = 1227.60e6

# Longueurs d'onde
lambda1 = 299792458 / f1
lambda2 = 299792458 / f2

# Constante de conversion pour le TEC
K = 40.3 * (f1**2 - f2**2) / (f1**2 * f2**2) *1e16

#Position du SIRTA
lat_deg = 48.71215112  # Latitude
lon_deg = 2.20856274   # Longitude
alt_m = 217.826       # Altitude

from pyproj import Transformer
transformer = Transformer.from_crs("epsg:4979", "epsg:4978")  # WGS84 -> ECEF
x_rec, y_rec, z_rec = transformer.transform(lat_deg, lon_deg, alt_m)

# Liste pour stocker les DataFrames de chaque satellite
df_list = []

# Boucle sur tous les satellites disponibles
for satellite in data.sv.values:

    # # Obtenir les coordonnées du satellite à ces temps
    # try:
    #     sat_pos = gr.sv_position(nav, sat_data.time, satellite)
    # except Exception as e:
    #     print(f"Erreur position satellite {satellite} : {e}")
    #     continue
    #
    # elev = elevation_angle(
    #     sat_pos["x"].values,
    #     sat_pos["y"].values,
    #     sat_pos["z"].values,
    #     x_rec, y_rec, z_rec
    # )
    #
    # # Filtrer si élévation < 30°
    # if np.nanmean(elev) < 30:
    #     continue

    sat_data = data.sel(sv=satellite)
    print(sat_data)
    print("\n")
    sat_data = sat_data.dropna(dim="time", subset=["P1", "P2"])

    # Extraire pseudo-distances et phases
    P1 = sat_data["P1"].values  # Code L1
    P2 = sat_data["P2"].values  # Code L2

    # TEC basé sur les pseudo-distances
    TEC_P = (P2 - P1) / K
    if len(TEC_P) == 0:
        print(f"Aucune donnée valide pour le satellite {satellite}. \n")
        continue

    # Lissage exponentiel du TEC
    alpha = 0.1
    TEC_LS = np.zeros_like(TEC_P)
    TEC_LS[0] = TEC_P[0]

    for i in range(1, len(TEC_P)):
        TEC_LS[i] = alpha * TEC_P[i] + (1 - alpha) * TEC_LS[i - 1]

    # Stocker les résultats dans un DataFrame
    df_sat = pd.DataFrame({
        "Time": sat_data.time.values,
        "TEC_P": TEC_P,
        "TEC_LS": TEC_LS,
        "Satellite": satellite
    })

    print(f"Période horaire du satellite {satellite} : ")
    print(df_sat["Time"].min(), "-", df_sat["Time"].max())
    print("\n")

    df_list.append(df_sat)

# Fusionner toutes les données satellites
df_all = pd.concat(df_list, ignore_index=True)

# Affichage des résultats en TEC_P pour tous les satellites
plt.figure(figsize=(12, 6))
for satellite in df_all["Satellite"].unique():
    df_sat = df_all[df_all["Satellite"] == satellite]  #.dropna(subset=["TEC_LS"])
    plt.plot(df_sat["Time"], df_sat["TEC_LS"], "o", markersize=1, label=satellite)

df_all["Time"] = pd.to_datetime(df_all["Time"])
df_avg = df_all.groupby("Time")["TEC_LS"].mean().reset_index()
plt.plot(df_avg["Time"], df_avg["TEC_LS"], color="black", linewidth=1.5, label="Moyenne (tous satellites)")

plt.xlabel("Temps")
plt.ylabel("TEC (TECU)")
plt.title("sTEC lissé pour tous les satellites")
plt.legend(ncol=4, fontsize=8)
plt.xticks(rotation=45)
plt.grid()
plt.show()
