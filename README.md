# Calibration radiométrique d'images thermiques drone — Niakhar

Pipeline de calibration radiométrique physiquement fondée pour images thermiques acquises par drone DJI Mavic 3T sur le site de Niakhar, Sénégal.

École Polytechnique de Thiès.

---

## Contexte

Les caméras thermiques non refroidies (uncooled) embarquées sur drone ne produisent pas directement des températures absolues — elles produisent des valeurs numériques brutes (Digital Numbers, DN). L'objectif de ce pipeline est de convertir ces DN en températures de surface physiquement correctes (°C), en exploitant :

- Des **plaques étalons** de référence (matériaux à émissivité connue) déployées sur le terrain
- Les données de la **tour à flux Eddy covariance** de Niakhar (T_air, HR, rayonnement atmosphérique descendant L↓)
- L'**équation de transfert radiatif** (Berni et al. 2009)

---

## Approche

La calibration se déroule en deux phases :

**Phase 1 — Régression DN→L (Jolivot 2017)**
Pour chaque plaque étalon, on calcule la luminance théorique attendue par le capteur :
```
L_theo = τ * [ε*B(Ts) + (1-ε)*L↓] + L↑
```
La régression linéaire sur les paires `(DN, L_theo)` donne les coefficients `a, b` tels que :
```
L_sensor = a*DN + b
```

**Phase 2 — Inversion physique pixel par pixel (Berni 2009)**
```
B(Ts) = [(L_sensor - L↑) / τ - (1-ε)*L↓] / ε
T_surface = (B(Ts) / σ)^0.25 - 273.15
```

---

## Structure du projet

```
niakhar/
├── dji_irp.exe              # Outil DJI (requis, non inclus dans le repo)
├── *.dll                    # Librairies DJI (requises, non incluses)
│
├── calibration/
│   ├── extract_dn.py        # Extraction des DN bruts depuis un RJPEG DJI
│   ├── extract_dn_test.py   # Script de test rapide et visualisation DN
│   ├── calibrate.py         # Pipeline complet DN → L → T surface
│   └── data/
│       ├── plaques.csv      # Plaques étalons (DN, T_ref, émissivité)
│       └── tour_flux.csv    # Données tour à flux (T_air, HR, L↓, altitude)
│
└── Data_DJI/
    └── Images thermiques/   # RJPEG DJI (.JPG) — non inclus dans le repo
```

---

## Prérequis

```bash
pip install -r requirements.txt
```

`dji_irp.exe` et ses `.dll` associées doivent être placés à la racine du projet. Ils sont fournis par DJI et ne sont pas redistribuables.

---

## Utilisation

**1. Extraction des DN bruts depuis un RJPEG :**

```python
from extract_dn import extract_dn

dn = extract_dn("chemin/vers/image_T.JPG")  # retourne np.ndarray (512, 640) uint16
```

**2. Pipeline de calibration complet :**

```bash
python calibration/calibrate.py
```

Produit :
- Affichage des coefficients de régression `L = a*DN + b`
- Image de température de surface calibrée (°C)
- Visualisation sauvegardée dans `calibration/T_calibree.png`

**3. Test rapide visualisation DN :**

```bash
python calibration/extract_dn_test.py
```

---

## Format des fichiers de données

**`calibration/data/plaques.csv`**
```csv
plaque,DN_moyen,T_ref_celsius,emissivite
noire,19500,47.0,0.95
grise,16000,38.0,0.94
blanche,14200,31.0,0.92
```

**`calibration/data/tour_flux.csv`**
```csv
T_air_celsius,HR_pct,L_down_atm,altitude_vol_m
38.5,32.0,415.0,100.0
```

---

## Références

- Berni et al. (2009). *Thermal and Narrowband Multispectral Remote Sensing for Vegetation Monitoring From an Unmanned Aerial Vehicle.* IEEE Transactions on Geoscience and Remote Sensing, 47(3).
- Jolivot et al. (2017). *Calibration radiométrique d'images thermiques acquises par drone.* Revue Française de Photogrammétrie et Télédétection.
- Drogue et al. (2020). *Cartographie thermique par drone en milieu urbain.* Colloque AIC, Metz.

---

## Notes

- Les DN bruts sont extraits via `dji_irp.exe -a extract` (mode non documenté officiellement, validé expérimentalement — uint16, plage typique 13 000–22 000).
- Les données de plaques et de la tour à flux sont actuellement simulées (valeurs réalistes pour Niakhar). Elles seront remplacées par les mesures terrain lors des campagnes de vol.
- Le coefficient d'atténuation `k` dans Beer-Lambert est une approximation empirique pour la bande 7.5–13 µm en conditions sahéliennes. À affiner avec les données réelles.

# DJI Tools and Stuff

This project aims to consolidate various DJI-related tools, including thermal conversion tools, geotagged frame extraction from video, and more. If you're passionate about DJI technology and interested in contributing, your help is more than welcome!

## Features

- **Thermal Conversion Tools:** Convert thermal imagery captured by DJI devices into various formats.
- **Geotagged Frame Extraction:** Extract frames from video files with geotag information for mapping applications.
- **Altitude Offset Adjustment Tool:** Enables setting an offset to the altitude of images, useful for correcting altitude discrepancies in geotagged photos or applying uniform adjustments across a batch of images.
- **New Tools Coming Soon!!**

## DJI Image Processor .EXE

For user convenience, we provide a packaged executable named `DJI_Image_Processor`. This executable includes the entire suite of DJI Tools and Stuff, making it easier for users to utilize the various functionalities without the need to set up dependencies manually, check out the installation steps for the .exe below!

## NEW VERSION!!

### Version 1.4 just released!
Added descriptions of inputs and parameters,
Fixed many bugs, 
Improved Speed and usability.

## How to Contribute

We welcome contributions from the community to enhance the functionality and features of this project. If you're interested in contributing, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and ensure the code is properly formatted.
4. Test your changes to make sure they work as expected.
5. Commit your changes and create a pull request.

## Contact Information

For any questions, suggestions, or contributions, feel free to reach out:

**Name:** Miro Rava  
**Contact:** [contact@miro-rava.com](mailto:contact@miro-rava.com)  
**Website:** [www.miro-rava.com](http://www.miro-rava.com)

## Getting Started

To get started with DJI Tools and Stuff, follow the steps below:

### Installation from Github

Preliminary Steps:
1. Download FFmpeg as a local executable file from [FFmpeg's official website](https://ffmpeg.org/download.html).
2. Place the downloaded `ffmpeg.exe` file in the main folder of the project.

Installation Steps:
1. Clone the repository: `git clone https://github.com/MiroRavaProj/DJI-Tools-and-Stuff.git`
2. Navigate to the project directory: `cd DJI-Tools-and-Stuff`
3. Open the folder with the source code and put the source code in the main folder

### Running the Program

To run the program, start `DJI_Image_Processor.pyw` by double-clicking on the file or executing it from a command line interface.

### Installation with .EXE file

1. Download the .exe release zipped package 
2. Extract the zip file (self extracting .exe)
3. Simply click the DJI_Image_Processor.exe file!

#### Disclaimer: If the .exe doesn't work (or the .exe file is not downloaded) is probably because your antivirus detected the file as malicious, since it is a side project there isn't an associated certificate with this .exe file.


## License

This project is licensed under the General Public License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Thanks to Phil Harvey for Exiftool!

Enjoy using DJI Tools and Stuff! If you find any issues or have suggestions, please open an issue or reach out through the provided contact information. Contributions are highly appreciated.
