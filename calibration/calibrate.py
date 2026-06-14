"""
calibrate.py — Pipeline de calibration radiométrique thermique drone.
Régression DN → L_sensor sur plaques étalons, puis inversion physique (Berni 2009).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from extract_dn import extract_dn

SIGMA = 5.67e-8  # Stefan-Boltzmann [W/m²/K⁴]


# ── Fonctions physiques ───────────────────────────────────────────────────

def temp_to_radiance(T_celsius: float) -> float:
    """Stefan-Boltzmann : T [°C] → B(Ts) [W/m²]."""
    return SIGMA * (T_celsius + 273.15) ** 4


def beer_lambert(z: float, T_air: float, HR: float) -> float:
    """
    Transmittance atmosphérique τ = exp(-k*z).
    k empirique pour bande 7.5-13µm en conditions sahéliennes.
    """
    k = 0.0002 * (HR / 100) * (1 + 0.02 * T_air)
    return np.exp(-k * z)


def compute_L_up(T_air: float, tau: float) -> float:
    """L↑_atm : émission propre de l'atmosphère vers le capteur."""
    return (1 - tau) * SIGMA * (T_air + 273.15) ** 4


# ── Étape 1 : calculer L_theo pour chaque plaque ─────────────────────────

def compute_L_theo(plaques: pd.DataFrame,
                    tau: float, L_down: float, L_up: float) -> pd.Series:
    """
    Pour chaque plaque étalon, calcule la luminance théorique reçue par le capteur :
    L_theo = τ * [ε*B(Ts) + (1-ε)*L↓] + L↑
    """
    B_ts = plaques['T_ref_celsius'].apply(temp_to_radiance)
    eps  = plaques['emissivite']
    L_theo = tau * (eps * B_ts + (1 - eps) * L_down) + L_up
    return L_theo


# ── Étape 2 : régression DN → L_sensor ───────────────────────────────────

def calibrate_dn_to_L(plaques: pd.DataFrame,
                        tau: float, L_down: float, L_up: float):
    """
    Régression linéaire sur paires (DN_moyen, L_theo).
    Retourne coefficients (a, b) tels que L_sensor = a*DN + b.
    """
    L_theo = compute_L_theo(plaques, tau, L_down, L_up)
    plaques = plaques.copy()
    plaques['L_theo'] = L_theo

    print("\n── Plaques étalons ──────────────────────────────────────────")
    print(plaques[['plaque', 'DN_moyen', 'T_ref_celsius', 'emissivite', 'L_theo']].to_string(index=False))

    a, b = np.polyfit(plaques['DN_moyen'], plaques['L_theo'], 1)

    L_pred = a * plaques['DN_moyen'] + b
    rmse   = np.sqrt(np.mean((plaques['L_theo'] - L_pred) ** 2))

    print(f"\nRégression DN→L : L = {a:.6f} × DN + {b:.4f} W/m²")
    print(f"RMSE calibration : {rmse:.6f} W/m²")
    return a, b


# ── Étape 3 : inversion physique pixel par pixel ──────────────────────────

def invert_to_temperature(dn: np.ndarray, a: float, b: float,
                            tau: float, L_down: float, L_up: float,
                            emissivite: float = 0.95) -> np.ndarray:
    """
    Inversion complète (Berni 2009) :
    L_sensor = a*DN + b
    B(Ts) = [(L_sensor - L↑) / τ - (1-ε)*L↓] / ε
    T_surface = (B(Ts)/σ)^0.25 - 273.15
    """
    L_sensor = a * dn.astype(float) + b

    eps      = emissivite
    B_surface = ((L_sensor - L_up) / tau - (1 - eps) * L_down) / eps
    B_surface = np.clip(B_surface, 0, None)

    T_surface = (B_surface / SIGMA) ** 0.25 - 273.15
    return T_surface


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    IMG_PATH      = r"Data_DJI\Images thermiques\DJI_202604251331_001_ThermiqueEPT\DJI_20260425140653_0925_T.JPG"
    PLAQUES_CSV   = "calibration/data/plaques.csv"
    TOUR_FLUX_CSV = "calibration/data/tour_flux.csv"

    # Chargement données
    plaques   = pd.read_csv(PLAQUES_CSV)
    tour_flux = pd.read_csv(TOUR_FLUX_CSV).iloc[0]  # une seule ligne

    T_AIR  = tour_flux['T_air_celsius']
    HR     = tour_flux['HR_pct']
    L_DOWN = tour_flux['L_down_atm']
    Z_VOL  = tour_flux['altitude_vol_m']

    # Paramètres atmosphériques
    tau   = beer_lambert(Z_VOL, T_AIR, HR)
    L_up  = compute_L_up(T_AIR, tau)

    print(f"Atmosphère simulée (tour à flux) :")
    print(f"  τ(z={Z_VOL}m, T={T_AIR}°C, HR={HR}%) = {tau:.4f}")
    print(f"  L↑_atm = {L_up:.2f} W/m²")
    print(f"  L↓_atm = {L_DOWN:.2f} W/m²")

    # Extraction DN
    print("\nExtraction DN...")
    dn = extract_dn(IMG_PATH)
    print(f"DN — min: {dn.min()} | max: {dn.max()} | moy: {dn.mean():.1f}")

    # Régression DN → L
    a, b = calibrate_dn_to_L(plaques, tau, L_DOWN, L_up)

    # Inversion physique → T surface
    print("\nInversion physique...")
    T_image = invert_to_temperature(dn, a, b, tau, L_DOWN, L_up)
    print(f"T surface — min: {T_image.min():.1f}°C | max: {T_image.max():.1f}°C | moy: {T_image.mean():.1f}°C")

    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(dn, cmap='hot')
    axes[0].set_title('DN bruts')
    axes[0].axis('off')

    im = axes[1].imshow(T_image, cmap='RdYlBu_r', vmin=20, vmax=60)
    axes[1].set_title('T surface calibrée (°C)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], label='°C')

    plt.tight_layout()
    plt.savefig("calibration/T_calibree.png", dpi=150)
    plt.show()