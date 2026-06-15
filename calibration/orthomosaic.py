"""
orthomosaic.py — Carte thermique multi-images d'un vol DJI Mavic 3T.

Pour chaque image R-JPEG du dossier de vol :
  1. Extraction DN via dji_irp.exe
  2. Carte d'émissivité par seuillage DN (sol / végétation)
  3. Calibration radiométrique + inversion physique → T surface (°C)
  4. Assemblage en mosaïque simple (grille)

Sorties dans output_dir/ :
  - T_<nom_image>.png       : carte thermique par image
  - epsilon_<nom>.png       : carte d'émissivité par image
  - mosaic_thermique.png    : mosaïque thermique générale
  - metriques.csv           : RMSE calibration + stats T par image
  - rapport_global.png      : histogramme + boxplot températures global

Usage :
  python calibration/orthomosaic.py
  ou importer build_thermal_mosaic() depuis une interface.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── Import modules locaux ─────────────────────────────────────────────────
# On suppose que orthomosaic.py est dans le même dossier que calibrate.py
sys.path.insert(0, os.path.dirname(__file__))
from calibrate import (
    beer_lambert, compute_L_up, calibrate_dn_to_L,
    compute_L_theo, SIGMA
)
from extract_dn import extract_dn


# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION — à adapter selon le vol
# ══════════════════════════════════════════════════════════════════════════

VOL_DIR = r"C:\Users\user\Downloads\niakhar\Data_DJI\Images thermiques\DJI_202604251331_001_ThermiqueEPT"
PLAQUES_CSV   = "calibration/data/plaques.csv"
TOUR_FLUX_CSV = "calibration/data/tour_flux.csv"
OUTPUT_DIR    = "calibration/orthomosaic_output"

# Altitude simulée si absente des EXIF (le prof a dit qu'on ne la connaît pas)
ALTITUDE_SIMULE_M = 100.0

# Seuillage DN pour carte d'émissivité
# DN en dessous de DN_SEUIL_VEG → végétation (zones froides) → ε = 0.97
# DN au dessus                   → sol nu                    → ε = 0.95
DN_SEUIL_VEG = 16500   # à ajuster selon l'histogramme réel des images
EPSILON_SOL = 0.95
EPSILON_VEG = 0.97


# ══════════════════════════════════════════════════════════════════════════
# CARTE D'ÉMISSIVITÉ PAR SEUILLAGE DN
# ══════════════════════════════════════════════════════════════════════════

def build_epsilon_map(dn: np.ndarray,
                      seuil: int = DN_SEUIL_VEG,
                      eps_sol: float = EPSILON_SOL,
                      eps_veg: float = EPSILON_VEG) -> np.ndarray:
    """
    Carte d'émissivité pixel par pixel par seuillage DN.

    Principe :
      DN < seuil  → zone froide → végétation → ε = eps_veg (0.97)
      DN >= seuil → zone chaude → sol nu     → ε = eps_sol (0.95)

    Justification : en thermique, la végétation apparaît plus froide
    que le sol nu à Niakhar en milieu de journée. Le seuil est un
    compromis empirique — à affiner avec les vraies images terrain.

    Returns:
        epsilon_map : np.ndarray float64, même shape que dn
    """
    epsilon_map = np.where(dn < seuil, eps_veg, eps_sol).astype(np.float64)
    return epsilon_map


# ══════════════════════════════════════════════════════════════════════════
# INVERSION PHYSIQUE AVEC CARTE ε VARIABLE
# ══════════════════════════════════════════════════════════════════════════

def invert_to_temperature_variable_eps(
        dn: np.ndarray,
        a: float, b: float,
        tau: float, L_down: float, L_up: float,
        epsilon_map: np.ndarray) -> np.ndarray:
    """
    Inversion physique pixel par pixel avec ε variable (carte).

    Équation de transfert radiatif inversée (Berni 2009) :
        L_sensor = a*DN + b
        B(Ts) = [(L_sensor - L↑) / τ - (1-ε)*L↓] / ε
        T_surface = (B(Ts)/σ)^0.25 - 273.15

    Args:
        dn          : matrice DN bruts (H, W) uint16
        a, b        : coefficients régression DN→L
        tau         : transmittance atmosphérique
        L_down      : luminance atmosphérique descendante [W/m²]
        L_up        : luminance atmosphérique montante [W/m²]
        epsilon_map : carte d'émissivité (H, W) float64, même shape que dn

    Returns:
        T_surface : np.ndarray float64 en °C, même shape que dn
    """
    L_sensor  = a * dn.astype(np.float64) + b
    eps       = epsilon_map

    B_surface = ((L_sensor - L_up) / tau - (1.0 - eps) * L_down) / eps
    B_surface = np.clip(B_surface, 0.0, None)

    T_surface = (B_surface / SIGMA) ** 0.25 - 273.15
    return T_surface


# ══════════════════════════════════════════════════════════════════════════
# CALIBRATION D'UNE SEULE IMAGE — renvoie T + métriques
# ══════════════════════════════════════════════════════════════════════════

def process_single_image(img_path: str,
                         plaques: pd.DataFrame,
                         T_air: float, HR: float,
                         L_down: float, altitude_m: float
                         ) -> dict:
    """
    Pipeline complet sur une image R-JPEG :
      extract_dn → epsilon_map → calibrate → invert → métriques

    Returns dict avec clés :
        'name', 'T_image', 'dn', 'epsilon_map',
        'a', 'b', 'rmse_calib', 'tau',
        'T_mean', 'T_min', 'T_max', 'T_std'
    """
    name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\n  ▶ {name}")

    # 1. Extraction DN
    dn = extract_dn(img_path)
    print(f"     DN  min={dn.min()} max={dn.max()} moy={dn.mean():.0f}")

    # 2. Paramètres atmosphériques
    tau  = beer_lambert(altitude_m, T_air, HR)
    L_up = compute_L_up(T_air, tau)

    # 3. Régression DN → L (calibration sur les plaques)
    from calibrate import compute_L_theo
    L_theo = compute_L_theo(plaques, tau, L_down, L_up)
    plaques_tmp = plaques.copy()
    plaques_tmp['L_theo'] = L_theo
    a, b = np.polyfit(plaques_tmp['DN_moyen'], plaques_tmp['L_theo'], 1)

    L_pred = a * plaques_tmp['DN_moyen'] + b
    rmse_calib = float(np.sqrt(np.mean((plaques_tmp['L_theo'] - L_pred) ** 2)))
    print(f"     L = {a:.6f}×DN + {b:.4f}  |  RMSE_calib = {rmse_calib:.4f} W/m²")

    # 4. Carte d'émissivité par seuillage
    epsilon_map = build_epsilon_map(dn)
    pct_veg = 100.0 * np.mean(epsilon_map == EPSILON_VEG)
    print(f"     ε map : {pct_veg:.1f}% végétation / {100-pct_veg:.1f}% sol")

    # 5. Inversion physique → T surface
    T_image = invert_to_temperature_variable_eps(
        dn, a, b, tau, L_down, L_up, epsilon_map
    )
    print(f"     T surface  min={T_image.min():.1f}°C  max={T_image.max():.1f}°C  "
          f"moy={T_image.mean():.1f}°C  std={T_image.std():.1f}°C")

    return {
        'name'       : name,
        'T_image'    : T_image,
        'dn'         : dn,
        'epsilon_map': epsilon_map,
        'a'          : a,
        'b'          : b,
        'tau'        : tau,
        'rmse_calib' : rmse_calib,
        'T_mean'     : float(T_image.mean()),
        'T_min'      : float(T_image.min()),
        'T_max'      : float(T_image.max()),
        'T_std'      : float(T_image.std()),
    }


# ══════════════════════════════════════════════════════════════════════════
# VISUALISATION PAR IMAGE
# ══════════════════════════════════════════════════════════════════════════

def save_image_figures(result: dict, output_dir: str,
                       vmin: float = 20.0, vmax: float = 60.0):
    """
    Sauvegarde pour une image :
      - carte thermique T (PNG)
      - carte d'émissivité ε (PNG)
    """
    name = result['name']
    T    = result['T_image']
    eps  = result['epsilon_map']
    dn   = result['dn']

    # ── Carte thermique ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Calibration thermique — {name}", fontsize=11, y=1.01)

    axes[0].imshow(dn, cmap='hot')
    axes[0].set_title('DN bruts')
    axes[0].axis('off')

    im = axes[1].imshow(T, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'T surface calibrée (°C)\n'
                      f'moy={result["T_mean"]:.1f}°C  '
                      f'RMSE_cal={result["rmse_calib"]:.3f} W/m²')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], label='°C', fraction=0.046)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"T_{name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Carte d'émissivité ────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    cmap_eps = plt.cm.RdYlGn
    im2 = ax2.imshow(eps, cmap=cmap_eps,
                     vmin=EPSILON_VEG - 0.01, vmax=EPSILON_SOL + 0.01)
    ax2.set_title(f'Carte d\'émissivité ε — {name}\n'
                  f'Vert=végétation (ε={EPSILON_VEG}) | Rouge=sol (ε={EPSILON_SOL})')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, label='ε', fraction=0.046)
    plt.tight_layout()
    out_eps = os.path.join(output_dir, f"epsilon_{name}.png")
    plt.savefig(out_eps, dpi=150, bbox_inches='tight')
    plt.close(fig2)

    print(f"     → Sauvegardé : T_{name}.png  |  epsilon_{name}.png")


# ══════════════════════════════════════════════════════════════════════════
# MOSAÏQUE THERMIQUE (grille d'images)
# ══════════════════════════════════════════════════════════════════════════

def build_mosaic(results: list[dict], output_dir: str,
                 vmin: float = 20.0, vmax: float = 60.0):
    """
    Assemble toutes les cartes T en une grille visuelle (mosaïque).

    Note : c'est une juxtaposition visuelle, PAS une orthomosaïque
    géoréférencée (ça nécessiterait Metashape/ODM + coordonnées GPS).
    Les images sont disposées dans l'ordre alphabétique du nom de fichier.
    """
    n = len(results)
    if n == 0:
        return

    # Grille la plus carrée possible
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 3.5))
    fig.suptitle(
        f'Mosaïque thermique — Vol DJI Mavic 3T — Niakhar\n'
        f'{n} image(s) calibrée(s)  |  altitude simulée {ALTITUDE_SIMULE_M}m',
        fontsize=12
    )

    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlBu_r

    for i, res in enumerate(results):
        ax = axes_flat[i]
        ax.imshow(res['T_image'], cmap=cmap, norm=norm)
        # Titre court : timestamp extrait du nom
        short = res['name'].replace('DJI_', '').split('_T')[0]
        ax.set_title(f"{short}\n{res['T_mean']:.1f}°C moy", fontsize=7)
        ax.axis('off')

    # Masquer les axes vides
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis('off')

    # Colorbar globale
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes_flat[:n].tolist() if n > 1 else axes_flat[0],
                 label='Température de surface (°C)',
                 fraction=0.02, pad=0.04)

    plt.tight_layout()
    out = os.path.join(output_dir, "mosaic_thermique.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  → Mosaïque : {out}")


# ══════════════════════════════════════════════════════════════════════════
# MÉTRIQUES GLOBALES + RAPPORT VISUEL
# ══════════════════════════════════════════════════════════════════════════

def save_metrics(results: list[dict], output_dir: str):
    """Exporte metriques.csv avec une ligne par image."""
    rows = []
    for r in results:
        rows.append({
            'image'       : r['name'],
            'tau'         : round(r['tau'], 4),
            'a_coeff'     : round(r['a'], 8),
            'b_offset'    : round(r['b'], 4),
            'rmse_calib_Wm2': round(r['rmse_calib'], 6),
            'T_min_C'     : round(r['T_min'], 2),
            'T_max_C'     : round(r['T_max'], 2),
            'T_mean_C'    : round(r['T_mean'], 2),
            'T_std_C'     : round(r['T_std'], 2),
        })
    df = pd.DataFrame(rows)
    out = os.path.join(output_dir, "metriques.csv")
    df.to_csv(out, index=False)
    print(f"  → Métriques : {out}")
    return df


def save_global_report(results: list[dict], df_metrics: pd.DataFrame,
                       output_dir: str):
    """
    Rapport visuel global :
      - Histogramme des T de toutes les images empilées
      - Boxplot T par image
      - RMSE calibration par image (bar chart)
      - Évolution T_mean par image
    """
    all_T = np.concatenate([r['T_image'].flatten() for r in results])
    names_short = [r['name'].replace('DJI_', '').split('_T')[0]
                   for r in results]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Rapport global — Calibration thermique Niakhar', fontsize=13)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Histogramme global des températures ────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(all_T, bins=80, color='#e07b39', edgecolor='white', linewidth=0.3)
    ax1.set_xlabel('Température de surface (°C)')
    ax1.set_ylabel('Nombre de pixels')
    ax1.set_title(f'Distribution globale T surface\n'
                  f'({len(results)} image(s) — {all_T.size:,} pixels)')
    ax1.axvline(all_T.mean(), color='navy', lw=1.5,
                label=f'Moy = {all_T.mean():.1f}°C')
    ax1.axvline(np.median(all_T), color='crimson', lw=1.5, ls='--',
                label=f'Médiane = {np.median(all_T):.1f}°C')
    ax1.legend(fontsize=8)

    # ── 2. Boxplot T par image ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    data_box = [r['T_image'].flatten() for r in results]
    bp = ax2.boxplot(data_box, patch_artist=True, notch=False,
                     medianprops=dict(color='navy', linewidth=1.5))
    for patch in bp['boxes']:
        patch.set_facecolor('#6ab0de')
        patch.set_alpha(0.7)
    ax2.set_xticks(range(1, len(results) + 1))
    ax2.set_xticklabels(names_short, rotation=45, ha='right', fontsize=7)
    ax2.set_ylabel('Température (°C)')
    ax2.set_title('Distribution T par image')

    # ── 3. RMSE calibration par image ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    colors_rmse = ['#d62728' if v > 1.0 else '#2ca02c'
                   for v in df_metrics['rmse_calib_Wm2']]
    ax3.bar(names_short, df_metrics['rmse_calib_Wm2'],
            color=colors_rmse, edgecolor='white')
    ax3.axhline(1.0, color='gray', ls='--', lw=1,
                label='Seuil qualité 1.0 W/m²')
    ax3.set_ylabel('RMSE calibration (W/m²)')
    ax3.set_title('RMSE régression DN→L par image\n(vert < 1 W/m² : bon)')
    ax3.set_xticklabels(names_short, rotation=45, ha='right', fontsize=7)
    ax3.legend(fontsize=8)

    # ── 4. T_mean par image (évolution temporelle approximative) ──────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(range(len(results)), df_metrics['T_mean_C'],
             'o-', color='#e07b39', lw=2, ms=6, label='T_mean')
    ax4.fill_between(
        range(len(results)),
        df_metrics['T_mean_C'] - df_metrics['T_std_C'],
        df_metrics['T_mean_C'] + df_metrics['T_std_C'],
        alpha=0.2, color='#e07b39', label='±1σ'
    )
    ax4.set_xticks(range(len(results)))
    ax4.set_xticklabels(names_short, rotation=45, ha='right', fontsize=7)
    ax4.set_ylabel('Température moyenne (°C)')
    ax4.set_title('Évolution T_mean par image\n(ordre alphabétique ≈ ordre temporel)')
    ax4.legend(fontsize=8)

    out = os.path.join(output_dir, "rapport_global.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Rapport global : {out}")


# ══════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE PRINCIPAL — aussi importable comme fonction
# ══════════════════════════════════════════════════════════════════════════

def build_thermal_mosaic(
        vol_dir: str = VOL_DIR,
        plaques_csv: str = PLAQUES_CSV,
        tour_flux_csv: str = TOUR_FLUX_CSV,
        output_dir: str = OUTPUT_DIR,
        altitude_m: float = ALTITUDE_SIMULE_M,
        dn_seuil_veg: int = DN_SEUIL_VEG,
        vmin_viz: float = 20.0,
        vmax_viz: float = 60.0,
        max_images: int = None,   # None = toutes les images
):
    """
    Pipeline orthomosaïque thermique complet.

    Args:
        vol_dir      : dossier contenant les .JPG R-JPEG du vol
        plaques_csv  : chemin vers plaques.csv
        tour_flux_csv: chemin vers tour_flux.csv
        output_dir   : dossier de sortie (créé si absent)
        altitude_m   : altitude de vol simulée (m AGL)
        dn_seuil_veg : seuil DN pour distinguer sol / végétation
        vmin_viz     : borne basse colormap (°C)
        vmax_viz     : borne haute colormap (°C)
        max_images   : limiter le nombre d'images traitées (debug)

    Returns:
        results    : liste de dicts par image (T_image, métriques, etc.)
        df_metrics : DataFrame métriques exporté
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Chargement des données de référence ───────────────────────────
    plaques   = pd.read_csv(plaques_csv)
    tour_flux = pd.read_csv(tour_flux_csv).iloc[0]

    T_AIR  = float(tour_flux['T_air_celsius'])
    HR     = float(tour_flux['HR_pct'])
    L_DOWN = float(tour_flux['L_down_atm'])

    print("═" * 60)
    print("  Pipeline thermique Niakhar — DJI Mavic 3T")
    print("═" * 60)
    print(f"  Dossier vol   : {vol_dir}")
    print(f"  Altitude sim. : {altitude_m} m")
    print(f"  Tour à flux   : T_air={T_AIR}°C  HR={HR}%  L↓={L_DOWN} W/m²")
    print(f"  Sortie        : {output_dir}")

    # ── Listing images ─────────────────────────────────────────────────
    jpgs = sorted(glob.glob(os.path.join(vol_dir, "*_T.JPG")))
    if not jpgs:
        # Fallback : tous les .JPG du dossier
        jpgs = sorted(glob.glob(os.path.join(vol_dir, "*.JPG")))
    if not jpgs:
        raise FileNotFoundError(
            f"Aucun fichier .JPG trouvé dans : {vol_dir}\n"
            "Vérifier le chemin VOL_DIR dans la configuration."
        )
    if max_images:
        jpgs = jpgs[:max_images]

    print(f"\n  {len(jpgs)} image(s) R-JPEG trouvée(s)\n")

    # ── Traitement image par image ─────────────────────────────────────
    results = []
    for img_path in jpgs:
        try:
            res = process_single_image(
                img_path, plaques, T_AIR, HR, L_DOWN, altitude_m
            )
            results.append(res)
            save_image_figures(res, output_dir, vmin=vmin_viz, vmax=vmax_viz)
        except Exception as e:
            print(f"  ✗ Erreur sur {os.path.basename(img_path)} : {e}")
            continue

    if not results:
        print("\n  Aucune image traitée avec succès.")
        return [], pd.DataFrame()

    # ── Mosaïque + métriques + rapport ────────────────────────────────
    print("\n── Assemblage mosaïque ──────────────────────────────────────")
    build_mosaic(results, output_dir, vmin=vmin_viz, vmax=vmax_viz)

    print("\n── Métriques & rapport ──────────────────────────────────────")
    df_metrics = save_metrics(results, output_dir)
    save_global_report(results, df_metrics, output_dir)

    # ── Résumé console ─────────────────────────────────────────────────
    all_T = np.concatenate([r['T_image'].flatten() for r in results])
    print("\n══ Résumé global ════════════════════════════════════════════")
    print(f"  Images traitées  : {len(results)} / {len(jpgs)}")
    print(f"  T surface globale: moy={all_T.mean():.1f}°C  "
          f"min={all_T.min():.1f}°C  max={all_T.max():.1f}°C")
    print(f"  RMSE calib moy   : {df_metrics['rmse_calib_Wm2'].mean():.4f} W/m²")
    print(f"  Sorties dans     : {os.path.abspath(output_dir)}")
    print("═" * 60)

    return results, df_metrics


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    build_thermal_mosaic()