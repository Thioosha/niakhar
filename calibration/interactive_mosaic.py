"""
interactive_mosaic.py — Visualisation interactive de la carte thermique drone.

Génère un fichier HTML standalone (mosaic_interactive.html) avec :
  - Grille d'images thermiques cliquables
  - Hover pixel par pixel → affiche la température en °C
  - Histogramme global des températures
  - Stats par image au clic

Dépendances : pip install plotly numpy pandas
(pas besoin de matplotlib pour ce module)

Usage :
    python calibration/interactive_mosaic.py
    → ouvre mosaic_interactive.html dans le navigateur

Ou depuis une interface :
    from interactive_mosaic import build_interactive_html
    html_path = build_interactive_html(results, output_dir)
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd

# ── Import pipeline existant ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from calibrate import beer_lambert, compute_L_up, compute_L_theo, SIGMA
from extract_dn import extract_dn


# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

VOL_DIR       = r"C:\Users\user\Downloads\niakhar\Data_DJI\Images thermiques\DJI_202604251331_001_ThermiqueEPT"
PLAQUES_CSV   = "calibration/data/plaques.csv"
TOUR_FLUX_CSV = "calibration/data/tour_flux.csv"
OUTPUT_DIR    = "calibration/orthomosaic_output"
ALTITUDE_M    = 100.0

# Seuillage émissivité
DN_SEUIL_VEG  = 16500
EPSILON_SOL   = 0.95
EPSILON_VEG   = 0.97

# Masque ciel : DN en dessous de ce seuil = pixels ciel → ignorés (NaN)
# Le ciel est très froid → DN très bas → après inversion T explose vers des valeurs aberrantes
# Valeur typique : ~13000-14000 pour le ciel à Niakhar
DN_MIN_SOL    = 14500   # pixels en dessous → considérés ciel → NaN

# Limites colormap pour la visualisation
VMIN_C = 20.0
VMAX_C = 60.0

# Pour la démo/soutenance : limiter le nombre d'images dans la visu interactive
# (977 images = HTML de 500Mo, inutilisable — on prend un sous-ensemble représentatif)
MAX_IMAGES_INTERACTIF = 25   # None = toutes


# ══════════════════════════════════════════════════════════════════════════
# PIPELINE PAR IMAGE (copie légère de orthomosaic.py avec fix masque ciel)
# ══════════════════════════════════════════════════════════════════════════

def build_epsilon_map(dn: np.ndarray) -> np.ndarray:
    """Carte ε par seuillage DN : végétation (froid) vs sol (chaud)."""
    return np.where(dn < DN_SEUIL_VEG, EPSILON_VEG, EPSILON_SOL).astype(np.float64)


def process_image(img_path: str, plaques: pd.DataFrame,
                  T_air: float, HR: float, L_down: float,
                  altitude_m: float) -> dict | None:
    """
    Pipeline complet sur une image R-JPEG avec masque ciel.

    Le masque ciel (DN < DN_MIN_SOL) met les pixels à NaN avant inversion
    pour éviter les températures aberrantes (T > 80°C) sur les pixels de ciel.

    Returns dict avec T_image (float32, NaN sur ciel), DN, métriques.
    """
    name = os.path.splitext(os.path.basename(img_path))[0]

    try:
        dn = extract_dn(img_path)
    except Exception as e:
        print(f"  ✗ {name} : extraction DN échouée — {e}")
        return None

    # Paramètres atmosphériques
    tau  = beer_lambert(altitude_m, T_air, HR)
    L_up = compute_L_up(T_air, tau)

    # Régression DN → L sur les plaques
    L_theo = compute_L_theo(plaques, tau, L_down, L_up)
    plaques_tmp = plaques.copy()
    plaques_tmp['L_theo'] = L_theo
    a, b = np.polyfit(plaques_tmp['DN_moyen'], plaques_tmp['L_theo'], 1)

    L_pred = a * plaques_tmp['DN_moyen'] + b
    rmse_calib = float(np.sqrt(np.mean((plaques_tmp['L_theo'] - L_pred) ** 2)))

    # Carte émissivité
    epsilon_map = build_epsilon_map(dn)

    # Inversion physique avec masque ciel
    dn_f = dn.astype(np.float64)

    # Masque : pixels de ciel mis à NaN AVANT l'inversion
    # → évite les T < 0°C ou > 80°C sur les bords/ciel
    sky_mask = dn < DN_MIN_SOL

    L_sensor  = a * dn_f + b
    eps       = epsilon_map
    B_surface = ((L_sensor - L_up) / tau - (1.0 - eps) * L_down) / eps
    B_surface = np.clip(B_surface, 0.0, None)
    T_surface = (B_surface / SIGMA) ** 0.25 - 273.15

    # Appliquer le masque : ciel → NaN
    T_surface = T_surface.astype(np.float32)
    T_surface[sky_mask] = np.nan

    # Stats en ignorant NaN
    valid = T_surface[~np.isnan(T_surface)]
    pct_veg = 100.0 * np.mean(epsilon_map[~sky_mask] == EPSILON_VEG) if valid.size > 0 else 0.0

    print(f"  ▶ {name}  |  T: {np.nanmean(T_surface):.1f}°C moy  "
          f"[{np.nanmin(T_surface):.1f} – {np.nanmax(T_surface):.1f}]  "
          f"|  ciel masqué: {sky_mask.sum()} px  |  RMSE_cal={rmse_calib:.3f} W/m²")

    return {
        'name'       : name,
        'T_image'    : T_surface,         # float32, NaN sur ciel
        'dn'         : dn,
        'epsilon_map': epsilon_map,
        'sky_mask'   : sky_mask,
        'a'          : a,
        'b'          : b,
        'tau'        : tau,
        'rmse_calib' : rmse_calib,
        'T_mean'     : float(np.nanmean(T_surface)),
        'T_min'      : float(np.nanmin(T_surface)) if valid.size > 0 else float('nan'),
        'T_max'      : float(np.nanmax(T_surface)) if valid.size > 0 else float('nan'),
        'T_std'      : float(np.nanstd(T_surface)) if valid.size > 0 else float('nan'),
        'pct_veg'    : pct_veg,
    }


# ══════════════════════════════════════════════════════════════════════════
# GÉNÉRATION HTML INTERACTIF (Plotly CDN — standalone)
# ══════════════════════════════════════════════════════════════════════════

def _T_to_rgba_b64(T: np.ndarray, vmin: float, vmax: float) -> str:
    """
    Convertit une matrice T (float32, NaN=ciel) en image PNG base64
    avec colormap RdYlBu_r pour l'encodage dans le HTML.
    Nécessite matplotlib uniquement pour cette conversion.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import io, base64

    cmap = plt.cm.RdYlBu_r
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # Mapper T → RGBA (NaN → transparent)
    rgba = cmap(norm(np.nan_to_num(T, nan=vmin - 1)))
    # Mettre alpha=0 sur les pixels NaN (ciel)
    rgba[np.isnan(T), 3] = 0.0
    rgba = (rgba * 255).astype(np.uint8)

    # Encoder en PNG base64
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgba, mode='RGBA').save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def build_interactive_html(results: list[dict],
                            output_dir: str,
                            vmin: float = VMIN_C,
                            vmax: float = VMAX_C,
                            max_images: int = MAX_IMAGES_INTERACTIF) -> str:
    """
    Génère un HTML interactif standalone avec :
      - Sélecteur d'image (dropdown)
      - Carte thermique avec hover pixel → température exacte
      - Panneau stats (T_mean, T_min, T_max, RMSE_cal, % végétation)
      - Histogramme de la distribution T de l'image sélectionnée
      - Colorbar fixe

    Args:
        results    : liste de dicts depuis process_image()
        output_dir : dossier de sortie
        vmin/vmax  : bornes colormap
        max_images : sous-ensemble pour ne pas exploser la taille du HTML

    Returns:
        chemin vers le fichier HTML généré
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sous-ensemble si besoin
    subset = results[:max_images] if max_images else results
    n = len(subset)
    print(f"\n  Génération HTML interactif ({n} images)...")

    # Préparer les données JSON pour le JS
    images_data = []
    for i, r in enumerate(subset):
        T = r['T_image']  # float32, NaN = ciel
        H, W = T.shape

        # Arrondir à 1 décimale pour réduire la taille JSON
        # Remplacer NaN par null (JSON-safe)
        T_list = []
        for row in T:
            T_list.append([None if np.isnan(v) else round(float(v), 1) for v in row])

        # Image colorisée en base64 pour l'affichage rapide
        b64 = _T_to_rgba_b64(T, vmin, vmax)

        images_data.append({
            'name'      : r['name'],
            'short'     : r['name'].replace('DJI_', '').split('_T')[0],
            'T_mean'    : round(r['T_mean'], 1),
            'T_min'     : round(r['T_min'], 1) if not np.isnan(r['T_min']) else None,
            'T_max'     : round(r['T_max'], 1) if not np.isnan(r['T_max']) else None,
            'T_std'     : round(r['T_std'], 1) if not np.isnan(r['T_std']) else None,
            'rmse_calib': round(r['rmse_calib'], 3),
            'tau'       : round(r['tau'], 4),
            'pct_veg'   : round(r['pct_veg'], 1),
            'width'     : W,
            'height'    : H,
            'T_data'    : T_list,
            'img_b64'   : b64,
        })
        print(f"    [{i+1}/{n}] {r['name']} encodé")

    # Histogramme global (toutes images du subset)
    all_T_valid = []
    for r in subset:
        v = r['T_image'][~np.isnan(r['T_image'])].flatten()
        all_T_valid.extend(v.tolist())
    hist_counts, hist_edges = np.histogram(all_T_valid, bins=60,
                                           range=(vmin - 5, vmax + 10))
    hist_centers = ((hist_edges[:-1] + hist_edges[1:]) / 2).round(1).tolist()
    hist_counts  = hist_counts.tolist()

    # Sérialiser en JSON (ensure_ascii=True pour éviter tout problème d'encodage)
    json_data = json.dumps(images_data, ensure_ascii=True)
    json_hist = json.dumps({'centers': hist_centers, 'counts': hist_counts})

    # ── Template HTML ──────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Carte thermique interactive — Niakhar</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f1117;
    color: #e8eaf6;
    min-height: 100vh;
  }}
  header {{
    background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
    padding: 16px 28px;
    border-bottom: 1px solid #303F9F;
  }}
  header h1 {{
    font-size: 1.2rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: #e8eaf6;
  }}
  header p {{
    font-size: 0.78rem;
    color: #9fa8da;
    margin-top: 3px;
  }}
  .layout {{
    display: grid;
    grid-template-columns: 300px 1fr;
    grid-template-rows: auto 1fr;
    gap: 0;
    height: calc(100vh - 65px);
  }}
  .sidebar {{
    grid-row: 1 / 3;
    background: #1a1d2e;
    border-right: 1px solid #2a2d3e;
    padding: 18px 16px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 18px;
  }}
  .sidebar label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7986cb;
    display: block;
    margin-bottom: 6px;
  }}
  select {{
    width: 100%;
    background: #252840;
    color: #e8eaf6;
    border: 1px solid #3f51b5;
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 0.82rem;
    cursor: pointer;
    outline: none;
  }}
  select:focus {{ border-color: #7986cb; }}

  .stats-card {{
    background: #252840;
    border-radius: 8px;
    padding: 14px;
    border: 1px solid #303F9F;
  }}
  .stats-card h3 {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7986cb;
    margin-bottom: 10px;
  }}
  .stat-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid #303F9F22;
    font-size: 0.82rem;
  }}
  .stat-row:last-child {{ border-bottom: none; }}
  .stat-label {{ color: #9fa8da; }}
  .stat-val {{ font-weight: 600; color: #e8eaf6; font-variant-numeric: tabular-nums; }}
  .stat-val.hot {{ color: #ef5350; }}
  .stat-val.cool {{ color: #42a5f5; }}
  .stat-val.ok {{ color: #66bb6a; }}
  .stat-val.warn {{ color: #ffa726; }}

  .colorbar-wrap {{
    background: #252840;
    border-radius: 8px;
    padding: 14px;
    border: 1px solid #303F9F;
  }}
  .colorbar-wrap h3 {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7986cb;
    margin-bottom: 10px;
  }}
  #colorbar-svg {{ width: 100%; }}

  .main-panel {{
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    overflow: hidden;
  }}
  .image-title {{
    font-size: 0.85rem;
    color: #9fa8da;
  }}
  .image-title span {{ color: #e8eaf6; font-weight: 600; }}

  #hover-info {{
    background: #252840;
    border: 1px solid #3f51b5;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 0.82rem;
    color: #9fa8da;
    min-height: 32px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  #hover-info .temp-val {{
    font-size: 1.05rem;
    font-weight: 700;
    color: #ffcc02;
    font-variant-numeric: tabular-nums;
  }}

  #plot-thermal {{
    flex: 1;
    min-height: 0;
    border-radius: 8px;
    overflow: hidden;
    background: #1a1d2e;
  }}

  .bottom-panel {{
    background: #1a1d2e;
    border-top: 1px solid #2a2d3e;
    padding: 12px 16px;
    display: flex;
    gap: 16px;
    align-items: stretch;
  }}
  #plot-hist {{
    flex: 1;
    height: 130px;
  }}
  .note {{
    font-size: 0.72rem;
    color: #5c6bc0;
    line-height: 1.5;
    max-width: 220px;
    align-self: center;
  }}
</style>
</head>
<body>

<header>
  <h1>🌡 Carte thermique interactive — Niakhar, Sénégal</h1>
  <p>DJI Mavic 3T · Calibration radiométrique Berni 2009 · Altitude simulée {ALTITUDE_M}m · {n} images</p>
</header>

<div class="layout">

  <!-- ── Sidebar ── -->
  <aside class="sidebar">

    <div>
      <label>Sélectionner une image</label>
      <select id="img-select" onchange="loadImage(this.value)"></select>
    </div>

    <div class="stats-card">
      <h3>Statistiques — image courante</h3>
      <div class="stat-row">
        <span class="stat-label">T moyenne</span>
        <span class="stat-val" id="s-mean">—</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">T minimale (sol)</span>
        <span class="stat-val cool" id="s-min">—</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">T maximale (sol)</span>
        <span class="stat-val hot" id="s-max">—</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Écart-type</span>
        <span class="stat-val" id="s-std">—</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">RMSE calibration</span>
        <span class="stat-val" id="s-rmse">—</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Transmittance τ</span>
        <span class="stat-val" id="s-tau">—</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">% végétation (ε=0.97)</span>
        <span class="stat-val ok" id="s-veg">—</span>
      </div>
    </div>

    <div class="colorbar-wrap">
      <h3>Échelle de température (°C)</h3>
      <svg id="colorbar-svg" height="28" viewBox="0 0 268 28">
        <defs>
          <linearGradient id="cb-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <!-- RdYlBu_r approximation -->
            <stop offset="0%"   stop-color="#313695"/>
            <stop offset="17%"  stop-color="#4575b4"/>
            <stop offset="33%"  stop-color="#74add1"/>
            <stop offset="50%"  stop-color="#ffffbf"/>
            <stop offset="67%"  stop-color="#f46d43"/>
            <stop offset="83%"  stop-color="#d73027"/>
            <stop offset="100%" stop-color="#a50026"/>
          </linearGradient>
        </defs>
        <rect x="0" y="0" width="268" height="16" rx="3" fill="url(#cb-grad)"/>
        <text x="0"   y="26" fill="#9fa8da" font-size="10" text-anchor="start">{vmin:.0f}°C</text>
        <text x="134" y="26" fill="#9fa8da" font-size="10" text-anchor="middle">{(vmin+vmax)/2:.0f}°C</text>
        <text x="268" y="26" fill="#9fa8da" font-size="10" text-anchor="end">{vmax:.0f}°C</text>
      </svg>
    </div>

    <div class="note">
      Les pixels de ciel (DN &lt; {DN_MIN_SOL}) sont masqués (transparent).<br><br>
      Calibration : L = a·DN + b — régression sur 3 plaques étalons.<br><br>
      Inversion : équation de transfert radiatif (Berni 2009).
    </div>

  </aside>

  <!-- ── Main panel ── -->
  <div class="main-panel">
    <div class="image-title">
      Image : <span id="img-label">—</span>
    </div>
    <div id="hover-info">
      ☝ Survoler l'image pour voir la température du pixel
    </div>
    <div id="plot-thermal"></div>
  </div>

  <!-- ── Bottom histogram ── -->
  <div class="bottom-panel">
    <div id="plot-hist"></div>
    <div class="note">
      <strong style="color:#7986cb">Distribution globale</strong><br>
      Toutes les images du subset ({n} images, pixels sol uniquement).<br>
      Valeurs aberrantes masquées (ciel).
    </div>
  </div>

</div>

<script>
// ── Données ──────────────────────────────────────────────────────────────
const IMAGES = __IMAGES_JSON_PLACEHOLDER__;
const GLOBAL_HIST = __HIST_JSON_PLACEHOLDER__;
const VMIN = {vmin};
const VMAX = {vmax};

let currentIdx = 0;

// ── Colormap RdYlBu_r (approximation linéaire par interpolation) ─────────
// Couleurs clés de matplotlib RdYlBu_r
const CMAP_STOPS = [
  [0.000, [49,  54,  149]],
  [0.125, [69,  117, 180]],
  [0.250, [116, 173, 209]],
  [0.375, [171, 217, 233]],
  [0.500, [255, 255, 191]],
  [0.625, [254, 224, 144]],
  [0.750, [253, 174,  97]],
  [0.875, [244, 109,  67]],
  [1.000, [165,   0,  38]],
];

function cmapColor(t) {{
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < CMAP_STOPS.length; i++) {{
    const [t0, c0] = CMAP_STOPS[i-1];
    const [t1, c1] = CMAP_STOPS[i];
    if (t <= t1) {{
      const f = (t - t0) / (t1 - t0);
      return c0.map((v, k) => Math.round(v + f * (c1[k] - v)));
    }}
  }}
  return CMAP_STOPS[CMAP_STOPS.length-1][1];
}}

// ── Init dropdown ────────────────────────────────────────────────────────
function initSelect() {{
  const sel = document.getElementById('img-select');
  IMAGES.forEach((img, i) => {{
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = `${{img.short}} — ${{img.T_mean !== null ? img.T_mean + '°C' : '?'}} moy`;
    sel.appendChild(opt);
  }});
}}

// ── Mise à jour stats sidebar ────────────────────────────────────────────
function updateStats(img) {{
  const rmseOk = img.rmse_calib < 1.0;
  document.getElementById('s-mean').textContent  = img.T_mean !== null  ? img.T_mean + ' °C'  : '—';
  document.getElementById('s-min').textContent   = img.T_min  !== null  ? img.T_min  + ' °C'  : '—';
  document.getElementById('s-max').textContent   = img.T_max  !== null  ? img.T_max  + ' °C'  : '—';
  document.getElementById('s-std').textContent   = img.T_std  !== null  ? '± ' + img.T_std + ' °C' : '—';
  document.getElementById('s-rmse').textContent  = img.rmse_calib + ' W/m²';
  document.getElementById('s-rmse').className    = 'stat-val ' + (rmseOk ? 'ok' : 'warn');
  document.getElementById('s-tau').textContent   = img.tau;
  document.getElementById('s-veg').textContent   = img.pct_veg + '%';
  document.getElementById('img-label').textContent = img.name;
}}

// ── Rendu heatmap Plotly ─────────────────────────────────────────────────
function renderHeatmap(img) {{
  const T = img.T_data;  // Array 2D, null = ciel
  const H = img.height;
  const W = img.width;

  // Construire z (valeurs) et text (labels hover)
  const z    = [];
  const text = [];
  for (let r = 0; r < H; r++) {{
    z.push(T[r]);
    text.push(T[r].map(v =>
      v === null
        ? '<span style="color:#666">Ciel (masqué)</span>'
        : `<b>${{v.toFixed(1)}} °C</b><br>Pixel (${{r}}, ${{text.length - 1 + 1}})`
    ));
  }}

  const trace = {{
    type: 'heatmap',
    z: z,
    zmin: VMIN,
    zmax: VMAX,
    colorscale: [
      [0.000, 'rgb(49,54,149)'],
      [0.125, 'rgb(69,117,180)'],
      [0.250, 'rgb(116,173,209)'],
      [0.375, 'rgb(171,217,233)'],
      [0.500, 'rgb(255,255,191)'],
      [0.625, 'rgb(254,224,144)'],
      [0.750, 'rgb(253,174,97)'],
      [0.875, 'rgb(244,109,67)'],
      [1.000, 'rgb(165,0,38)'],
    ],
    showscale: false,
    hovertemplate: '%{{customdata}}<extra></extra>',
    customdata: text,
  }};

  const layout = {{
    margin: {{ l: 10, r: 10, t: 10, b: 10 }},
    paper_bgcolor: '#1a1d2e',
    plot_bgcolor:  '#1a1d2e',
    xaxis: {{ visible: false, scaleanchor: 'y' }},
    yaxis: {{ visible: false, autorange: 'reversed' }},
  }};

  const config = {{
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
    scrollZoom: true,
  }};

  Plotly.react('plot-thermal', [trace], layout, config);

  const plotDiv = document.getElementById('plot-thermal');

  // Retirer les anciens listeners avant d'en ajouter de nouveaux
  plotDiv.removeAllListeners('plotly_hover');
  plotDiv.removeAllListeners('plotly_unhover');

  // Hover → update info bar
  plotDiv.on('plotly_hover', function(data) {{
    const pt = data.points[0];
    const T_val = pt.z;
    const info  = document.getElementById('hover-info');
    if (T_val === null || T_val === undefined) {{
      info.innerHTML = '☁ <span style="color:#5c6bc0">Pixel de ciel — masqué (DN trop bas)</span>';
    }} else {{
      const pct = (T_val - VMIN) / (VMAX - VMIN);
      const [r,g,b] = cmapColor(pct);
      info.innerHTML = `
        🌡 Température : <span class="temp-val">${{T_val.toFixed(1)}} °C</span>
        &nbsp;·&nbsp; Pixel (col ${{pt.x}}, ligne ${{pt.y}})
        &nbsp;·&nbsp; <span style="display:inline-block;width:14px;height:14px;background:rgb(${{r}},${{g}},${{b}});border-radius:3px;vertical-align:middle;border:1px solid #fff3"></span>
      `;
    }}
  }});

  plotDiv.on('plotly_unhover', function() {{
    document.getElementById('hover-info').innerHTML =
      "☝ Survoler l'image pour voir la température du pixel";
  }});
}}

// ── Histogramme global ───────────────────────────────────────────────────
function renderHistogram() {{
  const trace = {{
    type: 'bar',
    x: GLOBAL_HIST.centers,
    y: GLOBAL_HIST.counts,
    marker: {{
      color: GLOBAL_HIST.centers.map(v => {{
        const pct = (v - VMIN) / (VMAX - VMIN);
        const [r,g,b] = cmapColor(pct);
        return `rgb(${{r}},${{g}},${{b}})`;
      }}),
      line: {{ width: 0 }}
    }},
    hovertemplate: '%{{x:.1f}}°C : %{{y:,}} pixels<extra></extra>',
  }};

  const layout = {{
    margin: {{ l: 40, r: 10, t: 8, b: 30 }},
    paper_bgcolor: '#1a1d2e',
    plot_bgcolor:  '#1a1d2e',
    xaxis: {{
      title: {{ text: 'T surface (°C)', font: {{ size: 10, color: '#9fa8da' }} }},
      color: '#9fa8da', gridcolor: '#2a2d3e', tickfont: {{ size: 9 }}
    }},
    yaxis: {{
      title: {{ text: 'Pixels', font: {{ size: 10, color: '#9fa8da' }} }},
      color: '#9fa8da', gridcolor: '#2a2d3e', tickfont: {{ size: 9 }}
    }},
    bargap: 0.05,
  }};

  Plotly.newPlot('plot-hist', [trace], layout, {{ responsive: true, displayModeBar: false }});
}}

// ── Chargement d'une image ───────────────────────────────────────────────
function loadImage(idx) {{
  currentIdx = parseInt(idx);
  const img = IMAGES[currentIdx];
  updateStats(img);
  renderHeatmap(img);
}}

// ── Init ──────────────────────────────────────────────────────────────────
window.addEventListener('load', () => {{
  initSelect();
  renderHistogram();
  loadImage(0);
}});
</script>
</body>
</html>"""

    out_path = os.path.join(output_dir, "mosaic_interactive.html")
    # ⚠️  Le f-string ci-dessus NE contient PAS les données JSON (trop gros,
    #     risque de collision avec les {} du template).
    #     On injecte json_data et json_hist via str.replace() APRÈS le f-string.
    html = html.replace('__IMAGES_JSON_PLACEHOLDER__', json_data)
    html = html.replace('__HIST_JSON_PLACEHOLDER__', json_hist)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n  ✅ HTML interactif : {out_path}  ({size_mb:.1f} Mo)")
    print(f"     → Ouvrir dans Chrome/Firefox pour la visu hover")
    return out_path


# ══════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════

def build_interactive_mosaic(
        vol_dir: str = VOL_DIR,
        plaques_csv: str = PLAQUES_CSV,
        tour_flux_csv: str = TOUR_FLUX_CSV,
        output_dir: str = OUTPUT_DIR,
        altitude_m: float = ALTITUDE_M,
        max_images: int = MAX_IMAGES_INTERACTIF,
) -> str:
    """
    Pipeline complet : extraction + calibration + HTML interactif.

    Returns:
        chemin vers mosaic_interactive.html
    """
    os.makedirs(output_dir, exist_ok=True)

    plaques   = pd.read_csv(plaques_csv)
    tour_flux = pd.read_csv(tour_flux_csv).iloc[0]
    T_AIR  = float(tour_flux['T_air_celsius'])
    HR     = float(tour_flux['HR_pct'])
    L_DOWN = float(tour_flux['L_down_atm'])

    print("═" * 60)
    print("  Pipeline interactif Niakhar — DJI Mavic 3T")
    print("═" * 60)
    print(f"  Vol     : {vol_dir}")
    print(f"  Alt sim : {altitude_m} m   |   Masque ciel DN < {DN_MIN_SOL}")
    print(f"  Tour    : T={T_AIR}°C  HR={HR}%  L↓={L_DOWN} W/m²")
    if max_images:
        print(f"  Subset  : {max_images} images max (HTML interactif)")
    print()

    # Listing images
    jpgs = sorted(glob.glob(os.path.join(vol_dir, "*_T.JPG")))
    if not jpgs:
        jpgs = sorted(glob.glob(os.path.join(vol_dir, "*.JPG")))
    if not jpgs:
        raise FileNotFoundError(f"Aucun .JPG dans : {vol_dir}")

    total = len(jpgs)
    if max_images:
        # Sous-ensemble régulièrement espacé pour être représentatif
        indices = np.linspace(0, total - 1, min(max_images, total), dtype=int)
        jpgs_subset = [jpgs[i] for i in indices]
    else:
        jpgs_subset = jpgs

    print(f"  {total} images trouvées → traitement de {len(jpgs_subset)}\n")

    results = []
    for img_path in jpgs_subset:
        res = process_image(img_path, plaques, T_AIR, HR, L_DOWN, altitude_m)
        if res:
            results.append(res)

    if not results:
        print("  Aucune image traitée.")
        return ""

    # Stats globales
    all_T = np.concatenate([r['T_image'][~np.isnan(r['T_image'])].flatten()
                            for r in results])
    print(f"\n  ── Résumé ──────────────────────────────────────────────")
    print(f"  Images   : {len(results)}")
    print(f"  T globale: moy={np.mean(all_T):.1f}°C  "
          f"min={np.min(all_T):.1f}°C  max={np.max(all_T):.1f}°C")
    print(f"  (T max corrigée vs 132°C avant masque ciel)")

    return build_interactive_html(results, output_dir, max_images=None)


if __name__ == "__main__":
    build_interactive_mosaic()