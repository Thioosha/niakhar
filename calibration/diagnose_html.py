"""
diagnose_html.py — Analyse le mosaic_interactive.html généré pour trouver le bug.

Usage :
    python calibration/diagnose_html.py calibration/orthomosaic_output/mosaic_interactive.html
"""
import sys
import re
import json
from pathlib import Path

if len(sys.argv) < 2:
    # Essai chemins par défaut
    candidates = [
        Path("calibration/orthomosaic_output/mosaic_interactive.html"),
        Path("mosaic_interactive.html"),
    ]
    html_path = next((p for p in candidates if p.exists()), None)
    if not html_path:
        print("Usage: python diagnose_html.py chemin/vers/mosaic_interactive.html")
        sys.exit(1)
else:
    html_path = Path(sys.argv[1])

print(f"📄 Fichier : {html_path}")
print(f"   Taille  : {html_path.stat().st_size / 1e6:.1f} Mo\n")

content = html_path.read_text(encoding='utf-8', errors='replace')

# ── 1. Vérifier que le bloc <script> existe ───────────────────────────────
script_blocks = re.findall(r'<script[^>]*>', content)
print(f"[1] Blocs <script> trouvés : {len(script_blocks)}")
for s in script_blocks:
    print(f"    {s}")

# ── 2. Chercher la ligne const IMAGES ────────────────────────────────────
match = re.search(r'const IMAGES\s*=\s*(.{0,200})', content)
if match:
    snippet = match.group(1)
    print(f"\n[2] const IMAGES = {snippet[:120]}...")
    # Est-ce que c'est du vrai JSON ?
    if snippet.strip().startswith('['):
        print("    ✅ Commence par '[' → tableau JSON présent")
        # Trouver le tableau complet (gros, on prend juste les 500 premiers chars)
        start = content.index('const IMAGES = ') + len('const IMAGES = ')
        bracket_depth = 0
        end = start
        for i, c in enumerate(content[start:start+200]):
            if c == '[': bracket_depth += 1
            elif c == ']': bracket_depth -= 1
            if bracket_depth == 0 and i > 0:
                end = start + i + 1
                break
        # Compter les éléments du tableau
        try:
            images_json = content[start:end] if end > start else content[start:start+10000]
            # Chercher combien d'objets {name:
            n_images = len(re.findall(r'"name"\s*:', content[start:start+5_000_000]))
            print(f"    📊 Nombre d'images détectées dans JSON : {n_images}")
        except Exception as e:
            print(f"    ⚠️  Erreur parsing : {e}")
    elif snippet.strip().startswith('__'):
        print("    ❌ PLACEHOLDER non substitué ! Le .replace() n'a pas été exécuté.")
        print("    → Le nouveau interactive_mosaic.py n'a pas été utilisé pour générer ce HTML.")
    elif snippet.strip() == ';' or snippet.strip() == '':
        print("    ❌ VIDE — le JSON n'a pas été injecté du tout")
    else:
        print(f"    ⚠️  Valeur inattendue : {snippet[:80]}")
else:
    print("\n[2] ❌ 'const IMAGES' introuvable dans le HTML !")
    print("    → Le bloc <script> est corrompu ou absent.")

# ── 3. Chercher GLOBAL_HIST ───────────────────────────────────────────────
match2 = re.search(r'const GLOBAL_HIST\s*=\s*(.{0,100})', content)
if match2:
    print(f"\n[3] const GLOBAL_HIST = {match2.group(1)[:80]}...")
else:
    print("\n[3] ❌ 'const GLOBAL_HIST' introuvable")

# ── 4. Vérifier Plotly CDN ───────────────────────────────────────────────
plotly_cdn = re.search(r'plotly[^"\']*\.min\.js', content)
print(f"\n[4] Plotly CDN : {'✅ ' + plotly_cdn.group() if plotly_cdn else '❌ ABSENT'}")

# ── 5. Chercher la fonction loadImage ────────────────────────────────────
has_load = 'function loadImage' in content
has_render = 'function renderHeatmap' in content
has_init = "window.addEventListener('load'" in content
print(f"\n[5] Fonctions JS :")
print(f"    loadImage     : {'✅' if has_load else '❌'}")
print(f"    renderHeatmap : {'✅' if has_render else '❌'}")
print(f"    window load   : {'✅' if has_init else '❌'}")

# ── 6. Chercher des erreurs de syntaxe JS évidentes ─────────────────────
# Chercher des { non fermés dans la partie JS (hors données)
script_start = content.rfind('<script>')
script_end   = content.rfind('</script>')
if script_start > 0 and script_end > script_start:
    js_block = content[script_start:script_end]
    print(f"\n[6] Taille du bloc JS : {len(js_block)/1e6:.2f} Mo")
    # Vérifier truncation
    if 'window.addEventListener' in js_block:
        print("    ✅ window.addEventListener présent → JS non tronqué")
    else:
        print("    ❌ window.addEventListener ABSENT → JS tronqué ou corrompu !")
else:
    print("\n[6] ❌ Impossible de localiser le bloc <script>...</script>")

# ── 7. Vérifier truncation générale ──────────────────────────────────────
ends_ok = content.rstrip().endswith('</html>')
print(f"\n[7] Fin du fichier '</html>' : {'✅' if ends_ok else '❌ FICHIER TRONQUÉ !'}")
if not ends_ok:
    print(f"    Derniers 100 chars : {repr(content[-100:])}")

print("\n" + "═"*60)
print("CONCLUSION :")
if not match:
    print("  → Le JSON n'est pas du tout dans le HTML. Vérifier la génération.")
elif match and '__' in match.group(1):
    print("  → Placeholder non substitué. Utilise bien le nouveau interactive_mosaic.py ?")
elif not ends_ok:
    print("  → Fichier tronqué. Problème d'écriture disque (59 Mo, mémoire ?).")
else:
    print("  → Structure HTML semble OK. Colle les erreurs de la console F12.")