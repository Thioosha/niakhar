"""
extract_dn.py — Extraction des DN bruts depuis un RJPEG DJI thermique.
Utilise dji_irp.exe en mode 'extract' pour obtenir la matrice uint16 brute.
Le script reste testable directement avec `python calibration/extract_dn.py`
"""

import numpy as np
import subprocess
import os

DJI_IRP_EXE = os.path.join(os.path.dirname(__file__), '..', 'dji_irp.exe')
IMG_HEIGHT, IMG_WIDTH = 512, 640


def extract_dn(rjpeg_path: str) -> np.ndarray:
    """
    Extrait la matrice DN brute (uint16, 512x640) depuis un RJPEG DJI thermique.
    Le fichier .raw temporaire est supprimé après lecture.
    """
    raw_path = rjpeg_path.replace(".JPG", "_tmp.raw").replace(".jpg", "_tmp.raw")

    result = subprocess.run(
        [DJI_IRP_EXE, "-s", rjpeg_path, "-a", "extract", "-o", raw_path],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"dji_irp.exe a échoué : {result.stderr}")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Fichier raw non généré : {raw_path}")

    dn = np.fromfile(raw_path, dtype='<u2').reshape((IMG_HEIGHT, IMG_WIDTH))
    os.remove(raw_path)
    return dn


if __name__ == "__main__":
    # Test rapide sur une image
    IMG_PATH = r"Data_DJI\Images thermiques\DJI_202604251331_001_ThermiqueEPT\DJI_20260425140653_0925_T.JPG"
    dn = extract_dn(IMG_PATH)
    print(f"Shape: {dn.shape} | Min: {dn.min()} | Max: {dn.max()} | Moy: {dn.mean():.1f}")