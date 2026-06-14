import numpy as np
import matplotlib.pyplot as plt
import subprocess, os

DJI_IRP_EXE = os.path.join(os.path.dirname(__file__), '..', 'dji_irp.exe')
IMG_PATH = r"Data_DJI\Images thermiques\DJI_202604251331_001_ThermiqueEPT\DJI_20260425140653_0925_T.JPG"

subprocess.run([DJI_IRP_EXE, "-s", IMG_PATH, "-a", "extract", "-o", "calibration/test.raw"])
dn = np.fromfile("calibration/test.raw", dtype='<u2').reshape((512, 640))

print(f"Shape: {dn.shape} | Min: {dn.min()} | Max: {dn.max()} | Moy: {dn.mean():.1f}")

plt.imshow(dn, cmap='hot')
plt.colorbar(label='DN brut')
plt.title('DN bruts — image thermique')
plt.savefig("calibration/dn_visu.png", dpi=150)
plt.show()