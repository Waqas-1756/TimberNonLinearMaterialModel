
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tests"))
import openseespy.opensees as ops
from common import define_material, run_history

for n in (60,120,240,480):
    ops.wipe()
    define_material(1)
    axial=np.linspace(0.0,-0.095,n)
    _,s=run_history(1,axial)
    sig=-s[:,0]
    print(
        f"N={n:4d}  peak={sig.max():.6f} MPa  "
        f"final={sig[-1]:.6f} MPa"
    )
