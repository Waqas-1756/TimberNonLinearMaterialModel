
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tests"))
import openseespy.opensees as ops
from common import define_material, trial_stress

def fd_tangent(eps, h=1e-8):
    eps=np.asarray(eps,float)
    s0=trial_stress(eps)
    C=np.zeros((6,6))
    for j in range(6):
        ep=eps.copy()
        ep[j]+=h
        C[:,j]=(trial_stress(ep)-s0)/h
    return C

ops.wipe()
define_material(1)

eps=np.array([-1e-4,0,0,0,0,0],float)
ops.setTrialStrain(*eps)
C_op=np.asarray(ops.getTangent(),float).reshape(6,6)
C_fd=fd_tangent(eps)

err=np.linalg.norm(C_op-C_fd)/np.linalg.norm(C_fd)
print("Elastic tangent relative error =",err)

if err > 1e-5:
    raise SystemExit("Elastic tangent check failed.")

print("Elastic tangent check passed.")
print("Nonlinear consistent tangent still needs separate implementation/verification.")
