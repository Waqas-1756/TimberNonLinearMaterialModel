
"""
Generate full Python reference CSV histories.

Run this with the standalone validated Python model only.
It does NOT require the C++ OpenSees material.
"""
from pathlib import Path
import runpy
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
MODEL = ROOT/"reference"/"python_reference_model.py"
ns = runpy.run_path(str(MODEL), run_name="reference_model")

run_comp = ns["run_uniaxial_stress_test"]
run_tens = ns["run_uniaxial_tension_test"]

sig_e = ns["SIGMA_E0_CALIBRATED_MPA"]
A = ns["A_COMP_CALIBRATED"]
B = ns["B_COMP_CALIBRATED"]

common = dict(
    stress_tol=1e-6,
    sigma_e0=sig_e,
    use_hardening=True,
    use_damage=True,
    Lc=5.0,
    dt=1.0,
    A_comp=A,
    B_comp=B,
    show_progress=True,
    constitutive_max_iter=30,
    inner_max_iter=12,
)

cases = {
    "parallel_compression_reference.csv":
        run_comp(direction=1, max_compressive_strain=0.095,
                 n_steps=120, **common),
    "perpendicular_compression_reference.csv":
        run_comp(direction=2, max_compressive_strain=0.12,
                 n_steps=120, **common),
    "parallel_tension_reference.csv":
        run_tens(direction=1, max_tensile_strain=0.025,
                 n_steps=160, sigma_e0=sig_e, Lc=5.0, dt=1.0),
    "perpendicular_tension_reference.csv":
        run_tens(direction=2, max_tensile_strain=0.012,
                 n_steps=160, sigma_e0=sig_e, Lc=5.0, dt=1.0),
}

for name, r in cases.items():
    data = {
        "strain_axial": r["axial_strain"],
        "stress_axial_MPa": r["axial_stress"],
        "damage_d1": r["damage"][:,0],
        "damage_d2": r["damage"][:,1],
        "damage_d3": r["damage"][:,2],
    }
    if "eq_plastic_strain" in r:
        data["eq_plastic_strain"] = r["eq_plastic_strain"]

    out = ROOT/"reference"/name
    pd.DataFrame(data).to_csv(out, index=False)
    print("Wrote", out)
