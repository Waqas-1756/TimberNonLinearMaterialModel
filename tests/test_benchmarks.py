
from pathlib import Path
import json
import numpy as np
import openseespy.opensees as ops
from common import define_material, run_history

ROOT = Path(__file__).resolve().parents[1]
B = json.loads((ROOT/"reference"/"benchmarks.json").read_text())["validated_results"]

def test_parallel_compression():
    ops.wipe()
    define_material(1)
    axial = np.linspace(0.0, -0.095, 120)
    _, s = run_history(1, axial)
    sig = -s[:,0]

    assert abs(sig.max() - B["parallel_compression_peak_MPa"]) < 0.35
    assert abs(axial[np.argmax(sig)] + B["parallel_compression_peak_strain"]) < 0.003
    assert abs(sig[-1] - B["parallel_compression_final_MPa"]) < 0.35

def test_perpendicular_compression():
    ops.wipe()
    define_material(2)
    axial = np.linspace(0.0, -0.12, 120)
    _, s = run_history(2, axial)
    assert abs((-s[-1,1]) - B["perpendicular_compression_final_MPa"]) < 0.10

def test_parallel_tension():
    ops.wipe()
    define_material(3)
    axial = np.linspace(0.0, 0.025, 160)
    _, s = run_history(1, axial)
    assert abs(s[:,0].max() - B["parallel_tension_peak_MPa"]) < 0.20

def test_perpendicular_tension():
    ops.wipe()
    define_material(4)
    axial = np.linspace(0.0, 0.012, 160)
    _, s = run_history(2, axial)
    assert abs(s[:,1].max() - B["perpendicular_tension_peak_MPa"]) < 0.03
