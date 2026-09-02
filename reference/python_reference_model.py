
"""
Standalone 3D orthotropic Hoffman timber material driver
========================================================

Purpose
-------
This is a FIRST validation-stage implementation based on:
H. Eslami, L.B. Jayasinghe, D. Waldmann (2021),
"Nonlinear three-dimensional anisotropic material model for failure analysis of timber",
Engineering Failure Analysis 130, 105764.

What is implemented here
------------------------
1) 3D orthotropic elastic stiffness matrix (paper Eqs. 1-7)
2) Hoffman yield surface (paper Eqs. 10-22)
3) Associative flow direction df/dsigma = P*sigma + Q
4) Elastic predictor / perfect-plastic return mapping for the quadratic Hoffman surface
5) Paper Table 1 material properties for the compression-validation example
6) Stress-based damage initiation indices F1c, F1t, F2t, F3t (paper Eqs. 37-40)
7) Tensile damage evolution function (paper Eq. 35) and viscous regularization (Eqs. 41-42)
8) Standalone uniaxial strain drivers parallel and perpendicular to grain

Important limitation
--------------------
The published paper gives h = 1200 MPa but does not clearly provide all numerical
constants needed to reproduce the complete compression-damage evolution (notably A and B
in Eq. 36). Therefore this file intentionally runs the plasticity validation initially as
PERFECT PLASTICITY (hardening disabled) rather than inventing missing constants.

Once the hardening convention and A/B values are established, those blocks can be added
without changing the basic class architecture.

Stress/strain ordering
----------------------
[11, 22, 33, 12, 13, 23]

The shear strain entries are ENGINEERING shear strains:
gamma12, gamma13, gamma23.
"""

import copy
import math
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Paper Table 1: compression-test validation parameters
# Units: MPa, N/mm, dimensionless strains
# ----------------------------------------------------------------------
PAPER_TABLE1 = {
    # Elasticity
    "E1": 2050.8,
    "E2": 172.1,
    "E3": 172.1,
    "G23": 68.0,
    "G12": 145.2,
    "G13": 145.2,
    "nu23": 0.50,
    "nu12": 0.45,
    "nu13": 0.45,

    # Strengths
    # Positive magnitudes are supplied here.
    "fc1": 35.0,
    "fc2": 2.5,
    "fc3": 2.5,
    "ft1": 20.0,
    "ft2": 0.7,
    "ft3": 0.7,
    "f12": 5.0,
    "f13": 5.0,
    "f23": 0.5,

    # Paper's listed hardening modulus.
    # Not activated in this first driver because the paper's full numerical
    # hardening normalization is not sufficiently specified for a clean,
    # reproducible standalone implementation from Table 1 alone.
    "h": 1200.0,

    # Fracture energies
    "Gf1t": 60.0,
    "Gf2t": 0.5,
    "Gf3t": 0.5,

    # Paper's viscous damage parameter
    "eta": 1.0e-4,
}

# Approximate calibration read from Fig. 4(b) of Eslami et al. (2021).
# Red FEM curve: sigma ~2.70 MPa at strain ~0.02,
#                sigma ~3.45 MPa at strain ~0.12.
# Hence T2 ~7.5 MPa.
HARDENING_INTERPRETATION = "fixed_Q_only_sigma_ek_evolves"
T2_FIG4B_APPROX_MPA = 7.5
SIGMA_E0_FIG4B_APPROX_MPA = (
    PAPER_TABLE1["h"] * PAPER_TABLE1["fc2"]
    * (PAPER_TABLE1["E2"] - T2_FIG4B_APPROX_MPA)
    / (PAPER_TABLE1["E2"] * T2_FIG4B_APPROX_MPA)
)



def sigma_e0_from_perpendicular_tangent(T2, p=None):
    """
    Back-calculate the reference equivalent yield stress sigma_e from paper Eq. (24):

        h = (sigma_e / sigma_2c) * E2*T2/(E2 - T2)

    Therefore:

        sigma_e = h*sigma_2c*(E2 - T2)/(E2*T2)

    The paper reports h=1200 MPa but does not list sigma_e numerically in Table 1.
    This helper lets you determine sigma_e0 if you know/calibrate the post-yield
    tangent T2 from a uniaxial compression curve.

    T2 must be in MPa and satisfy 0 < T2 < E2.
    """
    pp = PAPER_TABLE1 if p is None else p
    E2 = float(pp["E2"])
    h = float(pp["h"])
    fc2 = float(pp["fc2"])
    T2 = float(T2)

    if not (0.0 < T2 < E2):
        raise ValueError(f"T2 must satisfy 0 < T2 < E2={E2} MPa.")

    return h * fc2 * (E2 - T2) / (E2 * T2)


# ----------------------------------------------------------------------
# CALIBRATED VALIDATION PARAMETERS
# ----------------------------------------------------------------------
# These are NOT values reported by Eslami et al.
# They were calibrated here against approximate digitization of Fig. 4:
#
#   sigma_e0 = 270.000 MPa
#   A_comp    = 0.600
#   B_comp    = 0.500
#
# Current validation performance:
#   - perpendicular compression: ~3.42 MPa at strain 0.12
#   - parallel compression: peak ~39-40 MPa followed by softening
#
# Treat these as provisional calibration values until independent
# experimental/digital data are used for formal validation.
SIGMA_E0_CALIBRATED_MPA = 270.000000000000
A_COMP_CALIBRATED = 0.600000000000
B_COMP_CALIBRATED = 0.500000000000



def make_calibrated_timber_material(
    Lc_mm,
    dt=1.0,
    use_hardening=True,
    use_damage=True,
):
    """
    Convenience constructor for the CURRENT calibrated prototype.

    Calibrated compression parameters:
        sigma_e0 = SIGMA_E0_CALIBRATED_MPA
        A_comp    = A_COMP_CALIBRATED
        B_comp    = B_COMP_CALIBRATED

    IMPORTANT:
    Lc_mm is the characteristic length used in the tensile fracture-energy
    damage law. It is NOT a universal material constant and must correspond
    to the chosen continuum/fiber discretization.
    """
    return HoffmanTimber3D(
        sigma_e0=SIGMA_E0_CALIBRATED_MPA,
        use_hardening=use_hardening,
        use_damage=use_damage,
        Lc=float(Lc_mm),
        dt=float(dt),
        A_comp=A_COMP_CALIBRATED,
        B_comp=B_COMP_CALIBRATED,
        tol=1e-8,
        max_iter=40,
        inner_tol=1e-9,
        inner_max_iter=20,
    )


def orthotropic_stiffness(p):
    """
    Build the 6x6 orthotropic elastic stiffness matrix.

    Stress vector:
        [s11, s22, s33, t12, t13, t23]

    Strain vector:
        [e11, e22, e33, g12, g13, g23]

    This construction is equivalent to the paper's Eqs. (1)-(7), but it is
    numerically safer to build the compliance matrix and invert it.
    """

    E1, E2, E3 = p["E1"], p["E2"], p["E3"]
    G12, G13, G23 = p["G12"], p["G13"], p["G23"]
    nu12, nu13, nu23 = p["nu12"], p["nu13"], p["nu23"]

    # Reciprocity:
    # nu_ji / E_j = nu_ij / E_i
    nu21 = nu12 * E2 / E1
    nu31 = nu13 * E3 / E1
    nu32 = nu23 * E3 / E2

    S = np.zeros((6, 6), dtype=float)

    S[0, 0] = 1.0 / E1
    S[1, 1] = 1.0 / E2
    S[2, 2] = 1.0 / E3

    S[0, 1] = -nu21 / E2
    S[1, 0] = -nu12 / E1

    S[0, 2] = -nu31 / E3
    S[2, 0] = -nu13 / E1

    S[1, 2] = -nu32 / E3
    S[2, 1] = -nu23 / E2

    S[3, 3] = 1.0 / G12
    S[4, 4] = 1.0 / G13
    S[5, 5] = 1.0 / G23

    C = np.linalg.inv(S)
    return C


def hoffman_P_Q(p, sigma_e=1.0):
    """
    Hoffman P matrix and Q vector following paper Eqs. (11)-(22).

    sigma_e may be set to 1.0 for a normalized initial yield surface.
    With sigma_e=1, f=0 reproduces the supplied uniaxial strengths.
    """
    fc1, fc2, fc3 = p["fc1"], p["fc2"], p["fc3"]
    ft1, ft2, ft3 = p["ft1"], p["ft2"], p["ft3"]
    f12, f13, f23 = p["f12"], p["f13"], p["f23"]

    se2 = sigma_e**2

    a12 = 0.5 * se2 * (
        1.0 / (fc1 * ft1)
        + 1.0 / (fc2 * ft2)
        - 1.0 / (fc3 * ft3)
    )
    a23 = 0.5 * se2 * (
        -1.0 / (fc1 * ft1)
        + 1.0 / (fc2 * ft2)
        + 1.0 / (fc3 * ft3)
    )
    a13 = 0.5 * se2 * (
        1.0 / (fc1 * ft1)
        - 1.0 / (fc2 * ft2)
        + 1.0 / (fc3 * ft3)
    )

    # With strengths entered as positive magnitudes:
    # Paper convention is reproduced so that:
    # sigma1 = +ft1 -> f=0
    # sigma1 = -fc1 -> f=0
    a11 = se2 * (fc1 - ft1) / (fc1 * ft1)
    a22 = se2 * (fc2 - ft2) / (fc2 * ft2)
    a33 = se2 * (fc3 - ft3) / (fc3 * ft3)

    a44 = se2 / (3.0 * f12**2)
    a55 = se2 / (3.0 * f13**2)
    a66 = se2 / (3.0 * f23**2)

    P = np.zeros((6, 6), dtype=float)

    P[0, 0] = 2.0 * (a13 + a12)
    P[1, 1] = 2.0 * (a23 + a12)
    P[2, 2] = 2.0 * (a13 + a23)

    P[0, 1] = P[1, 0] = -2.0 * a12
    P[0, 2] = P[2, 0] = -2.0 * a13
    P[1, 2] = P[2, 1] = -2.0 * a23

    P[3, 3] = 6.0 * a44
    P[4, 4] = 6.0 * a55
    P[5, 5] = 6.0 * a66

    Q = np.array([a11, a22, a33, 0.0, 0.0, 0.0], dtype=float)

    return P, Q


def hoffman_value(stress, P, Q, sigma_e=1.0):
    """Paper Eq. (20): f = 1/2 sigma^T P sigma + sigma^T Q - sigma_e^2."""
    s = np.asarray(stress, dtype=float)
    return 0.5 * s @ P @ s + s @ Q - sigma_e**2


def damage_indices(stress, p):
    """
    Paper Eqs. (37)-(40).

    Returns current stress-based failure indices.
    A value >= 1 means damage initiation criterion has been reached.
    """
    s1, s2, s3, s12, s13, s23 = np.asarray(stress, dtype=float)

    F1c = (-s1 / p["fc1"]) if s1 < 0.0 else 0.0
    F1t = ( s1 / p["ft1"]) if s1 > 0.0 else 0.0

    F2t = 0.0
    if s2 > 0.0:
        F2t = (
            (s2  / p["ft2"])**2
            + (s12 / p["f12"])**2
            + (s23 / p["f23"])**2
        )

    F3t = 0.0
    if s3 > 0.0:
        F3t = (
            (s3  / p["ft3"])**2
            + (s13 / p["f13"])**2
            + (s23 / p["f23"])**2
        )

    return {
        "F1c": F1c,
        "F1t": F1t,
        "F2t": F2t,
        "F3t": F3t,
    }


def tensile_damage(Fit, Lc, strength_t, Ei, Gfi):
    """
    Paper Eq. (35).

    d_it = 1 - (1/F_it) exp[(1-F_it) Lc sigma_it^2 / (Ei Gf_it)]

    Only active after Fit >= 1.
    Lc must be in mm when:
      Ei, strength_t are in MPa (= N/mm^2)
      Gfi is in N/mm
    """
    if Fit <= 1.0:
        return 0.0

    exponent = (1.0 - Fit) * Lc * strength_t**2 / (Ei * Gfi)
    d = 1.0 - (1.0 / Fit) * math.exp(exponent)
    return float(np.clip(d, 0.0, 0.999999))


def viscous_regularize_damage(d_old_v, d_new, eta, dt):
    """Paper Eq. (42)."""
    return (eta / (eta + dt)) * d_old_v + (dt / (eta + dt)) * d_new


class HoffmanTimber3D:
    """
    Standalone validation implementation of the Eslami et al. timber model.

    Included:
      * 3D orthotropic elasticity
      * Hoffman plasticity in compression
      * linear isotropic hardening
      * equivalent plastic strain
      * anisotropic tensile damage d1t, d2t, d3t
      * longitudinal compression damage d1c
      * viscous damage regularization
      * damaged orthotropic stiffness matrix

    IMPORTANT
    ---------
    The paper does NOT report numerical values of A and B in its compression
    damage Eq. (36). Therefore longitudinal compression damage is only active
    when A_comp and B_comp are explicitly supplied by the user.

    The tensile damage equation also requires the element characteristic
    length Lc. Lc is a mesh/model property, not a universal material constant,
    so it is supplied explicitly.

    Stress/strain ordering:
        [11, 22, 33, 12, 13, 23]

    Shear strain components are ENGINEERING shear strains.
    """

    def __init__(
        self,
        params=None,
        sigma_e0=1.0,
        use_hardening=True,
        use_damage=True,
        Lc=None,                 # mm
        dt=1.0,                 # pseudo-time/load-step increment for Eq. 42
        A_comp=None,            # paper Eq. 36; not reported in Table 1
        B_comp=None,            # paper Eq. 36; not reported in Table 1
        tol=1e-10,
        max_iter=120,
        inner_tol=1e-12,
        inner_max_iter=80,
    ):
        self.p = copy.deepcopy(PAPER_TABLE1 if params is None else params)
        self.C0 = orthotropic_stiffness(self.p)

        self.sigma_e0 = float(sigma_e0)
        if self.sigma_e0 <= 0.0:
            raise ValueError("sigma_e0 must be positive.")

        # Initial Hoffman matrices.
        # P remains fixed under the uniform isotropic-expansion interpretation.
        self.P, self.Q0 = hoffman_P_Q(self.p, sigma_e=self.sigma_e0)

        self.h = float(self.p["h"]) if use_hardening else 0.0
        self.use_hardening = bool(use_hardening)
        self.use_damage = bool(use_damage)

        self.Lc = None if Lc is None else float(Lc)
        self.dt = float(dt)
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0.")

        self.eta = float(self.p.get("eta", 1.0e-4))

        self.A_comp = None if A_comp is None else float(A_comp)
        self.B_comp = None if B_comp is None else float(B_comp)

        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.inner_tol = float(inner_tol)
        self.inner_max_iter = int(inner_max_iter)

        # Paper Eq. (55).
        self.Z = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])

        # ----------------------------
        # Committed constitutive state
        # ----------------------------
        self.strain_commit = np.zeros(6)
        self.plastic_strain_commit = np.zeros(6)
        self.eq_plastic_strain_commit = 0.0

        # Effective (undamaged) stress, paper Eq. (32).
        self.effective_stress_commit = np.zeros(6)

        # Cauchy/damaged stress, paper Eq. (33).
        self.stress_commit = np.zeros(6)

        # Non-regularized damage histories.
        # [d1t, d2t, d3t]
        self.damage_t_commit = np.zeros(3)
        self.damage_c1_commit = 0.0

        # Viscously regularized directional damage d1,d2,d3 used in Ed.
        self.damage_v_commit = np.zeros(3)

        # Track maximum failure indices so damage is irreversible.
        self.Fmax_t_commit = np.ones(3)
        self.Fmax_c1_commit = 1.0

        # ----------------------------
        # Trial state
        # ----------------------------
        self.strain_trial = self.strain_commit.copy()
        self.plastic_strain_trial = self.plastic_strain_commit.copy()
        self.eq_plastic_strain_trial = self.eq_plastic_strain_commit
        self.effective_stress_trial = self.effective_stress_commit.copy()
        self.stress_trial = self.stress_commit.copy()

        self.damage_t_trial = self.damage_t_commit.copy()
        self.damage_c1_trial = self.damage_c1_commit
        self.damage_v_trial = self.damage_v_commit.copy()
        self.Fmax_t_trial = self.Fmax_t_commit.copy()
        self.Fmax_c1_trial = self.Fmax_c1_commit

        self.Cdam_trial = self.C0.copy()
        self.last_dlambda = 0.0

    # ================================================================
    # Elasticity / hardening / Hoffman plasticity
    # ================================================================

    def get_initial_tangent(self):
        return self.C0.copy()

    def sigma_ek(self, eqp):
        # Paper Eq. (23): sigma_ek = sigma_e + h*k
        return self.sigma_e0 + self.h * float(eqp)

    def Q_at_eqp(self, eqp):
        """
        Diagnostic hardening interpretation:
        keep Hoffman Q FIXED at its initial value and evolve only sigma_ek.

        This tests whether the excessive hardening came from scaling Q(k)
        together with sigma_ek.
        """
        return self.Q0.copy()

    def yield_function(self, effective_stress, eqp=None):
        if eqp is None:
            eqp = self.eq_plastic_strain_trial

        s = np.asarray(effective_stress, dtype=float)
        Qk = self.Q_at_eqp(eqp)
        sek = self.sigma_ek(eqp)

        return 0.5 * s @ self.P @ s + s @ Qk - sek**2

    def flow_direction(self, effective_stress, eqp):
        # Paper Eq. (52).
        return (
            self.P @ np.asarray(effective_stress, dtype=float)
            + self.Q_at_eqp(eqp)
        )

    def equivalent_plastic_increment(self, deps_p):
        # Paper Eq. (54).
        v = np.asarray(deps_p, dtype=float)
        val = (2.0 / 3.0) * (v @ self.Z @ v)
        return math.sqrt(max(float(val), 0.0))

    def _compression_plasticity_active(self, stress):
        """
        The paper assigns no plastic flow to tension.
        Plastic correction is therefore only permitted when at least one
        normal effective-stress component is compressive.
        """
        s = np.asarray(stress, dtype=float)
        return bool(np.any(s[:3] < 0.0))

    def _state_for_lambda_hardening(self, sigma_trial, dlambda):
        dlambda = float(dlambda)

        if dlambda <= 0.0:
            k = float(self.eq_plastic_strain_commit)
            return np.asarray(sigma_trial, dtype=float).copy(), k, np.zeros(6)

        k = float(self.eq_plastic_strain_commit)
        sigma = np.asarray(sigma_trial, dtype=float).copy()
        I = np.eye(6)

        for _ in range(self.inner_max_iter):
            Qk = self.Q_at_eqp(k)

            # Paper Eq. (53), rearranged.
            A = I + dlambda * (self.C0 @ self.P)
            rhs = sigma_trial - dlambda * (self.C0 @ Qk)
            sigma_new = np.linalg.solve(A, rhs)

            g = self.P @ sigma_new + Qk

            # Paper Eqs. (57)-(58).
            gamma = math.sqrt(
                max(float((2.0 / 3.0) * (g @ self.Z @ g)), 0.0)
            )
            k_new = self.eq_plastic_strain_commit + dlambda * gamma

            k_relaxed = 0.5 * k + 0.5 * k_new

            if (
                abs(k_relaxed - k)
                <= self.inner_tol * max(1.0, abs(k_relaxed))
                and np.linalg.norm(sigma_new - sigma, ord=np.inf)
                <= self.inner_tol
                * max(1.0, np.linalg.norm(sigma_new, ord=np.inf))
            ):
                k = k_new
                sigma = sigma_new
                break

            k = k_relaxed
            sigma = sigma_new

        # Final consistent update.
        Qk = self.Q_at_eqp(k)
        A = I + dlambda * (self.C0 @ self.P)
        rhs = sigma_trial - dlambda * (self.C0 @ Qk)
        sigma = np.linalg.solve(A, rhs)

        g = self.P @ sigma + Qk
        deps_p = dlambda * g
        k = (
            self.eq_plastic_strain_commit
            + self.equivalent_plastic_increment(deps_p)
        )

        # One last correction with final k.
        Qk = self.Q_at_eqp(k)
        A = I + dlambda * (self.C0 @ self.P)
        rhs = sigma_trial - dlambda * (self.C0 @ Qk)
        sigma = np.linalg.solve(A, rhs)

        g = self.P @ sigma + Qk
        deps_p = dlambda * g
        k = (
            self.eq_plastic_strain_commit
            + self.equivalent_plastic_increment(deps_p)
        )

        return sigma, float(k), deps_p

    def _return_map_isotropic_hardening(self, sigma_trial):
        k_n = float(self.eq_plastic_strain_commit)

        # No plasticity in a purely tensile state.
        if not self._compression_plasticity_active(sigma_trial):
            return np.asarray(sigma_trial).copy(), k_n, np.zeros(6), 0.0

        f_trial = self.yield_function(sigma_trial, eqp=k_n)
        if f_trial <= self.tol:
            return np.asarray(sigma_trial).copy(), k_n, np.zeros(6), 0.0

        lo = 0.0
        hi = 1.0e-12

        for _ in range(140):
            s_hi, k_hi, depsp_hi = self._state_for_lambda_hardening(
                sigma_trial, hi
            )
            f_hi = self.yield_function(s_hi, eqp=k_hi)
            if f_hi <= 0.0:
                break
            hi *= 10.0
        else:
            raise RuntimeError(
                "Could not bracket Hoffman hardening return map."
            )

        best = None

        for _ in range(self.max_iter):
            mid = 0.5 * (lo + hi)

            s_mid, k_mid, depsp_mid = self._state_for_lambda_hardening(
                sigma_trial, mid
            )
            f_mid = self.yield_function(s_mid, eqp=k_mid)

            best = (s_mid, k_mid, depsp_mid, mid)

            if abs(f_mid) <= self.tol:
                return best

            if f_mid > 0.0:
                lo = mid
            else:
                hi = mid

        return best

    # ================================================================
    # Damage model: paper Eqs. (34)-(42)
    # ================================================================

    def failure_indices(self, effective_stress):
        """
        Paper Eqs. (37)-(40), evaluated using effective/undamaged stress.
        """
        s1, s2, s3, s12, s13, s23 = np.asarray(
            effective_stress, dtype=float
        )

        F1c = (-s1 / self.p["fc1"]) if s1 < 0.0 else 0.0
        F1t = ( s1 / self.p["ft1"]) if s1 > 0.0 else 0.0

        F2t = 0.0
        if s2 > 0.0:
            F2t = (
                (s2  / self.p["ft2"])**2
                + (s12 / self.p["f12"])**2
                + (s23 / self.p["f23"])**2
            )

        F3t = 0.0
        if s3 > 0.0:
            F3t = (
                (s3  / self.p["ft3"])**2
                + (s13 / self.p["f13"])**2
                + (s23 / self.p["f23"])**2
            )

        return np.array([F1t, F2t, F3t]), float(F1c)

    def _tensile_damage_from_F(self, F, direction):
        """
        Paper Eq. (35):
          dit = 1 - (1/Fit) exp[
                   (1-Fit) Lc sigma_it^2 / (Ei Gf_it)
                ]
        """
        F = float(F)
        if F <= 1.0:
            return 0.0

        if self.Lc is None:
            raise ValueError(
                "Tensile damage was triggered, but Lc was not supplied. "
                "Provide the element characteristic length in mm."
            )

        i = int(direction)
        if i == 1:
            Ei = self.p["E1"]
            ft = self.p["ft1"]
            Gf = self.p["Gf1t"]
        elif i == 2:
            Ei = self.p["E2"]
            ft = self.p["ft2"]
            Gf = self.p["Gf2t"]
        elif i == 3:
            Ei = self.p["E3"]
            ft = self.p["ft3"]
            Gf = self.p["Gf3t"]
        else:
            raise ValueError("direction must be 1, 2, or 3")

        exponent = (1.0 - F) * self.Lc * ft**2 / (Ei * Gf)
        d = 1.0 - (1.0 / F) * math.exp(exponent)
        return float(np.clip(d, 0.0, 0.999999))

    def _compression_damage_from_F(self, F1c):
        """
        Paper Eq. (36), read directly as:

          d1c = 1 - (1/F1c)(1-A) - A*exp[B(1-F1c)]

        A and B are material constants, but their numerical values are not
        reported in Eslami et al. Table 1. Therefore this branch is disabled
        unless A_comp and B_comp are explicitly provided.
        """
        F1c = float(F1c)

        if F1c <= 1.0:
            return 0.0

        if self.A_comp is None or self.B_comp is None:
            # Do not invent undocumented calibration constants.
            return float(self.damage_c1_commit)

        A = self.A_comp
        B = self.B_comp

        d = (
            1.0
            - (1.0 / F1c) * (1.0 - A)
            - A * math.exp(B * (1.0 - F1c))
        )
        return float(np.clip(d, 0.0, 0.999999))

    def damaged_stiffness(self, d):
        """
        Paper Eq. (34).

        d = [d1, d2, d3] are the regularized directional damage factors.
        """
        d1, d2, d3 = np.clip(np.asarray(d, dtype=float), 0.0, 0.999999)

        C = self.C0
        Cd = np.zeros((6, 6), dtype=float)

        # Normal terms.
        Cd[0, 0] = (1.0 - d1) * C[0, 0]
        Cd[1, 1] = (1.0 - d2) * C[1, 1]
        Cd[2, 2] = (1.0 - d3) * C[2, 2]

        Cd[0, 1] = Cd[1, 0] = (
            (1.0 - d1) * (1.0 - d2) * C[0, 1]
        )
        Cd[0, 2] = Cd[2, 0] = (
            (1.0 - d1) * (1.0 - d3) * C[0, 2]
        )
        Cd[1, 2] = Cd[2, 1] = (
            (1.0 - d2) * (1.0 - d3) * C[1, 2]
        )

        # Shear terms.
        Cd[3, 3] = (
            (1.0 - d1) * (1.0 - d2) * C[3, 3]
        )
        Cd[4, 4] = (
            (1.0 - d1) * (1.0 - d3) * C[4, 4]
        )
        Cd[5, 5] = (
            (1.0 - d2) * (1.0 - d3) * C[5, 5]
        )

        return Cd

    def _update_damage(self, effective_stress):
        """
        Compute irreversible raw damage, then viscously regularize it
        according to paper Eq. (42).
        """
        if not self.use_damage:
            self.damage_t_trial = self.damage_t_commit.copy()
            self.damage_c1_trial = float(self.damage_c1_commit)
            self.damage_v_trial = self.damage_v_commit.copy()
            self.Fmax_t_trial = self.Fmax_t_commit.copy()
            self.Fmax_c1_trial = float(self.Fmax_c1_commit)
            return

        Ft, F1c = self.failure_indices(effective_stress)

        # Maximum historical criteria -> irreversible damage.
        Fmax_t = np.maximum(self.Fmax_t_commit, Ft)
        Fmax_c1 = max(self.Fmax_c1_commit, F1c)

        dt_raw = np.zeros(3)
        for j in range(3):
            dt_raw[j] = self._tensile_damage_from_F(
                Fmax_t[j], direction=j + 1
            )

        # Damage cannot heal.
        dt_raw = np.maximum(self.damage_t_commit, dt_raw)

        dc1_raw = self._compression_damage_from_F(Fmax_c1)
        dc1_raw = max(self.damage_c1_commit, dc1_raw)

        # Directional raw damage.
        # In direction 1, tensile and longitudinal compression damage are
        # both possible. Using the larger irreversible damage variable is a
        # conservative combination; the paper does not state a separate
        # interaction equation for simultaneous d1t and d1c.
        d_raw = np.array(
            [
                max(dt_raw[0], dc1_raw),
                dt_raw[1],
                dt_raw[2],
            ],
            dtype=float,
        )

        # Paper Eq. (42).
        alpha_old = self.eta / (self.eta + self.dt)
        alpha_new = self.dt / (self.eta + self.dt)

        d_v = (
            alpha_old * self.damage_v_commit
            + alpha_new * d_raw
        )

        # Numerical irreversibility.
        d_v = np.maximum(self.damage_v_commit, d_v)
        d_v = np.clip(d_v, 0.0, 0.999999)

        self.damage_t_trial = dt_raw
        self.damage_c1_trial = float(dc1_raw)
        self.damage_v_trial = d_v
        self.Fmax_t_trial = Fmax_t
        self.Fmax_c1_trial = float(Fmax_c1)

    # ================================================================
    # Trial / commit interface
    # ================================================================

    def set_trial_strain(self, strain):
        """
        Source-aligned sequence:

          1. Elastic predictor using UNDAMAGED stiffness.
          2. Hoffman plastic return mapping + hardening in compression.
          3. Update plastic strain and effective stress.
          4. Evaluate damage criteria from effective stress.
          5. Build damaged stiffness Ed.
          6. Calculate final Cauchy stress = Ed * elastic strain.

        This follows the sequence described after paper Eq. (63).
        """
        eps = np.asarray(strain, dtype=float).reshape(6)
        deps = eps - self.strain_commit

        # Paper Eq. (48): predictor is in effective/undamaged stress space.
        sigma_eff_trial_el = (
            self.effective_stress_commit + self.C0 @ deps
        )

        sigma_eff_new, k_new, deps_p, dlambda = (
            self._return_map_isotropic_hardening(sigma_eff_trial_el)
        )

        plastic_strain_new = (
            self.plastic_strain_commit
            + np.asarray(deps_p, dtype=float)
        )

        elastic_strain_new = eps - plastic_strain_new

        # Damage criteria are evaluated after effective stress is updated.
        self._update_damage(sigma_eff_new)

        Cdam = self.damaged_stiffness(self.damage_v_trial)

        # Paper Eq. (33): final damaged/Cauchy stress.
        sigma_cauchy = Cdam @ elastic_strain_new

        self.strain_trial = eps.copy()
        self.plastic_strain_trial = plastic_strain_new.copy()
        self.eq_plastic_strain_trial = float(k_new)
        self.effective_stress_trial = sigma_eff_new.copy()
        self.stress_trial = sigma_cauchy.copy()
        self.Cdam_trial = Cdam.copy()
        self.last_dlambda = float(dlambda)

        # NOTE:
        # Cdam is not yet the exact consistent elastoplastic-damage tangent.
        # It is suitable for standalone response validation, but a production
        # OpenSees NDMaterial should return the algorithmic consistent tangent.
        return self.stress_trial.copy(), self.Cdam_trial.copy()

    def commit_state(self):
        self.strain_commit = self.strain_trial.copy()
        self.plastic_strain_commit = self.plastic_strain_trial.copy()
        self.eq_plastic_strain_commit = float(
            self.eq_plastic_strain_trial
        )
        self.effective_stress_commit = (
            self.effective_stress_trial.copy()
        )
        self.stress_commit = self.stress_trial.copy()

        self.damage_t_commit = self.damage_t_trial.copy()
        self.damage_c1_commit = float(self.damage_c1_trial)
        self.damage_v_commit = self.damage_v_trial.copy()

        self.Fmax_t_commit = self.Fmax_t_trial.copy()
        self.Fmax_c1_commit = float(self.Fmax_c1_trial)

    def revert_to_last_commit(self):
        self.strain_trial = self.strain_commit.copy()
        self.plastic_strain_trial = self.plastic_strain_commit.copy()
        self.eq_plastic_strain_trial = float(
            self.eq_plastic_strain_commit
        )
        self.effective_stress_trial = (
            self.effective_stress_commit.copy()
        )
        self.stress_trial = self.stress_commit.copy()

        self.damage_t_trial = self.damage_t_commit.copy()
        self.damage_c1_trial = float(self.damage_c1_commit)
        self.damage_v_trial = self.damage_v_commit.copy()

        self.Fmax_t_trial = self.Fmax_t_commit.copy()
        self.Fmax_c1_trial = float(self.Fmax_c1_commit)

        self.Cdam_trial = self.damaged_stiffness(
            self.damage_v_trial
        )

    def reset(self):
        self.strain_commit[:] = 0.0
        self.plastic_strain_commit[:] = 0.0
        self.eq_plastic_strain_commit = 0.0
        self.effective_stress_commit[:] = 0.0
        self.stress_commit[:] = 0.0

        self.damage_t_commit[:] = 0.0
        self.damage_c1_commit = 0.0
        self.damage_v_commit[:] = 0.0
        self.Fmax_t_commit[:] = 1.0
        self.Fmax_c1_commit = 1.0

        self.revert_to_last_commit()


def check_paper_strength_points():
    """
    Important sanity check:
    the normalized Hoffman surface should pass through the supplied
    uniaxial tension/compression/shear strengths.
    """
    p = PAPER_TABLE1
    P, Q = hoffman_P_Q(p)

    points = {
        "+ft1": np.array([ p["ft1"], 0, 0, 0, 0, 0]),
        "-fc1": np.array([-p["fc1"], 0, 0, 0, 0, 0]),
        "+ft2": np.array([0, p["ft2"], 0, 0, 0, 0]),
        "-fc2": np.array([0,-p["fc2"], 0, 0, 0, 0]),
        "+ft3": np.array([0, 0, p["ft3"], 0, 0, 0]),
        "-fc3": np.array([0, 0,-p["fc3"], 0, 0, 0]),
        "f12":  np.array([0, 0, 0, p["f12"], 0, 0]),
        "f13":  np.array([0, 0, 0, 0, p["f13"], 0]),
        "f23":  np.array([0, 0, 0, 0, 0, p["f23"]]),
    }

    print("\nHoffman strength-point checks (f should be approximately zero):")
    for name, stress in points.items():
        f = hoffman_value(stress, P, Q)
        print(f"{name:>5s}: f = {f:+.6e}")



def run_uniaxial_stress_test(direction=1, max_compressive_strain=0.10,
                             n_steps=120, stress_tol=1e-7,
                             max_newton=12, fd_eps=1e-7,
                             sigma_e0=1.0, use_hardening=True,
                             use_damage=False, Lc=None, dt=1.0,
                             A_comp=None, B_comp=None,
                             constitutive_max_iter=40,
                             inner_max_iter=20,
                             show_progress=False):
    """
    Uniaxial-stress material test with axial strain control.

    The axial strain in the selected material direction is prescribed, while
    the two transverse NORMAL strains are solved so that the corresponding
    transverse normal stresses are approximately zero.

    This is the appropriate material-point analogue of a free-sided uniaxial
    coupon test:

        direction = 1: prescribe e11, enforce s22 = s33 = 0
        direction = 2: prescribe e22, enforce s11 = s33 = 0
        direction = 3: prescribe e33, enforce s11 = s22 = 0

    All shear strains are kept zero.

    The nonlinear solution is performed at each load step by Newton iteration.
    A finite-difference 2x2 Jacobian is used deliberately here because this is
    a validation driver; it avoids assuming the current tangent is already
    consistent with the plastic return map.
    """
    if direction not in (1, 2, 3):
        raise ValueError("direction must be 1, 2, or 3")

    mat = HoffmanTimber3D(
        sigma_e0=sigma_e0,
        use_hardening=use_hardening,
        use_damage=use_damage,
        Lc=Lc,
        dt=dt,
        A_comp=A_comp,
        B_comp=B_comp,
        max_iter=constitutive_max_iter,
        inner_max_iter=inner_max_iter,
        tol=1e-8,
        inner_tol=1e-9,
    )

    axial_idx = direction - 1
    transverse_idx = [i for i in (0, 1, 2) if i != axial_idx]

    axial_strains = np.linspace(
        0.0, -abs(max_compressive_strain), int(n_steps)
    )

    axial_stress = []
    transverse_stress_1 = []
    transverse_stress_2 = []
    fvals = []
    all_strains = []
    eqp_history = []
    damage_history = []
    effective_stress_history = []

    # Start from the committed lateral strains and use each converged point
    # as the initial guess for the next load step.
    lateral_guess = np.zeros(2, dtype=float)

    for step, e_axial in enumerate(axial_strains):
        if show_progress:
            stride = max(1, len(axial_strains) // 10)
            if step % stride == 0 or step == len(axial_strains) - 1:
                print(
                    f"    direction {direction}: step {step+1}/{len(axial_strains)} "
                    f"({100.0*(step+1)/len(axial_strains):.0f}%)"
                )

        x = lateral_guess.copy()

        converged = False
        for it in range(max_newton):
            # Always evaluate from the SAME committed material state while
            # Newton is searching for the current trial strain vector.
            mat.revert_to_last_commit()

            eps = np.zeros(6, dtype=float)
            eps[axial_idx] = e_axial
            eps[transverse_idx[0]] = x[0]
            eps[transverse_idx[1]] = x[1]

            stress, _ = mat.set_trial_strain(eps)
            r = np.array(
                [stress[transverse_idx[0]], stress[transverse_idx[1]]],
                dtype=float,
            )

            if np.linalg.norm(r, ord=np.inf) <= stress_tol:
                converged = True
                break

            # Numerical Jacobian dr/d[e_transverse].
            J = np.zeros((2, 2), dtype=float)

            for j in range(2):
                x_pert = x.copy()

                # Scale perturbation slightly with the current unknown.
                hfd = max(fd_eps, abs(x[j]) * 1e-6)
                x_pert[j] += hfd

                mat.revert_to_last_commit()

                eps_p = np.zeros(6, dtype=float)
                eps_p[axial_idx] = e_axial
                eps_p[transverse_idx[0]] = x_pert[0]
                eps_p[transverse_idx[1]] = x_pert[1]

                stress_p, _ = mat.set_trial_strain(eps_p)
                r_p = np.array(
                    [
                        stress_p[transverse_idx[0]],
                        stress_p[transverse_idx[1]],
                    ],
                    dtype=float,
                )

                J[:, j] = (r_p - r) / hfd

            try:
                dx = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(J, -r, rcond=None)[0]

            # Mild safeguard against an excessive Newton jump.
            max_jump = 0.05
            dx = np.clip(dx, -max_jump, max_jump)
            x += dx

        if not converged:
            raise RuntimeError(
                f"Uniaxial-stress Newton solve failed at step {step}, "
                f"axial strain={e_axial:.6e}. Last transverse stress "
                f"residual={r} MPa"
            )

        # The current trial state corresponds to the converged eps/stress.
        lateral_guess = x.copy()

        axial_stress.append(stress[axial_idx])
        transverse_stress_1.append(stress[transverse_idx[0]])
        transverse_stress_2.append(stress[transverse_idx[1]])
        fvals.append(mat.yield_function(stress))
        all_strains.append(eps.copy())
        eqp_history.append(float(mat.eq_plastic_strain_trial))
        damage_history.append(mat.damage_v_trial.copy())
        effective_stress_history.append(mat.effective_stress_trial.copy())

        mat.commit_state()

    return {
        "axial_strain": np.asarray(axial_strains),
        "axial_stress": np.asarray(axial_stress),
        "transverse_stress_1": np.asarray(transverse_stress_1),
        "transverse_stress_2": np.asarray(transverse_stress_2),
        "yield_function": np.asarray(fvals),
        "strain_history": np.asarray(all_strains),
        "eq_plastic_strain": np.asarray(eqp_history),
        "damage": np.asarray(damage_history),
        "effective_stress": np.asarray(effective_stress_history),
        "direction": int(direction),
        "transverse_indices": tuple(transverse_idx),
    }



def run_uniaxial_tension_test(
    direction=1,
    max_tensile_strain=0.03,
    n_steps=400,
    stress_tol=1e-8,
    max_newton=30,
    fd_eps=1e-8,
    sigma_e0=1.0,
    Lc=None,
    dt=1.0,
):
    """
    Free-sided uniaxial TENSION material test.

    Plasticity is disabled automatically for pure tensile states, while the
    paper's tensile damage model remains active.

    The prescribed axial strain is positive. The two transverse normal strains
    are solved such that the corresponding Cauchy stresses remain ~0.
    """
    if direction not in (1, 2, 3):
        raise ValueError("direction must be 1, 2, or 3")

    mat = HoffmanTimber3D(
        sigma_e0=sigma_e0,
        use_hardening=True,
        use_damage=True,
        Lc=Lc,
        dt=dt,
    )

    axial_idx = direction - 1
    transverse_idx = [i for i in (0, 1, 2) if i != axial_idx]

    axial_strains = np.linspace(
        0.0, abs(max_tensile_strain), int(n_steps)
    )

    axial_stress = []
    damage_history = []
    effective_stress_history = []
    lateral_guess = np.zeros(2)

    for step, e_axial in enumerate(axial_strains):
        x = lateral_guess.copy()
        converged = False

        for _ in range(max_newton):
            mat.revert_to_last_commit()

            eps = np.zeros(6)
            eps[axial_idx] = e_axial
            eps[transverse_idx[0]] = x[0]
            eps[transverse_idx[1]] = x[1]

            stress, _ = mat.set_trial_strain(eps)

            r = np.array(
                [
                    stress[transverse_idx[0]],
                    stress[transverse_idx[1]],
                ]
            )

            if np.linalg.norm(r, ord=np.inf) <= stress_tol:
                converged = True
                break

            J = np.zeros((2, 2))

            for j in range(2):
                x_pert = x.copy()
                hfd = max(fd_eps, abs(x[j]) * 1e-6)
                x_pert[j] += hfd

                mat.revert_to_last_commit()

                eps_p = np.zeros(6)
                eps_p[axial_idx] = e_axial
                eps_p[transverse_idx[0]] = x_pert[0]
                eps_p[transverse_idx[1]] = x_pert[1]

                stress_p, _ = mat.set_trial_strain(eps_p)

                r_p = np.array(
                    [
                        stress_p[transverse_idx[0]],
                        stress_p[transverse_idx[1]],
                    ]
                )

                J[:, j] = (r_p - r) / hfd

            try:
                dx = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(J, -r, rcond=None)[0]

            dx = np.clip(dx, -0.05, 0.05)
            x += dx

        if not converged:
            raise RuntimeError(
                f"Tension transverse-stress solve failed at step {step}, "
                f"axial strain={e_axial:.6e}; residual={r} MPa"
            )

        lateral_guess = x.copy()

        axial_stress.append(stress[axial_idx])
        damage_history.append(mat.damage_v_trial.copy())
        effective_stress_history.append(
            mat.effective_stress_trial.copy()
        )

        mat.commit_state()

    return {
        "axial_strain": np.asarray(axial_strains),
        "axial_stress": np.asarray(axial_stress),
        "damage": np.asarray(damage_history),
        "effective_stress": np.asarray(effective_stress_history),
        "direction": direction,
    }


def plot_damage_demo(Lc=5.0, dt=1.0):
    """
    Demonstration of the published tensile-damage law.

    Lc=5 mm is ONLY a demonstration choice here, not a value reported in
    Eslami et al. Table 1. Use the actual characteristic length of your
    continuum/fiber discretization for quantitative work.
    """
    r = run_uniaxial_tension_test(
        direction=1,
        max_tensile_strain=0.03,
        n_steps=400,
        Lc=Lc,
        dt=dt,
    )

    plt.figure()
    plt.plot(
        r["axial_strain"],
        r["axial_stress"],
        label="Cauchy stress with tensile damage",
    )
    plt.axhline(
        PAPER_TABLE1["ft1"],
        linestyle="--",
        label=f"ft1 = {PAPER_TABLE1['ft1']} MPa",
    )
    plt.xlabel("Tensile strain")
    plt.ylabel("Tensile stress (MPa)")
    plt.title(
        f"Longitudinal tensile damage demo (Lc={Lc:g} mm)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(
        r["axial_strain"],
        r["damage"][:, 0],
        label="regularized d1",
    )
    plt.xlabel("Tensile strain")
    plt.ylabel("Damage d1")
    plt.title("Longitudinal tensile damage evolution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

def plot_validation_curves(sigma_e0=SIGMA_E0_CALIBRATED_MPA, n_steps=120):
    """
    FAST validation mode.

    Uses a modest number of steps and relaxed solver settings so the
    constitutive response can be inspected quickly. Increase n_steps and
    tolerances only after the formulation is validated.
    """

    print(f"\nUsing sigma_e0 = {sigma_e0:.6g} MPa for hardening validation.")
    print("FAST validation mode is active.")
    print(f"Number of strain steps per analysis = {n_steps}")

    print("\n[1/4] Parallel-to-grain perfect plasticity...")
    r1_pp = run_uniaxial_stress_test(
        direction=1,
        max_compressive_strain=0.10,
        n_steps=n_steps,
        sigma_e0=sigma_e0,
        use_hardening=False,
        use_damage=False,
        show_progress=True,
    )

    print("\n[2/4] Parallel-to-grain hardening...")
    r1_h = run_uniaxial_stress_test(
        direction=1,
        max_compressive_strain=0.10,
        n_steps=n_steps,
        sigma_e0=sigma_e0,
        use_hardening=True,
        use_damage=False,
        show_progress=True,
    )

    plt.figure()
    plt.plot(-r1_pp["axial_strain"], -r1_pp["axial_stress"],
             label="Perfect plasticity")
    plt.plot(-r1_h["axial_strain"], -r1_h["axial_stress"],
             label="Isotropic hardening (h=1200 MPa)")
    plt.axhline(PAPER_TABLE1["fc1"], linestyle="--",
                label=f"initial fc1 = {PAPER_TABLE1['fc1']} MPa")
    plt.xlabel("Compressive strain")
    plt.ylabel("Compressive stress (MPa)")
    plt.title("Compression parallel to grain")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    print("\n[3/4] Perpendicular-to-grain perfect plasticity...")
    r2_pp = run_uniaxial_stress_test(
        direction=2,
        max_compressive_strain=0.12,
        n_steps=n_steps,
        sigma_e0=sigma_e0,
        use_hardening=False,
        use_damage=False,
        show_progress=True,
    )

    print("\n[4/4] Perpendicular-to-grain hardening...")
    r2_h = run_uniaxial_stress_test(
        direction=2,
        max_compressive_strain=0.12,
        n_steps=n_steps,
        sigma_e0=sigma_e0,
        use_hardening=True,
        use_damage=False,
        show_progress=True,
    )

    plt.figure()
    plt.plot(-r2_pp["axial_strain"], -r2_pp["axial_stress"],
             label="Perfect plasticity")
    plt.plot(-r2_h["axial_strain"], -r2_h["axial_stress"],
             label="Isotropic hardening (h=1200 MPa)")
    plt.axhline(PAPER_TABLE1["fc2"], linestyle="--",
                label=f"initial fc2 = {PAPER_TABLE1['fc2']} MPa")
    plt.xlabel("Compressive strain")
    plt.ylabel("Compressive stress (MPa)")
    plt.title("Compression perpendicular to grain")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    max_t1 = max(
        np.max(np.abs(r1_h["transverse_stress_1"])),
        np.max(np.abs(r1_h["transverse_stress_2"])),
    )
    max_t2 = max(
        np.max(np.abs(r2_h["transverse_stress_1"])),
        np.max(np.abs(r2_h["transverse_stress_2"])),
    )

    print("\nCompleted hardening validation.")
    print(f"  Direction 1 max |transverse stress| = {max_t1:.3e} MPa")
    print(f"  Direction 2 max |transverse stress| = {max_t2:.3e} MPa")
    print(f"  Direction 1 final axial stress = {r1_h['axial_stress'][-1]:.6f} MPa")
    print(f"  Direction 2 final axial stress = {r2_h['axial_stress'][-1]:.6f} MPa")
    print(f"  Direction 1 final eqp = {r1_h['eq_plastic_strain'][-1]:.6e}")
    print(f"  Direction 2 final eqp = {r2_h['eq_plastic_strain'][-1]:.6e}")

    plt.show()



def plot_full_damage_validation(
    Lc_mm=5.0,
    dt=1.0,
    n_steps_comp=120,
    n_steps_tension=160,
):
    """
    Full standalone constitutive check with DAMAGE ACTIVE.

    Runs:
      1) compression parallel to grain:
         hardening only vs hardening + longitudinal compression damage

      2) compression perpendicular to grain:
         hardening response (no longitudinal compression-damage branch)

      3) tension parallel to grain:
         d1t fracture-energy damage

      4) tension perpendicular to grain:
         d2t fracture-energy damage

    Lc_mm=5 mm is only a demonstration/default numerical value.
    For quantitative use, replace it by the appropriate characteristic
    length for the material discretization.
    """

    print("\n============================================================")
    print("FULL HARDENING + DAMAGE VALIDATION")
    print("============================================================")
    print(f"sigma_e0 = {SIGMA_E0_CALIBRATED_MPA:.3f} MPa")
    print(f"A_comp    = {A_COMP_CALIBRATED:.3f}")
    print(f"B_comp    = {B_COMP_CALIBRATED:.3f}")
    print(f"Lc        = {Lc_mm:.3f} mm")
    print(f"eta       = {PAPER_TABLE1['eta']:.6g}")
    print("============================================================")

    # ----------------------------------------------------------
    # Compression parallel: compare hardening vs compression damage
    # ----------------------------------------------------------
    print("\n[1/4] Compression parallel to grain with d1c...")

    r1_h = run_uniaxial_stress_test(
        direction=1,
        max_compressive_strain=0.095,
        n_steps=n_steps_comp,
        stress_tol=1e-6,
        sigma_e0=SIGMA_E0_CALIBRATED_MPA,
        use_hardening=True,
        use_damage=False,
        show_progress=True,
        constitutive_max_iter=30,
        inner_max_iter=12,
    )

    r1_d = run_uniaxial_stress_test(
        direction=1,
        max_compressive_strain=0.095,
        n_steps=n_steps_comp,
        stress_tol=1e-6,
        sigma_e0=SIGMA_E0_CALIBRATED_MPA,
        use_hardening=True,
        use_damage=True,
        Lc=Lc_mm,
        dt=dt,
        A_comp=A_COMP_CALIBRATED,
        B_comp=B_COMP_CALIBRATED,
        show_progress=True,
        constitutive_max_iter=30,
        inner_max_iter=12,
    )

    plt.figure()
    plt.plot(
        -r1_h["axial_strain"],
        -r1_h["axial_stress"],
        label="Hoffman + hardening only",
    )
    plt.plot(
        -r1_d["axial_strain"],
        -r1_d["axial_stress"],
        label="Hardening + longitudinal compression damage d1c",
    )
    plt.axhline(
        PAPER_TABLE1["fc1"],
        linestyle="--",
        label=f"fc1 = {PAPER_TABLE1['fc1']} MPa",
    )
    plt.xlabel("Compressive strain")
    plt.ylabel("Compressive stress (MPa)")
    plt.title("Compression parallel to grain: effect of d1c")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(
        -r1_d["axial_strain"],
        r1_d["damage"][:, 0],
        label="regularized directional damage d1",
    )
    plt.xlabel("Compressive strain")
    plt.ylabel("Damage")
    plt.title("Longitudinal compression damage evolution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ----------------------------------------------------------
    # Compression perpendicular: hardening response
    # ----------------------------------------------------------
    print("\n[2/4] Compression perpendicular to grain...")

    r2 = run_uniaxial_stress_test(
        direction=2,
        max_compressive_strain=0.12,
        n_steps=n_steps_comp,
        stress_tol=1e-6,
        sigma_e0=SIGMA_E0_CALIBRATED_MPA,
        use_hardening=True,
        use_damage=True,
        Lc=Lc_mm,
        dt=dt,
        A_comp=A_COMP_CALIBRATED,
        B_comp=B_COMP_CALIBRATED,
        show_progress=True,
        constitutive_max_iter=30,
        inner_max_iter=12,
    )

    plt.figure()
    plt.plot(
        -r2["axial_strain"],
        -r2["axial_stress"],
        label="Calibrated model",
    )
    plt.axhline(
        PAPER_TABLE1["fc2"],
        linestyle="--",
        label=f"fc2 = {PAPER_TABLE1['fc2']} MPa",
    )
    plt.xlabel("Compressive strain")
    plt.ylabel("Compressive stress (MPa)")
    plt.title("Compression perpendicular to grain")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ----------------------------------------------------------
    # Tension parallel: d1t
    # ----------------------------------------------------------
    print("\n[3/4] Tension parallel to grain with d1t...")

    t1 = run_uniaxial_tension_test(
        direction=1,
        max_tensile_strain=0.025,
        n_steps=n_steps_tension,
        sigma_e0=SIGMA_E0_CALIBRATED_MPA,
        Lc=Lc_mm,
        dt=dt,
    )

    plt.figure()
    plt.plot(
        t1["axial_strain"],
        t1["axial_stress"],
        label="Tension parallel + d1t",
    )
    plt.axhline(
        PAPER_TABLE1["ft1"],
        linestyle="--",
        label=f"ft1 = {PAPER_TABLE1['ft1']} MPa",
    )
    plt.xlabel("Tensile strain")
    plt.ylabel("Tensile stress (MPa)")
    plt.title("Tension parallel to grain: tensile damage")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(
        t1["axial_strain"],
        t1["damage"][:, 0],
        label="d1t / regularized d1",
    )
    plt.xlabel("Tensile strain")
    plt.ylabel("Damage")
    plt.title("Parallel tensile damage evolution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ----------------------------------------------------------
    # Tension perpendicular: d2t
    # ----------------------------------------------------------
    print("\n[4/4] Tension perpendicular to grain with d2t...")

    t2 = run_uniaxial_tension_test(
        direction=2,
        max_tensile_strain=0.012,
        n_steps=n_steps_tension,
        sigma_e0=SIGMA_E0_CALIBRATED_MPA,
        Lc=Lc_mm,
        dt=dt,
    )

    plt.figure()
    plt.plot(
        t2["axial_strain"],
        t2["axial_stress"],
        label="Tension perpendicular + d2t",
    )
    plt.axhline(
        PAPER_TABLE1["ft2"],
        linestyle="--",
        label=f"ft2 = {PAPER_TABLE1['ft2']} MPa",
    )
    plt.xlabel("Tensile strain")
    plt.ylabel("Tensile stress (MPa)")
    plt.title("Tension perpendicular to grain: tensile damage")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(
        t2["axial_strain"],
        t2["damage"][:, 1],
        label="d2t / regularized d2",
    )
    plt.xlabel("Tensile strain")
    plt.ylabel("Damage")
    plt.title("Perpendicular tensile damage evolution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    eps1 = -r1_d["axial_strain"]
    sig1 = -r1_d["axial_stress"]
    eps2 = -r2["axial_strain"]
    sig2 = -r2["axial_stress"]

    print("\n---------------- RESPONSE SUMMARY ----------------")
    print(
        f"Parallel compression peak = {np.max(sig1):.4f} MPa "
        f"at strain {eps1[np.argmax(sig1)]:.5f}"
    )
    print(
        f"Parallel compression final stress = {sig1[-1]:.4f} MPa"
    )
    print(
        f"Parallel final damage d1 = {r1_d['damage'][-1,0]:.6f}"
    )
    print(
        f"Perpendicular compression final stress = {sig2[-1]:.4f} MPa"
    )
    print(
        f"Parallel tension peak = {np.max(t1['axial_stress']):.4f} MPa"
    )
    print(
        f"Parallel tension final d1 = {t1['damage'][-1,0]:.6f}"
    )
    print(
        f"Perpendicular tension peak = {np.max(t2['axial_stress']):.4f} MPa"
    )
    print(
        f"Perpendicular tension final d2 = {t2['damage'][-1,1]:.6f}"
    )
    print("--------------------------------------------------")

    plt.show()

    return {
        "compression_parallel": r1_d,
        "compression_perpendicular": r2,
        "tension_parallel": t1,
        "tension_perpendicular": t2,
    }


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)

    print("Paper Table 1 parameters:")
    for k, v in PAPER_TABLE1.items():
        print(f"  {k:>5s} = {v}")

    C = orthotropic_stiffness(PAPER_TABLE1)
    print("\nOrthotropic stiffness matrix C [MPa]:")
    print(C)

    check_paper_strength_points()

    # Initial elastic estimates for uniaxial strain-controlled loading.
    print("\nNominal material strengths from Table 1:")
    print(f"  fc1 = {PAPER_TABLE1['fc1']} MPa")
    print(f"  fc2 = {PAPER_TABLE1['fc2']} MPa")
    print(f"  ft1 = {PAPER_TABLE1['ft1']} MPa")
    print(f"  ft2 = {PAPER_TABLE1['ft2']} MPa")

    print(f"\nApprox. Fig. 4(b) T2 = {T2_FIG4B_APPROX_MPA:.3f} MPa")
    print(f"Calibrated sigma_e0 = {SIGMA_E0_FIG4B_APPROX_MPA:.3f} MPa")
    plot_full_damage_validation(
        Lc_mm=5.0,          # demonstration value; replace for production use
        dt=1.0,
        n_steps_comp=120,
        n_steps_tension=160,
    )
