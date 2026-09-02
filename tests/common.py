
import numpy as np
import openseespy.opensees as ops

PARAMS = dict(
    E1=2050.8, E2=172.1, E3=172.1,
    nu12=0.45, nu13=0.45, nu23=0.50,
    G12=145.2, G13=145.2, G23=68.0,
    fc1=35.0, fc2=2.5, fc3=2.5,
    ft1=20.0, ft2=0.7, ft3=0.7,
    f12=5.0, f13=5.0, f23=0.5,
    h=1200.0, sigmaE0=270.0,
    Acomp=0.60, Bcomp=0.50,
    Gf1t=60.0, Gf2t=0.5, Gf3t=0.5,
    eta=1.0e-4, Lc=5.0, dt=1.0,
)

def define_material(tag=1):
    p = PARAMS
    ops.nDMaterial(
        "TimberHoffman3D", tag,
        p["E1"], p["E2"], p["E3"],
        p["nu12"], p["nu13"], p["nu23"],
        p["G12"], p["G13"], p["G23"],
        p["fc1"], p["fc2"], p["fc3"],
        p["ft1"], p["ft2"], p["ft3"],
        p["f12"], p["f13"], p["f23"],
        p["h"], p["sigmaE0"], p["Acomp"], p["Bcomp"],
        p["Gf1t"], p["Gf2t"], p["Gf3t"],
        p["eta"], p["Lc"], p["dt"]
    )
    ops.testNDMaterial(tag)

def trial_stress(eps):
    ops.setTrialStrain(*[float(v) for v in eps])
    return np.asarray(ops.getStress(), dtype=float)

def solve_free_sided_step(axial, direction, lateral_guess,
                          tol=1e-7, max_iter=80, fd=1e-7):
    ai = direction - 1
    tr = [i for i in (0,1,2) if i != ai]
    x = np.asarray(lateral_guess, dtype=float).copy()

    for _ in range(max_iter):
        eps = np.zeros(6)
        eps[ai] = axial
        eps[tr[0]], eps[tr[1]] = x

        s = trial_stress(eps)
        r = np.array([s[tr[0]], s[tr[1]]])

        if np.linalg.norm(r, np.inf) <= tol:
            return eps, s, x

        J = np.zeros((2,2))
        for j in range(2):
            xp = x.copy()
            hfd = max(fd, abs(x[j])*1e-6)
            xp[j] += hfd

            ep = np.zeros(6)
            ep[ai] = axial
            ep[tr[0]], ep[tr[1]] = xp
            sp = trial_stress(ep)
            rp = np.array([sp[tr[0]], sp[tr[1]]])
            J[:,j] = (rp-r)/hfd

        try:
            dx = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(J, -r, rcond=None)[0]

        # Damage can make the numerical Jacobian poorly conditioned.  Use a
        # short backtracking line search instead of accepting a full Newton
        # step that increases the transverse-stress residual.
        dx = np.clip(dx, -0.05, 0.05)
        r_norm = np.linalg.norm(r, np.inf)
        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
            candidate = x + alpha*dx
            ec = np.zeros(6)
            ec[ai] = axial
            ec[tr[0]], ec[tr[1]] = candidate
            rc = np.asarray(trial_stress(ec))[tr]
            if np.linalg.norm(rc, np.inf) < r_norm:
                x = candidate
                accepted = True
                break

        if not accepted:
            x += 0.0625*dx

    raise RuntimeError(f"Free-sided solve failed at axial strain {axial}")

def run_history(direction, axial_history):
    lateral = np.zeros(2)
    strain_hist, stress_hist = [], []

    for axial in axial_history:
        eps, stress, lateral = solve_free_sided_step(
            axial, direction, lateral
        )
        strain_hist.append(eps.copy())
        stress_hist.append(stress.copy())
        ops.commitStrain()

    return np.asarray(strain_hist), np.asarray(stress_hist)
