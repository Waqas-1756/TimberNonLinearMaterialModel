# TimberNonLinearMaterialModel
TimberHoffman3D is a 3D orthotropic continuum material for timber, implemented as an OpenSees nDMaterial. It combines orthotropic linear elasticity, a Hoffman failure surface with isotropic hardening in compression, and directional tensile/compressive damage with viscous regularization.
The model accepts elastic constants, Poisson ratios, shear moduli, longitudinal/transverse strengths, shear strengths, hardening parameters, fracture energies, viscosity, and characteristic length. It supports direct OpenSeesPy material testing through testNDMaterial, setTrialStrain, getStress, getTangent, and commitStrain.
Validation included:
- Parallel and perpendicular compression benchmarks
- Parallel and perpendicular tension benchmarks
- Elastic tangent finite-difference verification
- Step-size sensitivity from 60 to 480 strain increments
All four supplied benchmark tests pass. The elastic tangent check has relative error 6.64e-13, and the step-size study shows stable peak and final stresses.
Known limitation: getTangent() currently returns the damaged elastic stiffness, not a fully consistent elastoplastic-damage algorithmic tangent. Further element-level and nonlinear dynamic validation is recommended before production-scale analyses.
