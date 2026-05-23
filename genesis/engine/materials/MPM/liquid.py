from typing import Any

import quadrants as qd
from pydantic import StrictBool

import genesis as gs
from genesis.typing import NonNegativeFloat

from .base import Base


@qd.data_oriented
class Liquid(Base):
    """
    The liquid material class for MPM.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m³). Default is 1000.
    viscous : bool, optional
        Whether the liquid is viscous. Simply sets mu to zero when non-viscous. Default is False.
    gamma : float, optional
        Backward-compatible SPH-like surface/cohesion strength. If the split controls below are left at 0.0, ``gamma``
        is used for both particle cohesion and grid surface tension.
    particle_cohesion_gamma : float, optional
        SPH-like particle-particle cohesion strength. Use this to tune droplet clustering independently of grid surface
        tension. Defaults to 0.0, which falls back to ``gamma``.
    surface_tension_gamma : float, optional
        Grid color-field surface-tension strength. Use this to tune free-surface restoration independently of particle
        cohesion. Defaults to 0.0, which falls back to ``gamma``.
    ground_horizontal_damping : float, optional
        Near-ground horizontal damping used by the MPM surface-tension extension. Set to a value in [0, 1] manually in
        practice. Defaults to 0.0, which makes the solver use its global fallback damping.
    sph_like : bool, optional
        Whether to interpret ``stiffness``, ``exponent``, ``mu`` and ``gamma`` using SPH-style numerical parameters.
        When enabled, ``lam`` is set from the linearized WCSPH bulk modulus ``stiffness * exponent`` and viscosity is
        enabled automatically.
    stiffness : float, optional
        SPH-style stiffness used only when ``sph_like=True``.
    exponent : float, optional
        SPH-style equation-of-state exponent used only when ``sph_like=True``. Default is 7.0.
    """

    viscous: StrictBool = False

    # Backward-compatible combined control. Existing scripts that pass gamma=... continue to work.
    gamma: NonNegativeFloat = 0.0

    # Split controls for the enhanced MPM solver. A value of 0.0 means "inherit from gamma" for cohesion/surface terms.
    particle_cohesion_gamma: NonNegativeFloat = 0.0
    surface_tension_gamma: NonNegativeFloat = 0.0
    ground_horizontal_damping: NonNegativeFloat = 0.0

    # SPH-style compatibility parameters. These let MPM.Liquid accept the same high-level controls as SPH.Liquid.
    sph_like: StrictBool = False
    stiffness: NonNegativeFloat = 0.0
    exponent: NonNegativeFloat = 7.0

    def model_post_init(self, context: Any) -> None:
        if self.sph_like:
            # WCSPH linearization around rho=rho0:
            # p = stiffness * ((rho/rho0)^exponent - 1)
            # K = rho0 * dp/drho = stiffness * exponent
            # MPM liquid uses the volumetric stress coefficient lam for the same role.
            self.lam = self.stiffness * self.exponent
            self.viscous = True

        # Backward compatibility: if the new split knobs are not specified, inherit the old gamma value.
        if self.particle_cohesion_gamma == 0.0:
            self.particle_cohesion_gamma = self.gamma
        if self.surface_tension_gamma == 0.0:
            self.surface_tension_gamma = self.gamma

        super().model_post_init(context)

        if not self.viscous:
            self.mu = 0.0

        self.update_F_S_Jp = self._update_F_S_Jp_liquid
        self.update_stress = self._update_stress_liquid
        # Viscous liquid uses U @ V.T in the 2*mu*(F_tmp - U@V.T)@F_tmp.T term; when mu==0 that term vanishes and SVD
        # is pure waste.
        self.needs_svd = self.viscous

    @qd.func
    def _update_F_S_Jp_liquid(self, J, F_tmp, U, S, V, Jp):
        F_new = qd.Matrix.identity(gs.qd_float, 3) * qd.pow(J, 1.0 / 3.0)
        S_new = S
        Jp_new = Jp
        return F_new, S_new, Jp_new

    @qd.func
    def _update_stress_liquid(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        stress = 2 * self.mu * (F_tmp - U @ V.transpose()) @ F_tmp.transpose() + qd.Matrix.identity(
            gs.qd_float, 3
        ) * self.lam * J * (J - 1)
        return stress
