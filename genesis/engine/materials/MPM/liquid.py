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
        Surface-tension / cohesion strength used by the optional MPM surface-tension extension.
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
    gamma: NonNegativeFloat = 0.0

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
