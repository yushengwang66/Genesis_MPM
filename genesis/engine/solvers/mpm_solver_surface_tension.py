import numpy as np
import quadrants as qd

import genesis as gs
from .mpm_solver import MPMSolver as _BaseMPMSolver


@qd.data_oriented
class MPMSolver(_BaseMPMSolver):
    """MPM solver extension with optional SPH-style numerical surface tension.

    Surface tension is enabled only when at least one MPM material has gamma > 0.
    With gamma=0, this class follows the original MPMSolver path.
    """

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)
        self._has_surface_tension = False
        self._material_gammas = []
        self._surface_tension_max_acc = 500.0

    def add_material(self, material):
        super().add_material(material)
        gamma = float(getattr(material, "gamma", 0.0))
        while len(self._material_gammas) <= material.idx:
            self._material_gammas.append(0.0)
        self._material_gammas[material.idx] = gamma
        if gamma > 0.0:
            self._has_surface_tension = True

    def build(self):
        super().build()
        if self.is_active and self._has_surface_tension:
            shape = (self._sim.substeps_local, *self._grid_res, self._B)
            self.surface_color = qd.field(dtype=gs.qd_float, shape=shape)
            self.surface_gamma_num = qd.field(dtype=gs.qd_float, shape=shape)
            self.surface_gamma_den = qd.field(dtype=gs.qd_float, shape=shape)
            self.surface_grad_c = qd.Vector.field(3, dtype=gs.qd_float, shape=shape)
            self.surface_normal = qd.Vector.field(3, dtype=gs.qd_float, shape=shape)
            self.surface_curvature = qd.field(dtype=gs.qd_float, shape=shape)

    def substep_pre_coupling(self, f):
        super().substep_pre_coupling(f)
        if self._has_surface_tension and not self._sim.requires_grad:
            self.reset_surface_fields(f)
            self.project_surface_color(f)
            self.compute_surface_normal(f)
            self.compute_surface_curvature(f)
            self.apply_surface_tension_to_grid(f)

    @qd.kernel
    def reset_surface_fields(self, f: qd.i32):
        for i, j, k, i_b in qd.ndrange(*self._grid_res, self._B):
            self.surface_color[f, i, j, k, i_b] = gs.qd_float(0.0)
            self.surface_gamma_num[f, i, j, k, i_b] = gs.qd_float(0.0)
            self.surface_gamma_den[f, i, j, k, i_b] = gs.qd_float(0.0)
            self.surface_grad_c[f, i, j, k, i_b] = qd.Vector.zero(gs.qd_float, 3)
            self.surface_normal[f, i, j, k, i_b] = qd.Vector.zero(gs.qd_float, 3)
            self.surface_curvature[f, i, j, k, i_b] = gs.qd_float(0.0)

    @qd.kernel
    def project_surface_color(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[f, i_p, i_b].active:
                gamma = gs.qd_float(0.0)
                for mat_idx in qd.static(self._materials_idx):
                    if self.particles_info[i_p].material_idx == mat_idx:
                        gamma = gs.qd_float(self._material_gammas[mat_idx])

                if gamma > gs.qd_float(0.0):
                    base = qd.floor(self.particles[f, i_p, i_b].pos * self._inv_dx - 0.5).cast(gs.qd_int)
                    fx = self.particles[f, i_p, i_b].pos * self._inv_dx - base.cast(gs.qd_float)
                    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
                    for offset in qd.static(qd.grouped(self.stencil_range())):
                        cell_ijk = base - self._grid_offset + offset
                        weight = gs.qd_float(1.0)
                        for d in qd.static(range(3)):
                            weight *= w[offset[d]][d]
                        color_contrib = weight * self._particle_volume_real * self._inv_dx * self._inv_dx * self._inv_dx
                        qd.atomic_add(self.surface_color[f, cell_ijk, i_b], color_contrib)
                        qd.atomic_add(self.surface_gamma_num[f, cell_ijk, i_b], color_contrib * gamma)
                        qd.atomic_add(self.surface_gamma_den[f, cell_ijk, i_b], color_contrib)

    @qd.kernel
    def compute_surface_normal(self, f: qd.i32):
        for ii, jj, kk, i_b in qd.ndrange(*self._grid_res, self._B):
            if (
                ii > 0
                and jj > 0
                and kk > 0
                and ii < self._grid_res[0] - 1
                and jj < self._grid_res[1] - 1
                and kk < self._grid_res[2] - 1
            ):
                grad_c = qd.Vector(
                    [
                        (self.surface_color[f, ii + 1, jj, kk, i_b] - self.surface_color[f, ii - 1, jj, kk, i_b])
                        * 0.5
                        * self._inv_dx,
                        (self.surface_color[f, ii, jj + 1, kk, i_b] - self.surface_color[f, ii, jj - 1, kk, i_b])
                        * 0.5
                        * self._inv_dx,
                        (self.surface_color[f, ii, jj, kk + 1, i_b] - self.surface_color[f, ii, jj, kk - 1, i_b])
                        * 0.5
                        * self._inv_dx,
                    ],
                    dt=gs.qd_float,
                )
                norm = grad_c.norm(gs.EPS)
                self.surface_grad_c[f, ii, jj, kk, i_b] = grad_c
                if norm > gs.qd_float(1e-6):
                    self.surface_normal[f, ii, jj, kk, i_b] = grad_c / norm

    @qd.kernel
    def compute_surface_curvature(self, f: qd.i32):
        for ii, jj, kk, i_b in qd.ndrange(*self._grid_res, self._B):
            if (
                ii > 0
                and jj > 0
                and kk > 0
                and ii < self._grid_res[0] - 1
                and jj < self._grid_res[1] - 1
                and kk < self._grid_res[2] - 1
            ):
                div_n = (
                    self.surface_normal[f, ii + 1, jj, kk, i_b][0]
                    - self.surface_normal[f, ii - 1, jj, kk, i_b][0]
                    + self.surface_normal[f, ii, jj + 1, kk, i_b][1]
                    - self.surface_normal[f, ii, jj - 1, kk, i_b][1]
                    + self.surface_normal[f, ii, jj, kk + 1, i_b][2]
                    - self.surface_normal[f, ii, jj, kk - 1, i_b][2]
                ) * 0.5 * self._inv_dx
                self.surface_curvature[f, ii, jj, kk, i_b] = -div_n

    @qd.kernel
    def apply_surface_tension_to_grid(self, f: qd.i32):
        for ii, jj, kk, i_b in qd.ndrange(*self._grid_res, self._B):
            I = (ii, jj, kk)
            if self.grid[f, I, i_b].mass > gs.EPS and self.surface_gamma_den[f, I, i_b] > gs.EPS:
                gamma = self.surface_gamma_num[f, I, i_b] / self.surface_gamma_den[f, I, i_b]
                grad_c = self.surface_grad_c[f, I, i_b]
                grad_norm = grad_c.norm(gs.EPS)

                if gamma > gs.qd_float(0.0) and grad_norm > gs.qd_float(1e-6):
                    normal = grad_c / grad_norm
                    curvature = self.surface_curvature[f, I, i_b]

                    # SPH-like numerical surface tension / cohesion. Here gamma is an acceleration-scale control,
                    # not a physical N/m coefficient. This makes MPM.Liquid(gamma=...) behave closer to SPH.Liquid.
                    a_st = gamma * curvature * normal

                    a_norm = a_st.norm(gs.EPS)
                    max_a = gs.qd_float(self._surface_tension_max_acc)
                    if a_norm > max_a:
                        a_st = a_st / a_norm * max_a

                    self.grid[f, I, i_b].vel_in += self.grid[f, I, i_b].mass * self.substep_dt * a_st
