import numpy as np
import quadrants as qd

import genesis as gs
from .mpm_solver import MPMSolver as _BaseMPMSolver


@qd.data_oriented
class MPMSolver(_BaseMPMSolver):
    """MPM solver extension with optional SPH-style cohesion/surface-tension controls.

    Enhanced controls:
    1. particle-particle short-range cohesion before p2g,
    2. grid color-field surface tension with a 26-neighbor isotropic stencil,
    3. near-ground horizontal damping to reduce contact-line spreading.

    Backward compatibility:
    - Existing materials using ``gamma`` still work.
    - New materials can independently set ``particle_cohesion_gamma`` and ``surface_tension_gamma``.
    """

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)
        self._has_surface_tension = False
        self._material_particle_cohesion_gammas = []
        self._material_surface_tension_gammas = []
        self._material_ground_horizontal_dampings = []

        # Safety clamps for the added numerical forces.
        self._surface_tension_max_acc = 500.0
        self._particle_cohesion_max_acc = 500.0

        # Particle-particle cohesion support radius. A radius of ~2.5 particle spacings usually captures enough
        # neighbors for a compact droplet while keeping the O(N^2) fallback acceptable for small ferrofluid robots.
        self._particle_cohesion_radius_factor = 2.5

        # Contact-line damping near a flat ground plane at z=0. This is intentionally conservative: it damps horizontal
        # momentum only in a thin band close to the ground, reducing spreading without pinning the whole droplet.
        self._ground_z = 0.0
        self._ground_damping_height_factor = 3.0
        self._ground_horizontal_damping = 0.65

    def add_material(self, material):
        super().add_material(material)
        particle_gamma = float(getattr(material, "particle_cohesion_gamma", getattr(material, "gamma", 0.0)))
        surface_gamma = float(getattr(material, "surface_tension_gamma", getattr(material, "gamma", 0.0)))
        ground_damping = float(getattr(material, "ground_horizontal_damping", 0.0))

        while len(self._material_particle_cohesion_gammas) <= material.idx:
            self._material_particle_cohesion_gammas.append(0.0)
            self._material_surface_tension_gammas.append(0.0)
            self._material_ground_horizontal_dampings.append(0.0)

        self._material_particle_cohesion_gammas[material.idx] = particle_gamma
        self._material_surface_tension_gammas[material.idx] = surface_gamma
        self._material_ground_horizontal_dampings[material.idx] = ground_damping

        if particle_gamma > 0.0 or surface_gamma > 0.0 or ground_damping > 0.0:
            self._has_surface_tension = True

    def build(self):
        super().build()
        if self.is_active and self._has_surface_tension:
            shape = (self._sim.substeps_local, *self._grid_res, self._B)
            self.surface_color = qd.field(dtype=gs.qd_float, shape=shape)
            self.surface_gamma_num = qd.field(dtype=gs.qd_float, shape=shape)
            self.surface_gamma_den = qd.field(dtype=gs.qd_float, shape=shape)
            self.surface_ground_damping_num = qd.field(dtype=gs.qd_float, shape=shape)
            self.surface_ground_damping_den = qd.field(dtype=gs.qd_float, shape=shape)
            self.surface_grad_c = qd.Vector.field(3, dtype=gs.qd_float, shape=shape)
            self.surface_normal = qd.Vector.field(3, dtype=gs.qd_float, shape=shape)
            self.surface_curvature = qd.field(dtype=gs.qd_float, shape=shape)

    def substep_pre_coupling(self, f):
        # Inline the base MPM pre-coupling flow so particle-level cohesion can be applied before p2g uses particle vel.
        if self._sim.requires_grad:
            self.reset_grid_and_grad(f)
            self.compute_F_tmp(f)
            self.svd(f)
        else:
            if self.needs_svd:
                self.compute_F_tmp_and_svd(f)
            else:
                self.compute_F_tmp_only(f)

        if self._has_surface_tension and not self._sim.requires_grad:
            self.apply_particle_cohesion(f)

        self.p2g(
            f,
            self.sim.coupler.rigid_solver.geoms_state,
            self.sim.coupler.rigid_solver.geoms_info,
            self.sim.coupler.rigid_solver.links_state,
            self.sim.coupler.rigid_solver._rigid_global_info,
            self.sim.coupler.rigid_solver.collider._sdf._sdf_info,
            self.sim.coupler.rigid_solver.collider._collider_static_config,
        )

        if self._has_surface_tension and not self._sim.requires_grad:
            self.reset_surface_fields(f)
            self.project_surface_color(f)
            self.compute_surface_normal(f)
            self.compute_surface_curvature(f)
            self.apply_surface_tension_to_grid(f)
            self.apply_ground_horizontal_damping(f)

    @qd.kernel
    def reset_surface_fields(self, f: qd.i32):
        for i, j, k, i_b in qd.ndrange(*self._grid_res, self._B):
            self.surface_color[f, i, j, k, i_b] = gs.qd_float(0.0)
            self.surface_gamma_num[f, i, j, k, i_b] = gs.qd_float(0.0)
            self.surface_gamma_den[f, i, j, k, i_b] = gs.qd_float(0.0)
            self.surface_ground_damping_num[f, i, j, k, i_b] = gs.qd_float(0.0)
            self.surface_ground_damping_den[f, i, j, k, i_b] = gs.qd_float(0.0)
            self.surface_grad_c[f, i, j, k, i_b] = qd.Vector.zero(gs.qd_float, 3)
            self.surface_normal[f, i, j, k, i_b] = qd.Vector.zero(gs.qd_float, 3)
            self.surface_curvature[f, i, j, k, i_b] = gs.qd_float(0.0)

    @qd.func
    def _func_particle_cohesion_gamma(self, i_p):
        gamma = gs.qd_float(0.0)
        for mat_idx in qd.static(self._materials_idx):
            if self.particles_info[i_p].material_idx == mat_idx:
                gamma = gs.qd_float(self._material_particle_cohesion_gammas[mat_idx])
        return gamma

    @qd.func
    def _func_surface_tension_gamma(self, i_p):
        gamma = gs.qd_float(0.0)
        for mat_idx in qd.static(self._materials_idx):
            if self.particles_info[i_p].material_idx == mat_idx:
                gamma = gs.qd_float(self._material_surface_tension_gammas[mat_idx])
        return gamma

    @qd.func
    def _func_ground_horizontal_damping(self, i_p):
        damping = gs.qd_float(0.0)
        for mat_idx in qd.static(self._materials_idx):
            if self.particles_info[i_p].material_idx == mat_idx:
                damping = gs.qd_float(self._material_ground_horizontal_dampings[mat_idx])
        return damping

    @qd.kernel
    def apply_particle_cohesion(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[f, i_p, i_b].active:
                gamma_i = self._func_particle_cohesion_gamma(i_p)
                if gamma_i > gs.qd_float(0.0):
                    xi = self.particles[f, i_p, i_b].pos
                    acc = qd.Vector.zero(gs.qd_float, 3)
                    support_radius = gs.qd_float(self._particle_cohesion_radius_factor * self._particle_size)

                    # O(N^2) fallback: acceptable for small droplets, but should later be replaced by spatial hashing.
                    for j_p in range(self._n_particles):
                        if j_p != i_p and self.particles_ng[f, j_p, i_b].active:
                            xj = self.particles[f, j_p, i_b].pos
                            d_ij = xi - xj
                            dist = d_ij.norm(gs.EPS)

                            if dist > gs.qd_float(1e-6) and dist < support_radius:
                                q = dist / support_radius
                                # Smooth compact kernel: strong at close range, zero at support radius.
                                w = (gs.qd_float(1.0) - q) * (gs.qd_float(1.0) - q)
                                direction_to_neighbor = -d_ij / dist
                                acc += gamma_i * w * direction_to_neighbor

                    acc_norm = acc.norm(gs.EPS)
                    max_acc = gs.qd_float(self._particle_cohesion_max_acc)
                    if acc_norm > max_acc:
                        acc = acc / acc_norm * max_acc

                    self.particles[f, i_p, i_b].vel += self.substep_dt * acc

    @qd.kernel
    def project_surface_color(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[f, i_p, i_b].active:
                surface_gamma = self._func_surface_tension_gamma(i_p)
                ground_damping = self._func_ground_horizontal_damping(i_p)

                if surface_gamma > gs.qd_float(0.0) or ground_damping > gs.qd_float(0.0):
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
                        qd.atomic_add(self.surface_gamma_num[f, cell_ijk, i_b], color_contrib * surface_gamma)
                        qd.atomic_add(self.surface_gamma_den[f, cell_ijk, i_b], color_contrib)
                        qd.atomic_add(self.surface_ground_damping_num[f, cell_ijk, i_b], color_contrib * ground_damping)
                        qd.atomic_add(self.surface_ground_damping_den[f, cell_ijk, i_b], color_contrib)

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
                grad_c = qd.Vector.zero(gs.qd_float, 3)
                weight_sum = gs.qd_float(0.0)

                # 26-neighbor isotropic gradient stencil. Compared with the old 6-neighbor central difference, this
                # reduces axis-aligned square artifacts by including face, edge and corner directions.
                for offset_raw in qd.static(qd.grouped(qd.ndrange(3, 3, 3))):
                    sx = offset_raw[0] - 1
                    sy = offset_raw[1] - 1
                    sz = offset_raw[2] - 1
                    if sx != 0 or sy != 0 or sz != 0:
                        direction = qd.Vector([sx, sy, sz], dt=gs.qd_float)
                        dist2 = direction.dot(direction)
                        weight = gs.qd_float(1.0) / dist2
                        c_nb = self.surface_color[f, ii + sx, jj + sy, kk + sz, i_b]
                        grad_c += c_nb * direction * weight
                        weight_sum += weight

                if weight_sum > gs.qd_float(0.0):
                    grad_c = grad_c * self._inv_dx / weight_sum

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
                n_center = self.surface_normal[f, ii, jj, kk, i_b]
                div_n = gs.qd_float(0.0)
                weight_sum = gs.qd_float(0.0)

                # 26-neighbor divergence of normal. This is not a full geometric curvature estimator, but it is more
                # isotropic than the old axis-only stencil and works as a compact numerical cohesion control.
                for offset_raw in qd.static(qd.grouped(qd.ndrange(3, 3, 3))):
                    sx = offset_raw[0] - 1
                    sy = offset_raw[1] - 1
                    sz = offset_raw[2] - 1
                    if sx != 0 or sy != 0 or sz != 0:
                        direction = qd.Vector([sx, sy, sz], dt=gs.qd_float)
                        dist2 = direction.dot(direction)
                        weight = gs.qd_float(1.0) / dist2
                        n_nb = self.surface_normal[f, ii + sx, jj + sy, kk + sz, i_b]
                        div_n += (n_nb - n_center).dot(direction) * weight
                        weight_sum += weight

                if weight_sum > gs.qd_float(0.0):
                    div_n = div_n * self._inv_dx / weight_sum

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

                    # SPH-like numerical surface tension. Here gamma is an acceleration-scale control, not a physical
                    # N/m coefficient. Use particle_cohesion_gamma separately for particle-particle clustering.
                    a_st = gamma * curvature * normal

                    a_norm = a_st.norm(gs.EPS)
                    max_a = gs.qd_float(self._surface_tension_max_acc)
                    if a_norm > max_a:
                        a_st = a_st / a_norm * max_a

                    self.grid[f, I, i_b].vel_in += self.grid[f, I, i_b].mass * self.substep_dt * a_st

    @qd.kernel
    def apply_ground_horizontal_damping(self, f: qd.i32):
        for ii, jj, kk, i_b in qd.ndrange(*self._grid_res, self._B):
            I = (ii, jj, kk)
            if self.grid[f, I, i_b].mass > gs.EPS:
                z = (kk + self._grid_offset[2]) * self._dx
                ground_band = gs.qd_float(self._ground_damping_height_factor * self._particle_size)
                if z > gs.qd_float(self._ground_z) and z < gs.qd_float(self._ground_z) + ground_band:
                    damping = gs.qd_float(self._ground_horizontal_damping)
                    if self.surface_ground_damping_den[f, I, i_b] > gs.EPS:
                        material_damping = self.surface_ground_damping_num[f, I, i_b] / self.surface_ground_damping_den[f, I, i_b]
                        if material_damping > gs.qd_float(0.0):
                            damping = material_damping
                    self.grid[f, I, i_b].vel_in[0] *= damping
                    self.grid[f, I, i_b].vel_in[1] *= damping
