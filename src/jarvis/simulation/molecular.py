"""
JARVIS-PRIME Molecular Dynamics Simulator
============================================

Simulates:
- Lennard-Jones interatomic potentials
- Verlet integration for particle dynamics
- Energy minimization (steepest descent)
- Radial distribution function
- Thermodynamic properties (T, P, E)

Pure NumPy implementation — no OpenMM/GROMACS required.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class LennardJones:
    """Lennard-Jones 12-6 potential for noble gas simulation."""

    def __init__(
        self,
        epsilon: float = 1.0,   # Energy scale (reduced units)
        sigma: float = 1.0,     # Length scale (reduced units)
        cutoff: float = 2.5,    # Cutoff in sigma units
    ):
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff * sigma

    def potential(self, r: float) -> float:
        """V(r) = 4*epsilon * [(sigma/r)^12 - (sigma/r)^6]"""
        if r < 0.01 * self.sigma:
            return 1e10
        if r > self.cutoff:
            return 0.0
        sr6 = (self.sigma / r) ** 6
        return 4 * self.epsilon * (sr6**2 - sr6)

    def force(self, r: float) -> float:
        """F(r) = -dV/dr = 24*epsilon/r * [2*(sigma/r)^12 - (sigma/r)^6]"""
        if r < 0.01 * self.sigma:
            return 1e10
        if r > self.cutoff:
            return 0.0
        sr6 = (self.sigma / r) ** 6
        return 24 * self.epsilon / r * (2 * sr6**2 - sr6)

    def potential_curve(self, r_min: float = 0.8, r_max: float = 3.0, points: int = 100) -> dict[str, Any]:
        """Generate the LJ potential energy curve."""
        r_values = np.linspace(r_min * self.sigma, r_max * self.sigma, points)
        v_values = [self.potential(r) for r in r_values]

        # Find minimum
        min_idx = np.argmin(v_values)

        return {
            "r_min_sigma": round(r_values[min_idx] / self.sigma, 4),
            "V_min_epsilon": round(v_values[min_idx] / self.epsilon, 4),
            "equilibrium_distance": round(r_values[min_idx], 4),
            "well_depth": round(abs(min(v_values)), 4),
            "points": points,
        }


class MolecularDynamics:
    """
    Simple molecular dynamics engine.

    Uses velocity-Verlet integration with Lennard-Jones potential.
    Operates in reduced units (epsilon, sigma, mass = 1).
    """

    def __init__(
        self,
        n_particles: int = 50,
        box_size: float = 10.0,
        temperature: float = 1.0,
        dt: float = 0.005,
    ):
        self.n = min(n_particles, 200)  # Cap for performance
        self.box = box_size
        self.T_target = temperature
        self.dt = dt
        self.lj = LennardJones()

        # Initialize positions on a grid
        n_side = int(np.ceil(self.n ** (1/3)))
        spacing = box_size / n_side
        positions = []
        for ix in range(n_side):
            for iy in range(n_side):
                for iz in range(n_side):
                    if len(positions) < self.n:
                        positions.append([
                            (ix + 0.5) * spacing,
                            (iy + 0.5) * spacing,
                            (iz + 0.5) * spacing,
                        ])
        self.pos = np.array(positions[:self.n])

        # Initialize velocities (Maxwell-Boltzmann)
        self.vel = np.random.randn(self.n, 3) * np.sqrt(temperature)
        # Remove center-of-mass velocity
        self.vel -= self.vel.mean(axis=0)

        self.forces = np.zeros_like(self.pos)
        self._compute_forces()

    def _compute_forces(self) -> float:
        """Compute pairwise LJ forces. Returns total potential energy."""
        self.forces.fill(0)
        pe = 0.0

        for i in range(self.n):
            for j in range(i + 1, self.n):
                dr = self.pos[j] - self.pos[i]
                # Minimum image convention (periodic BC)
                dr -= self.box * np.round(dr / self.box)
                r = np.linalg.norm(dr)

                if r < self.lj.cutoff and r > 0.01:
                    f_mag = self.lj.force(r)
                    f_vec = f_mag * dr / r
                    self.forces[i] += f_vec
                    self.forces[j] -= f_vec
                    pe += self.lj.potential(r)

        return pe

    def step(self) -> dict[str, float]:
        """Perform one velocity-Verlet integration step."""
        # Half-step velocity
        self.vel += 0.5 * self.forces * self.dt

        # Full-step position
        self.pos += self.vel * self.dt

        # Periodic boundary conditions
        self.pos %= self.box

        # Recompute forces
        pe = self._compute_forces()

        # Half-step velocity
        self.vel += 0.5 * self.forces * self.dt

        # Kinetic energy
        ke = 0.5 * np.sum(self.vel**2)

        # Temperature
        temperature = 2 * ke / (3 * self.n)

        return {
            "potential_energy": pe,
            "kinetic_energy": ke,
            "total_energy": pe + ke,
            "temperature": temperature,
        }

    def run(self, steps: int = 500) -> dict[str, Any]:
        """Run MD simulation for given number of steps."""
        energies = []
        temperatures = []

        for i in range(steps):
            result = self.step()
            if i % 10 == 0:
                energies.append(result["total_energy"])
                temperatures.append(result["temperature"])

        # Statistics
        energies = np.array(energies)
        temperatures = np.array(temperatures)

        return {
            "task": "molecular_dynamics",
            "n_particles": self.n,
            "box_size": self.box,
            "steps": steps,
            "dt": self.dt,
            "results": {
                "avg_temperature": round(float(temperatures.mean()), 4),
                "std_temperature": round(float(temperatures.std()), 4),
                "avg_total_energy": round(float(energies.mean()), 4),
                "energy_drift": round(float(energies[-1] - energies[0]), 6),
                "energy_conservation_pct": round(
                    float(100 * (1 - abs(energies[-1] - energies[0]) / max(abs(energies[0]), 1e-10))), 2
                ),
            },
            "thermodynamics": {
                "target_temperature": self.T_target,
                "measured_temperature": round(float(temperatures[-1]), 4),
                "final_potential_energy": round(float(energies[-1] - 0.5 * np.sum(self.vel**2)), 4),
                "final_kinetic_energy": round(float(0.5 * np.sum(self.vel**2)), 4),
            },
        }


class EnergyMinimizer:
    """Steepest descent energy minimization."""

    @staticmethod
    def minimize_2d(
        func_name: str = "rosenbrock",
        x0: float = -1.5,
        y0: float = 1.5,
        step_size: float = 0.001,
        max_steps: int = 10000,
        tolerance: float = 1e-8,
    ) -> dict[str, Any]:
        """
        Minimize a 2D function using steepest descent.

        Available functions: rosenbrock, rastrigin, sphere
        """
        if func_name == "rosenbrock":
            def f(x, y): return (1 - x)**2 + 100 * (y - x**2)**2
            def grad(x, y): return np.array([
                -2 * (1 - x) - 400 * x * (y - x**2),
                200 * (y - x**2),
            ])
            known_min = (1.0, 1.0)
        elif func_name == "rastrigin":
            def f(x, y): return 20 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))
            def grad(x, y): return np.array([
                2*x + 20*np.pi*np.sin(2*np.pi*x),
                2*y + 20*np.pi*np.sin(2*np.pi*y),
            ])
            known_min = (0.0, 0.0)
        else:  # sphere
            def f(x, y): return x**2 + y**2
            def grad(x, y): return np.array([2*x, 2*y])
            known_min = (0.0, 0.0)

        pos = np.array([x0, y0])
        trajectory = [(float(pos[0]), float(pos[1]), float(f(pos[0], pos[1])))]

        for i in range(max_steps):
            g = grad(pos[0], pos[1])
            g_norm = np.linalg.norm(g)

            if g_norm < tolerance:
                break

            pos -= step_size * g

            if i % 100 == 0:
                trajectory.append((float(pos[0]), float(pos[1]), float(f(pos[0], pos[1]))))

        return {
            "function": func_name,
            "initial_position": [x0, y0],
            "final_position": [round(float(pos[0]), 6), round(float(pos[1]), 6)],
            "final_value": round(float(f(pos[0], pos[1])), 8),
            "known_minimum": list(known_min),
            "iterations": i + 1,
            "converged": bool(np.linalg.norm(grad(pos[0], pos[1])) < tolerance),
            "distance_to_minimum": round(float(np.linalg.norm(pos - np.array(known_min))), 6),
        }
