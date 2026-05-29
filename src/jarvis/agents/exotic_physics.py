"""
JARVIS-PRIME Exotic Physics Agent
===================================

Domain specialist for gravitational engineering, exotic propulsion,
Casimir effect engineering, and fundamental physics simulation.

Capabilities:
- Lense-Thirring frame-dragging computation
- Alcubierre warp metric energy analysis
- Casimir force engineering (parallel plates + metamaterial enhancement)
- Exotic propulsion concept survey and falsifiable prediction generation
- Research journal with hypothesis tracking

Mathematical Framework:
    Lense-Thirring:  Ω_LT = GJ / (c² r³)
    Alcubierre:      ds² = -dt² + (dx - v_s f(r_s) dt)² + dy² + dz²
    Casimir:         F/A = -π² ℏc / (240 d⁴)
    Gravitomagnetic: B_g = -G/(c²) [3r(J·r)/r⁵ - J/r³]
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jarvis.agents.base_agent import BaseAgent


# ══════════════════════════════════════════════════════════
# Physical Constants
# ══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PhysicsConstants:
    """Fundamental physical constants (SI units, CODATA 2018)."""
    G: float = 6.67430e-11          # Gravitational constant (m³ kg⁻¹ s⁻²)
    c: float = 2.99792458e8         # Speed of light (m/s)
    hbar: float = 1.054571817e-34   # Reduced Planck constant (J·s)
    h: float = 6.62607015e-34       # Planck constant (J·s)
    epsilon_0: float = 8.854187817e-12   # Vacuum permittivity (F/m)
    mu_0: float = 1.2566370621e-6   # Vacuum permeability (H/m)
    k_B: float = 1.380649e-23       # Boltzmann constant (J/K)
    e: float = 1.602176634e-19      # Elementary charge (C)
    m_e: float = 9.1093837015e-31   # Electron mass (kg)
    M_sun: float = 1.989e30         # Solar mass (kg)
    M_earth: float = 5.972e24       # Earth mass (kg)
    R_earth: float = 6.371e6        # Earth radius (m)
    J_earth: float = 7.07e33        # Earth angular momentum (kg·m²/s)


CONST = PhysicsConstants()


# ══════════════════════════════════════════════════════════
# Lense-Thirring Frame-Dragging Calculator
# ══════════════════════════════════════════════════════════

class LenseThirringCalculator:
    """
    Gravitomagnetic frame-dragging effects.

    The Lense-Thirring precession rate for a test gyroscope at
    distance r from a rotating mass with angular momentum J:

        Ω_LT = G · J / (c² · r³)

    Gravity Probe B measured the Lense-Thirring precession of
    Earth as ~39 milliarcseconds/year, confirming GR.
    """

    def precession_rate(
        self,
        angular_momentum: float,
        distance: float,
    ) -> float:
        """
        Lense-Thirring precession rate (rad/s).

        Args:
            angular_momentum: J in kg·m²/s
            distance: r in meters from center of mass

        Returns:
            Precession rate in rad/s

        Example (Earth at surface):
            >>> lt = LenseThirringCalculator()
            >>> omega = lt.precession_rate(CONST.J_earth, CONST.R_earth)
            >>> # Should be ~7.2e-14 rad/s ≈ 39 mas/yr
        """
        return CONST.G * angular_momentum / (CONST.c**2 * distance**3)

    def gravitomagnetic_field(
        self,
        J_vec: np.ndarray,
        r_vec: np.ndarray,
    ) -> np.ndarray:
        """
        Gravitomagnetic field vector B_g at position r.

        B_g = -(G/c²) · [3r̂(J·r̂)/r³ - J/r³]

        where r̂ = r/|r|

        Args:
            J_vec: Angular momentum vector (3D)
            r_vec: Position vector (3D)

        Returns:
            Gravitomagnetic field vector (3D)
        """
        r_mag = np.linalg.norm(r_vec)
        if r_mag < 1e-30:
            return np.zeros(3)

        J_dot_r = np.dot(J_vec, r_vec)
        prefactor = -CONST.G / CONST.c**2

        term1 = 3 * r_vec * J_dot_r / r_mag**5
        term2 = J_vec / r_mag**3

        return prefactor * (term1 - term2)

    def frame_dragging_velocity(
        self,
        angular_momentum: float,
        distance: float,
        theta: float = np.pi / 2,
    ) -> float:
        """
        Frame-dragging velocity in the equatorial plane.

        v_fd = 2GJ sin(θ) / (c² r²)
        """
        return (
            2 * CONST.G * angular_momentum * np.sin(theta)
            / (CONST.c**2 * distance**2)
        )

    def compute_field_map(
        self,
        angular_momentum: float,
        extent: float,
        grid_size: int = 50,
    ) -> dict[str, Any]:
        """
        Compute gravitomagnetic field over a 2D grid.

        Returns grid coordinates and field magnitude for visualization.
        """
        J_vec = np.array([0.0, 0.0, angular_momentum])
        coords = np.linspace(-extent, extent, grid_size)
        X, Y = np.meshgrid(coords, coords)
        B_mag = np.zeros_like(X)

        for i in range(grid_size):
            for j in range(grid_size):
                r_vec = np.array([X[i, j], Y[i, j], 0.0])
                r_norm = np.linalg.norm(r_vec)
                if r_norm > extent * 0.02:  # Avoid singularity
                    B = self.gravitomagnetic_field(J_vec, r_vec)
                    B_mag[i, j] = np.linalg.norm(B)

        return {
            "x": coords.tolist(),
            "y": coords.tolist(),
            "B_magnitude": B_mag.tolist(),
            "units": "m/s² (gravitomagnetic)",
        }

    def validate_against_gravity_probe_b(self) -> dict[str, Any]:
        """
        Validate our computation against Gravity Probe B data.

        GP-B measured: Ω_LT = 37.2 ± 7.2 mas/yr for Earth
        GR prediction: Ω_LT = 39.2 mas/yr
        """
        omega = self.precession_rate(CONST.J_earth, CONST.R_earth + 642e3)
        # Convert rad/s to milliarcseconds/year
        mas_per_year = omega * (180 / np.pi) * 3600 * 1000 * (365.25 * 24 * 3600)

        return {
            "computed_precession_rad_s": omega,
            "computed_precession_mas_yr": mas_per_year,
            "gpb_measured_mas_yr": 37.2,
            "gpb_uncertainty_mas_yr": 7.2,
            "gr_prediction_mas_yr": 39.2,
            "within_uncertainty": abs(mas_per_year - 37.2) < 7.2,
            "validation_status": "PASS" if abs(mas_per_year - 39.2) / 39.2 < 0.1 else "FAIL",
        }


# ══════════════════════════════════════════════════════════
# Alcubierre Warp Metric
# ══════════════════════════════════════════════════════════

class AlcubierreMetric:
    """
    Alcubierre warp drive metric computations.

    The Alcubierre metric (1994):
        ds² = -dt² + (dx - v_s(t)·f(r_s)·dt)² + dy² + dz²

    Shape function (top-hat):
        f(r_s) = [tanh(σ(r_s + R)) - tanh(σ(r_s - R))] / [2·tanh(σR)]

    Exotic matter energy density (York time derivative):
        ρ = -(v_s²/32πG) · (df/dr_s)² · (y² + z²) / r_s²

    Reference: M. Alcubierre, "The warp drive: hyper-fast travel
    within general relativity", Class. Quantum Grav. 11 L73 (1994)
    """

    def shape_function(
        self,
        r_s: float | np.ndarray,
        R: float = 100.0,
        sigma: float = 1.0,
    ) -> float | np.ndarray:
        """
        Top-hat shape function for the warp bubble.

        f(r_s) → 1 inside bubble (r_s < R)
        f(r_s) → 0 outside bubble (r_s > R)
        Transition width controlled by σ.
        """
        num = np.tanh(sigma * (r_s + R)) - np.tanh(sigma * (r_s - R))
        den = 2 * np.tanh(sigma * R)
        return num / den

    def shape_derivative(
        self,
        r_s: float | np.ndarray,
        R: float = 100.0,
        sigma: float = 1.0,
    ) -> float | np.ndarray:
        """Derivative df/dr_s of the shape function."""
        t1 = sigma * (1 - np.tanh(sigma * (r_s + R))**2)
        t2 = sigma * (1 - np.tanh(sigma * (r_s - R))**2)
        den = 2 * np.tanh(sigma * R)
        return (t1 - t2) / den

    def energy_density(
        self,
        v_s: float,
        r_s: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        R: float = 100.0,
        sigma: float = 1.0,
    ) -> np.ndarray:
        """
        Exotic matter energy density ρ (kg/m³ equivalent).

        ρ = -(v_s² / 32πG) · (df/dr_s)² · (y² + z²) / r_s²
        """
        df = self.shape_derivative(r_s, R, sigma)
        rho_perp_sq = y**2 + z**2
        r_s_safe = np.where(np.abs(r_s) < 1e-30, 1e-30, r_s)

        return -(v_s**2 / (32 * np.pi * CONST.G)) * df**2 * rho_perp_sq / r_s_safe**2

    def total_energy(
        self,
        velocity_c: float = 0.1,
        R: float = 100.0,
        sigma: float = 1.0,
        grid_size: int = 100,
    ) -> dict[str, Any]:
        """
        Numerically integrate total exotic energy requirement.

        E = ∫ ρ · c² dV

        Args:
            velocity_c: Ship velocity as fraction of c
            R: Bubble radius in meters
            sigma: Wall thickness parameter (larger = thinner wall)
            grid_size: Resolution of numerical integration

        Returns:
            Energy budget analysis
        """
        v_s = velocity_c * CONST.c
        extent = 3 * R
        lin = np.linspace(-extent, extent, grid_size)
        X, Y, Z = np.meshgrid(lin, lin, lin, indexing='ij')

        r_s = np.sqrt(X**2 + Y**2 + Z**2)
        rho = self.energy_density(v_s, r_s, Y, Z, R, sigma)

        dV = (2 * extent / grid_size)**3
        total_E = float(np.sum(rho) * dV * CONST.c**2)

        # Reference energies
        E_sun = CONST.M_sun * CONST.c**2
        E_jupiter = 1.898e27 * CONST.c**2

        return {
            "velocity_fraction_c": velocity_c,
            "velocity_m_s": v_s,
            "bubble_radius_m": R,
            "wall_thickness_sigma": sigma,
            "total_exotic_energy_J": total_E,
            "ratio_to_sun": abs(total_E) / E_sun,
            "ratio_to_jupiter": abs(total_E) / E_jupiter,
            "solar_masses_equivalent": abs(total_E) / E_sun,
            "feasibility": self._assess_feasibility(total_E),
        }

    def parameter_sweep(
        self,
        velocities: list[float] | None = None,
        radii: list[float] | None = None,
        sigmas: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Sweep over parameter space to find energy-minimizing configurations.
        """
        if velocities is None:
            velocities = [0.01, 0.05, 0.1, 0.5]
        if radii is None:
            radii = [10.0, 50.0, 100.0, 500.0]
        if sigmas is None:
            sigmas = [0.5, 1.0, 5.0, 10.0]

        results = []
        for v in velocities:
            for R in radii:
                for s in sigmas:
                    result = self.total_energy(v, R, s, grid_size=50)
                    results.append(result)

        return sorted(results, key=lambda x: abs(x["total_exotic_energy_J"]))

    def _assess_feasibility(self, energy_J: float) -> str:
        abs_E = abs(energy_J)
        E_sun = CONST.M_sun * CONST.c**2

        if abs_E > E_sun:
            return "INFEASIBLE: Exceeds solar rest mass energy. Requires fundamental physics breakthrough."
        elif abs_E > 1e40:
            return "EXTREME: Exceeds any conceivable engineering energy source. Speculative research only."
        elif abs_E > 1e20:
            return "CHALLENGING: Within planetary energy scale but requires novel energy generation."
        else:
            return "POTENTIALLY ACCESSIBLE: Energy requirement within extreme engineering limits."


# ══════════════════════════════════════════════════════════
# Casimir Force Engineering
# ══════════════════════════════════════════════════════════

class CasimirForceCalculator:
    """
    Engineering-scale Casimir force computations.

    Parallel plate force per unit area:
        F/A = -π² ℏc / (240 d⁴)

    Casimir energy per unit area:
        E/A = -π² ℏc / (720 d³)

    Current experimental achievements:
    - Metamaterials can amplify Casimir forces 10-100×
    - Casimir, Inc. (2026) claims MicroSparc chips: 1.5V @ 25μA
    - QET demonstrated: energy transfer in <40ms
    """

    def pressure(self, separation_m: float) -> float:
        """
        Casimir pressure between ideal parallel conducting plates.

        F/A = -π²ℏc / (240 d⁴)

        Args:
            separation_m: Plate separation in meters

        Returns:
            Pressure in Pascals (negative = attractive)
        """
        return -(np.pi**2 * CONST.hbar * CONST.c) / (240 * separation_m**4)

    def force(self, separation_m: float, area_m2: float) -> float:
        """Total Casimir force for finite plates (N)."""
        return self.pressure(separation_m) * area_m2

    def energy_density(self, separation_m: float) -> float:
        """Casimir energy per unit area (J/m²)."""
        return -(np.pi**2 * CONST.hbar * CONST.c) / (720 * separation_m**3)

    def metamaterial_enhanced(
        self,
        separation_m: float,
        area_m2: float,
        enhancement: float = 50.0,
    ) -> dict[str, float]:
        """
        Casimir force with metamaterial enhancement.

        Current experimental range: η = 10-100×
        """
        base_f = self.force(separation_m, area_m2)
        enhanced_f = base_f * enhancement

        return {
            "base_force_N": base_f,
            "enhanced_force_N": enhanced_f,
            "enhancement_factor": enhancement,
            "base_pressure_Pa": self.pressure(separation_m),
            "enhanced_pressure_Pa": self.pressure(separation_m) * enhancement,
        }

    def separation_sweep(
        self,
        min_sep: float = 1e-9,
        max_sep: float = 1e-5,
        n_points: int = 200,
    ) -> dict[str, list[float]]:
        """
        Sweep force over separation range for design optimization.
        """
        separations = np.logspace(np.log10(min_sep), np.log10(max_sep), n_points)
        pressures = [self.pressure(s) for s in separations]
        energies = [self.energy_density(s) for s in separations]

        return {
            "separations_m": separations.tolist(),
            "pressures_Pa": pressures,
            "energy_densities_J_m2": energies,
        }

    def cavity_power_estimate(
        self,
        separation_m: float,
        area_m2: float,
        oscillation_freq_Hz: float,
        enhancement: float = 50.0,
    ) -> dict[str, float]:
        """
        ORDER-OF-MAGNITUDE estimate of power from oscillating Casimir cavity.

        P ≈ |F_enhanced| × displacement × frequency

        WARNING: This is a rough theoretical estimate, NOT a prediction
        of actual extractable power. Thermodynamic constraints apply.
        """
        force = abs(self.force(separation_m, area_m2) * enhancement)
        displacement = separation_m * 0.1  # 10% of gap oscillation
        power_W = force * displacement * oscillation_freq_Hz

        return {
            "estimated_power_W": power_W,
            "force_N": force,
            "displacement_m": displacement,
            "frequency_Hz": oscillation_freq_Hz,
            "caveat": (
                "ORDER OF MAGNITUDE ESTIMATE ONLY. Actual extractable power "
                "subject to thermodynamic constraints, cavity Q-factor, "
                "and parasitic losses. Must not violate 2nd law."
            ),
        }

    def compare_to_casimir_inc(self) -> dict[str, Any]:
        """
        Compare our theoretical calculations to Casimir, Inc. claims.

        Casimir, Inc. (May 2026): MicroSparc chips
        - Claimed output: 1.5V @ 25μA = 37.5 μW
        - Technology: Engineered Casimir cavities in semiconductor
        """
        claimed_power = 1.5 * 25e-6  # 37.5 μW

        # Estimate what cavity parameters would be needed
        # Assume 1 cm² chip area, 100nm gap, 1 GHz oscillation
        our_estimate = self.cavity_power_estimate(
            separation_m=100e-9,
            area_m2=1e-4,
            oscillation_freq_Hz=1e9,
            enhancement=100.0,
        )

        return {
            "casimir_inc_claimed_power_W": claimed_power,
            "casimir_inc_claimed_voltage_V": 1.5,
            "casimir_inc_claimed_current_A": 25e-6,
            "our_theoretical_estimate_W": our_estimate["estimated_power_W"],
            "ratio": our_estimate["estimated_power_W"] / claimed_power if claimed_power > 0 else 0,
            "assessment": (
                "Independent verification required. Claims are extraordinary "
                "and demand extraordinary evidence. Our theoretical estimate "
                "provides an order-of-magnitude cross-check."
            ),
        }


# ══════════════════════════════════════════════════════════
# Research Journal
# ══════════════════════════════════════════════════════════

class ResearchJournal:
    """
    Falsification-first research journal for exotic physics.

    Every hypothesis MUST include:
    1. A clear mathematical prediction
    2. A falsifiable experimental test
    3. The sensitivity required to detect/refute

    IRON RULE: Never dismiss any hypothesis without first deriving
    its mathematical predictions and identifying a falsifiable test.
    """

    def __init__(self, journal_dir: Path | None = None):
        self.journal_dir = journal_dir
        self._hypotheses: list[dict[str, Any]] = []
        self._experiments: list[dict[str, Any]] = []

        if journal_dir:
            journal_dir.mkdir(parents=True, exist_ok=True)

    def log_hypothesis(
        self,
        hypothesis: str,
        mathematical_prediction: str,
        falsification_test: str,
        required_sensitivity: str = "",
        domain: str = "exotic_physics",
        references: list[str] | None = None,
    ) -> str:
        """
        Log a research hypothesis with falsifiable prediction.

        Returns the hypothesis ID.
        """
        hyp_id = f"HYP-{len(self._hypotheses):04d}"
        entry = {
            "id": hyp_id,
            "hypothesis": hypothesis,
            "mathematical_prediction": mathematical_prediction,
            "falsification_test": falsification_test,
            "required_sensitivity": required_sensitivity,
            "domain": domain,
            "references": references or [],
            "status": "proposed",
            "timestamp": time.time(),
            "results": None,
            "conclusion": None,
        }
        self._hypotheses.append(entry)
        self._persist()
        return hyp_id

    def log_experiment(
        self,
        hypothesis_id: str,
        methodology: str,
        results: dict[str, Any],
        conclusion: str,
        p_value: float | None = None,
    ) -> str:
        """Log an experimental result."""
        exp_id = f"EXP-{len(self._experiments):04d}"
        entry = {
            "id": exp_id,
            "hypothesis_id": hypothesis_id,
            "methodology": methodology,
            "results": results,
            "conclusion": conclusion,
            "p_value": p_value,
            "timestamp": time.time(),
        }
        self._experiments.append(entry)

        # Update hypothesis status
        for h in self._hypotheses:
            if h["id"] == hypothesis_id:
                h["results"] = results
                h["conclusion"] = conclusion
                if p_value and p_value < 0.05:
                    h["status"] = "supported"
                elif p_value and p_value >= 0.05:
                    h["status"] = "not_supported"
                else:
                    h["status"] = "inconclusive"
                break

        self._persist()
        return exp_id

    def get_open_hypotheses(self) -> list[dict[str, Any]]:
        """Get all hypotheses that haven't been tested yet."""
        return [h for h in self._hypotheses if h["status"] == "proposed"]

    def get_all(self) -> dict[str, Any]:
        """Get complete journal contents."""
        return {
            "hypotheses": self._hypotheses,
            "experiments": self._experiments,
            "stats": {
                "total_hypotheses": len(self._hypotheses),
                "open": sum(1 for h in self._hypotheses if h["status"] == "proposed"),
                "supported": sum(1 for h in self._hypotheses if h["status"] == "supported"),
                "refuted": sum(1 for h in self._hypotheses if h["status"] == "not_supported"),
            },
        }

    def _persist(self) -> None:
        if self.journal_dir:
            (self.journal_dir / "hypotheses.json").write_text(
                json.dumps(self._hypotheses, indent=2, default=str)
            )
            (self.journal_dir / "experiments.json").write_text(
                json.dumps(self._experiments, indent=2, default=str)
            )


# ══════════════════════════════════════════════════════════
# Exotic Physics Agent
# ══════════════════════════════════════════════════════════

class ExoticPhysicsAgent(BaseAgent):
    """
    Domain agent for exotic physics research and simulation.

    Manages GR solvers, Casimir engineering, and propulsion modeling.
    All results are logged to the research journal with falsifiable predictions.
    """

    SUPPORTED_TASKS = [
        "lense_thirring",
        "alcubierre_energy",
        "alcubierre_sweep",
        "casimir_analysis",
        "casimir_sweep",
        "casimir_inc_comparison",
        "propulsion_survey",
        "gravity_probe_b_validation",
        "full_analysis",
    ]

    def __init__(self, journal_dir: Path | None = None):
        super().__init__(name="ExoticPhysicsAgent", domain="exotic_physics")
        self.lense_thirring = LenseThirringCalculator()
        self.alcubierre = AlcubierreMetric()
        self.casimir = CasimirForceCalculator()
        self.journal = ResearchJournal(journal_dir)

    def get_capabilities(self) -> list[str]:
        return [
            "lense_thirring_computation",
            "alcubierre_metric_analysis",
            "casimir_force_engineering",
            "exotic_propulsion_survey",
            "gravitomagnetic_field_mapping",
            "research_hypothesis_tracking",
            "experimental_falsification_design",
        ]

    async def execute(self, task: dict[str, Any], tools: list[Any]) -> dict[str, Any]:
        """Execute an exotic physics task."""
        task_type = task.get("type", "full_analysis")

        handlers = {
            "lense_thirring": self._handle_lense_thirring,
            "alcubierre_energy": self._handle_alcubierre,
            "alcubierre_sweep": self._handle_alcubierre_sweep,
            "casimir_analysis": self._handle_casimir,
            "casimir_sweep": self._handle_casimir_sweep,
            "casimir_inc_comparison": self._handle_casimir_inc,
            "propulsion_survey": self._handle_propulsion_survey,
            "gravity_probe_b_validation": self._handle_gpb_validation,
            "full_analysis": self._handle_full_analysis,
        }

        handler = handlers.get(task_type)
        if handler is None:
            return {
                "error": f"Unknown task type: {task_type}",
                "supported_tasks": self.SUPPORTED_TASKS,
            }

        return handler(task)

    def _handle_lense_thirring(self, task: dict[str, Any]) -> dict[str, Any]:
        J = task.get("angular_momentum", CONST.J_earth)
        r = task.get("distance", CONST.R_earth)

        omega = self.lense_thirring.precession_rate(J, r)
        v_fd = self.lense_thirring.frame_dragging_velocity(J, r)

        # Convert to human-readable units
        mas_per_year = omega * (180 / np.pi) * 3600 * 1000 * (365.25 * 24 * 3600)

        hyp_id = self.journal.log_hypothesis(
            hypothesis=f"Frame-dragging at J={J:.2e} kg·m²/s, r={r:.2e} m is measurable",
            mathematical_prediction=f"Ω_LT = {omega:.6e} rad/s = {mas_per_year:.4f} mas/yr",
            falsification_test=f"Gyroscope with sensitivity < {omega/10:.2e} rad/s",
            required_sensitivity=f"< {omega/10:.2e} rad/s (10× signal threshold)",
        )

        return {
            "task": "lense_thirring",
            "angular_momentum_kg_m2_s": J,
            "distance_m": r,
            "precession_rate_rad_s": omega,
            "precession_rate_mas_yr": mas_per_year,
            "frame_dragging_velocity_m_s": v_fd,
            "hypothesis_id": hyp_id,
        }

    def _handle_alcubierre(self, task: dict[str, Any]) -> dict[str, Any]:
        velocity_c = task.get("velocity_c", 0.1)
        R = task.get("bubble_radius_m", 100.0)
        sigma = task.get("sigma", 1.0)

        result = self.alcubierre.total_energy(velocity_c, R, sigma)

        self.journal.log_hypothesis(
            hypothesis=f"Alcubierre bubble at {velocity_c}c with R={R}m requires computable exotic energy",
            mathematical_prediction=f"E_exotic = {result['total_exotic_energy_J']:.4e} J",
            falsification_test="Demonstrate negative energy density in laboratory Casimir cavity",
            references=["Alcubierre 1994, Class. Quantum Grav. 11 L73"],
        )

        return {"task": "alcubierre_energy", **result}

    def _handle_alcubierre_sweep(self, task: dict[str, Any]) -> dict[str, Any]:
        results = self.alcubierre.parameter_sweep()
        return {
            "task": "alcubierre_sweep",
            "configurations_tested": len(results),
            "optimal_config": results[0] if results else None,
            "all_results": results[:10],  # Top 10
        }

    def _handle_casimir(self, task: dict[str, Any]) -> dict[str, Any]:
        sep = task.get("separation_m", 100e-9)
        area = task.get("area_m2", 1e-6)
        enhancement = task.get("enhancement", 50.0)

        base_pressure = self.casimir.pressure(sep)
        enhanced = self.casimir.metamaterial_enhanced(sep, area, enhancement)
        energy = self.casimir.energy_density(sep)
        power = self.casimir.cavity_power_estimate(sep, area, 1e9, enhancement)

        self.journal.log_hypothesis(
            hypothesis=f"Metamaterial-enhanced Casimir force at d={sep*1e9:.1f}nm produces engineering-relevant pressure",
            mathematical_prediction=f"P_enhanced = {enhanced['enhanced_pressure_Pa']:.4e} Pa",
            falsification_test=f"AFM measurement with {sep*1e9:.1f}nm gap between metamaterial surfaces",
            required_sensitivity=f"< {abs(base_pressure)/100:.2e} Pa",
        )

        return {
            "task": "casimir_analysis",
            "separation_m": sep,
            "separation_nm": sep * 1e9,
            "area_m2": area,
            "base_pressure_Pa": base_pressure,
            "energy_density_J_m2": energy,
            **enhanced,
            "power_estimate": power,
        }

    def _handle_casimir_sweep(self, task: dict[str, Any]) -> dict[str, Any]:
        sweep = self.casimir.separation_sweep()
        return {"task": "casimir_sweep", **sweep}

    def _handle_casimir_inc(self, task: dict[str, Any]) -> dict[str, Any]:
        comparison = self.casimir.compare_to_casimir_inc()
        return {"task": "casimir_inc_comparison", **comparison}

    def _handle_gpb_validation(self, task: dict[str, Any]) -> dict[str, Any]:
        validation = self.lense_thirring.validate_against_gravity_probe_b()
        return {"task": "gravity_probe_b_validation", **validation}

    def _handle_propulsion_survey(self, task: dict[str, Any]) -> dict[str, Any]:
        return {
            "task": "propulsion_survey",
            "concepts": [
                {
                    "name": "Mach Effect Thruster (Woodward)",
                    "status": "Experimental — unconfirmed",
                    "principle": "Transient mass fluctuations from Mach's principle",
                    "equation": "δm = (1/4πG)(∂²E₀/∂t²) / (ρ₀c²)",
                    "falsification": "Thrust measurable at > 10× thermal noise floor",
                    "current_labs": ["CSUF (Broyles/Horn/Fearn)"],
                },
                {
                    "name": "Casimir Propulsion",
                    "status": "Theoretical — Casimir force experimentally verified",
                    "principle": "Asymmetric Casimir cavities for net unidirectional force",
                    "equation": "F/A = -π²ℏc/(240d⁴) × η_metamaterial",
                    "falsification": "Net force on asymmetric cavity in high vacuum",
                    "current_labs": ["Casimir, Inc. (commercial)", "Various university labs"],
                },
                {
                    "name": "Alcubierre Warp Drive",
                    "status": "Purely theoretical — requires exotic matter",
                    "principle": "Spacetime metric engineering",
                    "equation": "ds²=-dt²+(dx-v_s·f(r_s)dt)²+dy²+dz²",
                    "falsification": "Demonstrate negative energy density material",
                    "current_labs": ["NASA Eagleworks (inactive)", "Limitless Space Institute"],
                },
                {
                    "name": "Quantum Vacuum Thruster",
                    "status": "Speculative",
                    "principle": "Directional vacuum radiation pressure manipulation",
                    "equation": "Derived from QED — no consensus model",
                    "falsification": "Measurable thrust in ultra-high vacuum without propellant",
                    "current_labs": ["No known active programs"],
                },
                {
                    "name": "EM Drive (Cavity Thruster)",
                    "status": "Likely artifact — Dresden TU null result (2021)",
                    "principle": "RF cavity resonance (mechanism unknown)",
                    "equation": "No accepted theoretical basis",
                    "falsification": "Thrust persisting in improved vacuum setup",
                    "current_labs": ["Effectively discontinued"],
                },
            ],
        }

    def _handle_full_analysis(self, task: dict[str, Any]) -> dict[str, Any]:
        """Run a comprehensive exotic physics analysis."""
        results = {
            "task": "full_analysis",
            "lense_thirring": self._handle_lense_thirring(task),
            "gravity_probe_b": self._handle_gpb_validation(task),
            "alcubierre": self._handle_alcubierre(task),
            "casimir": self._handle_casimir(task),
            "casimir_inc": self._handle_casimir_inc(task),
            "propulsion_survey": self._handle_propulsion_survey(task),
            "research_journal": self.journal.get_all(),
        }
        return results
