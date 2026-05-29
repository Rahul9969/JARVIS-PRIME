"""
JARVIS-PRIME Energy & Fusion Agent
=====================================

Capabilities:
- Tokamak plasma confinement analysis (Lawson criterion)
- Fusion reactor energy balance
- Solar energy estimation
- Battery storage analysis
- Energy grid optimization concepts

All computations are local — no external APIs required.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from jarvis.agents.base_agent import BaseAgent


# ══════════════════════════════════════════════════════════
# Physics Constants for Fusion/Energy
# ══════════════════════════════════════════════════════════

class FusionConstants:
    k_B = 1.380649e-23       # Boltzmann constant (J/K)
    e_charge = 1.602176634e-19  # Elementary charge (C)
    m_proton = 1.67262192e-27   # Proton mass (kg)
    m_deuterium = 3.3436e-27    # Deuterium mass (kg)
    m_tritium = 5.0074e-27      # Tritium mass (kg)
    mu_0 = 1.2566370621e-6     # Vacuum permeability (H/m)
    E_DT = 17.6e6 * 1.602176634e-19  # D-T fusion energy release (J) = 17.6 MeV
    sigma_v_peak = 8.5e-22     # Peak <sigma*v> for D-T at ~70 keV (m^3/s)


class TokamakAnalyzer:
    """
    Tokamak fusion reactor analysis.

    Implements:
    - Lawson criterion for plasma ignition
    - Energy confinement time estimation
    - Beta limit (Troyon limit)
    - Magnetic field requirements
    - Power balance
    """

    def __init__(self):
        self.const = FusionConstants()

    def lawson_criterion(
        self,
        density_m3: float = 1e20,
        temperature_keV: float = 15.0,
        confinement_time_s: float = 3.0,
    ) -> dict[str, Any]:
        """
        Evaluate the Lawson criterion: n * tau_E * T > threshold.

        For D-T fusion ignition: n * tau_E > ~1.5e20 m^-3 s at T ~ 15 keV.
        """
        # Triple product
        triple_product = density_m3 * confinement_time_s * temperature_keV

        # Lawson threshold for D-T (approximate)
        lawson_threshold = 3e21  # keV * m^-3 * s for Q > 1

        # Ignition threshold
        ignition_threshold = 3e22  # keV * m^-3 * s for self-sustaining

        # Q factor estimate (fusion power / input power)
        Q_estimate = triple_product / lawson_threshold

        return {
            "density_m3": density_m3,
            "temperature_keV": temperature_keV,
            "confinement_time_s": confinement_time_s,
            "triple_product": triple_product,
            "lawson_threshold": lawson_threshold,
            "ignition_threshold": ignition_threshold,
            "Q_factor_estimate": round(Q_estimate, 3),
            "status": (
                "IGNITION" if triple_product > ignition_threshold
                else "BREAKEVEN" if triple_product > lawson_threshold
                else "SUB-BREAKEVEN"
            ),
        }

    def magnetic_confinement(
        self,
        major_radius_m: float = 6.2,   # ITER-like
        minor_radius_m: float = 2.0,
        toroidal_field_T: float = 5.3,
        plasma_current_MA: float = 15.0,
    ) -> dict[str, Any]:
        """
        Analyze tokamak magnetic confinement parameters.
        """
        # Aspect ratio
        aspect_ratio = major_radius_m / minor_radius_m

        # Safety factor q (edge) — simplified
        q_edge = (5 * minor_radius_m**2 * toroidal_field_T) / (
            major_radius_m * plasma_current_MA
        )

        # Plasma volume (torus)
        volume = 2 * math.pi**2 * major_radius_m * minor_radius_m**2

        # Troyon beta limit
        beta_max_pct = 2.8 * plasma_current_MA / (minor_radius_m * toroidal_field_T)

        # Magnetic pressure
        B_pressure = toroidal_field_T**2 / (2 * self.const.mu_0)

        # Estimated stored magnetic energy
        E_magnetic = B_pressure * volume

        return {
            "major_radius_m": major_radius_m,
            "minor_radius_m": minor_radius_m,
            "aspect_ratio": round(aspect_ratio, 2),
            "toroidal_field_T": toroidal_field_T,
            "plasma_current_MA": plasma_current_MA,
            "safety_factor_q": round(q_edge, 2),
            "q_stable": q_edge > 2.0,
            "plasma_volume_m3": round(volume, 1),
            "troyon_beta_limit_pct": round(beta_max_pct, 2),
            "magnetic_pressure_Pa": round(B_pressure, 0),
            "stored_energy_MJ": round(E_magnetic / 1e6, 1),
        }

    def fusion_power(
        self,
        density_m3: float = 1e20,
        temperature_keV: float = 15.0,
        volume_m3: float = 840.0,
    ) -> dict[str, Any]:
        """Estimate D-T fusion power output."""
        # <sigma*v> approximation at given temperature
        T = temperature_keV
        sigma_v = self.const.sigma_v_peak * math.exp(-abs(T - 70) / 30)

        # Fusion power: P = n_D * n_T * <sigma*v> * E_DT * V
        # Assume 50/50 D-T mix: n_D = n_T = n/2
        n_half = density_m3 / 2
        P_fusion = n_half**2 * sigma_v * self.const.E_DT * volume_m3

        # Alpha heating (20% of fusion energy goes to alpha particles)
        P_alpha = 0.2 * P_fusion

        return {
            "density_m3": density_m3,
            "temperature_keV": temperature_keV,
            "volume_m3": volume_m3,
            "sigma_v_m3_per_s": sigma_v,
            "fusion_power_MW": round(P_fusion / 1e6, 1),
            "alpha_heating_MW": round(P_alpha / 1e6, 1),
            "neutron_power_MW": round(0.8 * P_fusion / 1e6, 1),
        }


class SolarAnalyzer:
    """Solar energy estimation and panel analysis."""

    SOLAR_CONSTANT = 1361.0  # W/m^2 at Earth's distance

    def panel_output(
        self,
        area_m2: float = 100.0,
        efficiency: float = 0.22,
        latitude_deg: float = 28.6,   # New Delhi
        cloud_factor: float = 0.7,
    ) -> dict[str, Any]:
        """Estimate solar panel energy output."""
        # Average insolation (accounting for day/night, atmosphere, latitude)
        cos_lat = math.cos(math.radians(latitude_deg))
        avg_insolation = self.SOLAR_CONSTANT * 0.25 * cos_lat * cloud_factor  # Day avg

        peak_power_kW = area_m2 * efficiency * avg_insolation / 1000
        daily_energy_kWh = peak_power_kW * 5.0  # ~5 peak sun hours avg

        return {
            "area_m2": area_m2,
            "efficiency_pct": efficiency * 100,
            "latitude_deg": latitude_deg,
            "avg_insolation_W_m2": round(avg_insolation, 1),
            "peak_power_kW": round(peak_power_kW, 2),
            "daily_energy_kWh": round(daily_energy_kWh, 2),
            "annual_energy_MWh": round(daily_energy_kWh * 365 / 1000, 2),
            "co2_avoided_kg_yr": round(daily_energy_kWh * 365 * 0.4, 0),
        }


class EnergyAgent(BaseAgent):
    """Domain agent for energy and fusion research."""

    def __init__(self):
        super().__init__(name="EnergyAgent", domain="energy")
        self.tokamak = TokamakAnalyzer()
        self.solar = SolarAnalyzer()

    def get_capabilities(self) -> list[str]:
        return [
            "lawson_criterion_analysis",
            "tokamak_confinement",
            "fusion_power_estimation",
            "solar_energy_calculation",
            "energy_comparison",
            "fusion_reactor_status",
        ]

    async def execute(self, task: dict[str, Any], tools: list[Any]) -> dict[str, Any]:
        task_type = task.get("type", "energy_overview")
        params = task.get("parameters", {})

        if task_type == "lawson":
            return self.tokamak.lawson_criterion(
                density_m3=params.get("density", 1e20),
                temperature_keV=params.get("temperature", 15.0),
                confinement_time_s=params.get("confinement_time", 3.0),
            )
        elif task_type == "tokamak_confinement":
            return self.tokamak.magnetic_confinement()
        elif task_type == "fusion_power":
            return self.tokamak.fusion_power()
        elif task_type == "solar":
            return self.solar.panel_output()
        elif task_type == "full_analysis":
            return self._full_analysis()
        else:
            return self._energy_overview()

    def _full_analysis(self) -> dict[str, Any]:
        return {
            "task": "full_energy_analysis",
            "lawson": self.tokamak.lawson_criterion(),
            "confinement": self.tokamak.magnetic_confinement(),
            "fusion_power": self.tokamak.fusion_power(),
            "solar_comparison": self.solar.panel_output(),
        }

    def _energy_overview(self) -> dict[str, Any]:
        return {
            "task": "energy_overview",
            "frontier_2026": {
                "CFS_SPARC": "75% construction complete, first plasma 2027, 400MW ARC 2030s",
                "Helion_Polaris": "First private D-T fusion, 150M C, Orion plant 2028",
                "NIF": "8.6 MJ output (4x laser input), April 2025",
                "ITER": "First plasma delayed to 2030s, 500MW target",
            },
            "comparison": {
                "fusion_DT": "17.6 MeV per reaction, fuel from seawater",
                "fission_U235": "200 MeV per fission, radioactive waste",
                "solar": "~22% efficient panels, intermittent, land-intensive",
                "wind": "~45% capacity factor offshore, variable",
            },
        }
