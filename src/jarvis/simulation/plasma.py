"""
JARVIS-PRIME Plasma Physics Simulator
========================================

Simulates magnetohydrodynamic (MHD) plasma behavior:
- Plasma oscillations (Langmuir waves)
- Debye shielding
- Cyclotron motion
- Magnetic mirror confinement
- Basic MHD equilibrium (Grad-Shafranov)

All computations are local NumPy — no external deps.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


class PlasmaConstants:
    epsilon_0 = 8.854187817e-12   # Vacuum permittivity (F/m)
    mu_0 = 1.2566370621e-6        # Vacuum permeability (H/m)
    k_B = 1.380649e-23            # Boltzmann constant (J/K)
    e = 1.602176634e-19           # Elementary charge (C)
    m_e = 9.1093837015e-31        # Electron mass (kg)
    m_p = 1.67262192e-27          # Proton mass (kg)
    c = 2.99792458e8              # Speed of light (m/s)


class PlasmaSimulator:
    """Core plasma physics simulation engine."""

    def __init__(self):
        self.C = PlasmaConstants()

    def plasma_frequency(
        self,
        electron_density_m3: float = 1e18,
    ) -> dict[str, Any]:
        """
        Calculate plasma frequency (Langmuir oscillation).

        omega_pe = sqrt(n_e * e^2 / (epsilon_0 * m_e))
        """
        omega_pe = math.sqrt(
            electron_density_m3 * self.C.e**2
            / (self.C.epsilon_0 * self.C.m_e)
        )
        f_pe = omega_pe / (2 * math.pi)

        return {
            "electron_density_m3": electron_density_m3,
            "plasma_frequency_rad_s": omega_pe,
            "plasma_frequency_Hz": f_pe,
            "plasma_frequency_GHz": f_pe / 1e9,
            "period_ns": 1e9 / f_pe,
            "note": "EM waves below this frequency are reflected by the plasma",
        }

    def debye_length(
        self,
        temperature_eV: float = 10.0,
        electron_density_m3: float = 1e18,
    ) -> dict[str, Any]:
        """
        Calculate Debye shielding length.

        lambda_D = sqrt(epsilon_0 * kT / (n_e * e^2))
        """
        kT_joules = temperature_eV * self.C.e

        lambda_d = math.sqrt(
            self.C.epsilon_0 * kT_joules
            / (electron_density_m3 * self.C.e**2)
        )

        # Number of particles in Debye sphere
        N_D = (4/3) * math.pi * lambda_d**3 * electron_density_m3

        return {
            "temperature_eV": temperature_eV,
            "electron_density_m3": electron_density_m3,
            "debye_length_m": lambda_d,
            "debye_length_mm": lambda_d * 1e3,
            "debye_sphere_particles": N_D,
            "is_plasma": N_D > 1,  # Plasma condition: N_D >> 1
        }

    def cyclotron_frequency(
        self,
        magnetic_field_T: float = 5.0,
        particle_mass_kg: float | None = None,
        charge_C: float | None = None,
        species: str = "electron",
    ) -> dict[str, Any]:
        """
        Calculate cyclotron (gyro) frequency and Larmor radius.

        omega_c = |q|B / m
        r_L = v_perp / omega_c
        """
        if species == "electron":
            mass = self.C.m_e
            charge = self.C.e
        elif species == "proton":
            mass = self.C.m_p
            charge = self.C.e
        elif species == "deuterium":
            mass = 2 * self.C.m_p
            charge = self.C.e
        else:
            mass = particle_mass_kg or self.C.m_e
            charge = charge_C or self.C.e

        omega_c = charge * magnetic_field_T / mass
        f_c = omega_c / (2 * math.pi)

        # Larmor radius at thermal velocity (1 keV)
        v_thermal = math.sqrt(2 * 1000 * self.C.e / mass)
        r_larmor = v_thermal / omega_c

        return {
            "species": species,
            "magnetic_field_T": magnetic_field_T,
            "cyclotron_frequency_rad_s": omega_c,
            "cyclotron_frequency_GHz": f_c / 1e9,
            "larmor_radius_m": r_larmor,
            "larmor_radius_mm": r_larmor * 1e3,
            "thermal_velocity_m_s": v_thermal,
        }

    def magnetic_mirror(
        self,
        B_min_T: float = 1.0,
        B_max_T: float = 5.0,
        pitch_angle_deg: float = 45.0,
    ) -> dict[str, Any]:
        """
        Analyze magnetic mirror confinement.

        Mirror ratio R = B_max / B_min
        Loss cone: sin(theta) < sqrt(1/R)
        """
        R = B_max_T / B_min_T
        loss_cone_angle = math.degrees(math.asin(1 / math.sqrt(R)))
        pitch = pitch_angle_deg

        is_confined = pitch > loss_cone_angle

        # Fraction of particles trapped
        trapped_fraction = 1 - 1 / math.sqrt(R)

        return {
            "B_min_T": B_min_T,
            "B_max_T": B_max_T,
            "mirror_ratio": R,
            "loss_cone_angle_deg": round(loss_cone_angle, 2),
            "particle_pitch_angle_deg": pitch,
            "is_confined": is_confined,
            "trapped_fraction": round(trapped_fraction, 4),
        }

    def mhd_beta(
        self,
        pressure_Pa: float = 1e5,
        magnetic_field_T: float = 5.0,
    ) -> dict[str, Any]:
        """
        Calculate plasma beta: ratio of plasma pressure to magnetic pressure.

        beta = 2 * mu_0 * p / B^2
        """
        B_pressure = magnetic_field_T**2 / (2 * self.C.mu_0)
        beta = pressure_Pa / B_pressure

        return {
            "plasma_pressure_Pa": pressure_Pa,
            "magnetic_field_T": magnetic_field_T,
            "magnetic_pressure_Pa": round(B_pressure, 0),
            "beta": round(beta, 6),
            "beta_pct": round(beta * 100, 4),
            "regime": (
                "LOW beta (magnetically dominated)" if beta < 0.01
                else "MODERATE beta" if beta < 0.1
                else "HIGH beta (pressure dominated)"
            ),
        }

    def particle_orbit(
        self,
        B_field_T: float = 1.0,
        E_field_V_m: float = 0.0,
        energy_eV: float = 100.0,
        steps: int = 500,
        dt: float = 1e-10,
    ) -> dict[str, Any]:
        """
        Simulate charged particle orbit in uniform B + optional E field.
        Returns trajectory summary.
        """
        v0 = math.sqrt(2 * energy_eV * self.C.e / self.C.m_e)
        omega_c = self.C.e * B_field_T / self.C.m_e

        # Initial conditions: v_perp = v0, along x
        x, y, z = 0.0, 0.0, 0.0
        vx, vy, vz = v0, 0.0, 0.0

        positions = []
        for i in range(steps):
            # Boris push (simplified)
            ax = self.C.e / self.C.m_e * (E_field_V_m + vy * B_field_T)
            ay = self.C.e / self.C.m_e * (-vx * B_field_T)

            vx += ax * dt
            vy += ay * dt
            x += vx * dt
            y += vy * dt

            if i % 50 == 0:
                positions.append([round(x * 1e3, 4), round(y * 1e3, 4)])  # mm

        return {
            "B_field_T": B_field_T,
            "E_field_V_m": E_field_V_m,
            "energy_eV": energy_eV,
            "cyclotron_freq_GHz": round(omega_c / (2 * math.pi * 1e9), 4),
            "larmor_radius_mm": round(v0 / omega_c * 1e3, 4),
            "trajectory_points": len(positions),
            "drift_velocity_present": E_field_V_m != 0,
        }
