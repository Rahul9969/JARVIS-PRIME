"""
JARVIS-PRIME Legal & Financial Agent
=======================================

Capabilities:
- Patent analysis and prior art search concepts
- Regulatory compliance frameworks
- Financial modeling (compound interest, DCF, options)
- Risk assessment (VaR, Monte Carlo)
- Contract clause analysis templates
"""
from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

from jarvis.agents.base_agent import BaseAgent


class FinancialCalculator:
    """Financial modeling and analysis tools."""

    @staticmethod
    def compound_interest(
        principal: float = 10000,
        annual_rate: float = 0.08,
        years: int = 10,
        compounding: int = 12,
    ) -> dict[str, Any]:
        """Calculate compound interest."""
        r = annual_rate / compounding
        n = compounding * years
        final = principal * (1 + r) ** n
        total_interest = final - principal

        return {
            "principal": principal,
            "annual_rate_pct": annual_rate * 100,
            "years": years,
            "compounding_per_year": compounding,
            "final_value": round(final, 2),
            "total_interest": round(total_interest, 2),
            "effective_annual_rate_pct": round(((1 + r) ** compounding - 1) * 100, 4),
        }

    @staticmethod
    def dcf_valuation(
        cash_flows: list[float] | None = None,
        discount_rate: float = 0.10,
        terminal_growth: float = 0.03,
    ) -> dict[str, Any]:
        """Discounted Cash Flow valuation."""
        if cash_flows is None:
            cash_flows = [100, 110, 121, 133, 146]

        pv_flows = []
        for i, cf in enumerate(cash_flows):
            pv = cf / (1 + discount_rate) ** (i + 1)
            pv_flows.append(round(pv, 2))

        # Terminal value (Gordon Growth Model)
        terminal_cf = cash_flows[-1] * (1 + terminal_growth)
        terminal_value = terminal_cf / (discount_rate - terminal_growth)
        pv_terminal = terminal_value / (1 + discount_rate) ** len(cash_flows)

        total_pv = sum(pv_flows) + pv_terminal

        return {
            "cash_flows": cash_flows,
            "discount_rate_pct": discount_rate * 100,
            "pv_cash_flows": pv_flows,
            "terminal_growth_pct": terminal_growth * 100,
            "terminal_value": round(terminal_value, 2),
            "pv_terminal": round(pv_terminal, 2),
            "total_present_value": round(total_pv, 2),
        }

    @staticmethod
    def black_scholes(
        S: float = 100.0,   # Stock price
        K: float = 100.0,   # Strike price
        T: float = 1.0,     # Time to expiry (years)
        r: float = 0.05,    # Risk-free rate
        sigma: float = 0.2, # Volatility
    ) -> dict[str, Any]:
        """Black-Scholes option pricing."""
        from math import log, sqrt, exp
        from statistics import NormalDist

        norm = NormalDist()

        d1 = (log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        call_price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        put_price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        # Greeks
        delta_call = norm.cdf(d1)
        delta_put = delta_call - 1
        gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
        vega = S * norm.pdf(d1) * sqrt(T) / 100
        theta_call = (-(S * norm.pdf(d1) * sigma) / (2 * sqrt(T))
                      - r * K * exp(-r * T) * norm.cdf(d2)) / 365

        return {
            "model": "Black-Scholes",
            "stock_price": S,
            "strike_price": K,
            "time_to_expiry_yr": T,
            "risk_free_rate": r,
            "volatility": sigma,
            "call_price": round(call_price, 4),
            "put_price": round(put_price, 4),
            "greeks": {
                "delta_call": round(delta_call, 4),
                "delta_put": round(delta_put, 4),
                "gamma": round(gamma, 6),
                "vega": round(vega, 4),
                "theta_call_per_day": round(theta_call, 4),
            },
        }

    @staticmethod
    def monte_carlo_var(
        portfolio_value: float = 1_000_000,
        daily_volatility: float = 0.02,
        confidence: float = 0.95,
        days: int = 10,
        simulations: int = 10_000,
    ) -> dict[str, Any]:
        """Monte Carlo Value at Risk estimation."""
        np.random.seed(42)

        returns = np.random.normal(0, daily_volatility, (simulations, days))
        cumulative_returns = np.cumprod(1 + returns, axis=1)
        final_values = portfolio_value * cumulative_returns[:, -1]

        losses = portfolio_value - final_values
        var_level = np.percentile(losses, confidence * 100)
        cvar = np.mean(losses[losses >= var_level])

        return {
            "portfolio_value": portfolio_value,
            "daily_volatility_pct": daily_volatility * 100,
            "confidence_level": confidence,
            "holding_period_days": days,
            "simulations": simulations,
            "VaR": round(float(var_level), 2),
            "VaR_pct": round(float(var_level / portfolio_value * 100), 2),
            "CVaR_expected_shortfall": round(float(cvar), 2),
            "max_loss": round(float(np.max(losses)), 2),
            "median_portfolio_value": round(float(np.median(final_values)), 2),
        }


class LegalAnalyzer:
    """Legal analysis and compliance tools."""

    @staticmethod
    def patent_framework() -> dict[str, Any]:
        """Patent application framework."""
        return {
            "sections": [
                {"name": "Title", "desc": "Descriptive title of the invention"},
                {"name": "Abstract", "desc": "150-word summary of the invention"},
                {"name": "Background", "desc": "Prior art and problem statement"},
                {"name": "Summary", "desc": "Brief description of the solution"},
                {"name": "Detailed Description", "desc": "Full technical specification"},
                {"name": "Claims", "desc": "Independent and dependent claims defining scope"},
                {"name": "Drawings", "desc": "Figures illustrating the invention"},
            ],
            "databases": [
                "Google Patents (free)", "USPTO (free)", "WIPO (free)",
                "Espacenet (free)", "Google Scholar (free)",
            ],
            "cost_estimate": {
                "provisional_USD": "1500-5000",
                "utility_USD": "10000-25000",
                "PCT_international_USD": "5000-15000",
            },
        }

    @staticmethod
    def compliance_frameworks() -> dict[str, Any]:
        """Major regulatory compliance frameworks."""
        return {
            "frameworks": {
                "GDPR": {"region": "EU", "focus": "Data privacy", "penalty": "4% global revenue or 20M EUR"},
                "SOC2": {"region": "Global", "focus": "Security controls", "penalty": "Loss of certification"},
                "HIPAA": {"region": "US", "focus": "Health data", "penalty": "Up to $1.5M per violation"},
                "PCI_DSS": {"region": "Global", "focus": "Payment data", "penalty": "$5K-100K/month non-compliance"},
                "ISO_27001": {"region": "Global", "focus": "InfoSec management", "penalty": "Loss of certification"},
                "AI_Act": {"region": "EU", "focus": "AI risk classification", "penalty": "35M EUR or 7% revenue"},
            },
        }

    @staticmethod
    def contract_analysis() -> dict[str, Any]:
        """Contract clause analysis template."""
        return {
            "key_clauses": [
                {"clause": "Limitation of Liability", "risk": "high", "check": "Cap amount, exclusions"},
                {"clause": "Indemnification", "risk": "high", "check": "Mutual vs one-sided, scope"},
                {"clause": "IP Ownership", "risk": "critical", "check": "Work-for-hire, license grants"},
                {"clause": "Termination", "risk": "medium", "check": "For cause vs convenience, notice period"},
                {"clause": "Confidentiality", "risk": "medium", "check": "Duration, exceptions, return of data"},
                {"clause": "Force Majeure", "risk": "low", "check": "Covered events, notification requirements"},
                {"clause": "Governing Law", "risk": "medium", "check": "Jurisdiction, arbitration vs litigation"},
            ],
        }


class LegalFinancialAgent(BaseAgent):
    """Domain agent for legal and financial analysis."""

    def __init__(self):
        super().__init__(name="LegalFinancialAgent", domain="legal_financial")
        self.finance = FinancialCalculator()
        self.legal = LegalAnalyzer()

    def get_capabilities(self) -> list[str]:
        return [
            "compound_interest_calculation",
            "dcf_valuation",
            "black_scholes_pricing",
            "monte_carlo_var",
            "patent_framework",
            "compliance_analysis",
            "contract_review",
            "risk_assessment",
        ]

    async def execute(self, task: dict[str, Any], tools: list[Any]) -> dict[str, Any]:
        task_type = task.get("type", "overview")
        params = task.get("parameters", {})

        if task_type == "compound_interest":
            return self.finance.compound_interest(**{k: params[k] for k in params if k in ["principal", "annual_rate", "years"]})
        elif task_type == "dcf":
            return self.finance.dcf_valuation()
        elif task_type == "options":
            return self.finance.black_scholes()
        elif task_type == "var":
            return self.finance.monte_carlo_var()
        elif task_type == "patent":
            return self.legal.patent_framework()
        elif task_type == "compliance":
            return self.legal.compliance_frameworks()
        elif task_type == "contract":
            return self.legal.contract_analysis()
        elif task_type == "full_analysis":
            return self._full_analysis()
        else:
            return self._overview()

    def _full_analysis(self) -> dict[str, Any]:
        return {
            "task": "full_legal_financial",
            "dcf": self.finance.dcf_valuation(),
            "options": self.finance.black_scholes(),
            "var": self.finance.monte_carlo_var(),
            "compliance": self.legal.compliance_frameworks(),
        }

    def _overview(self) -> dict[str, Any]:
        return {
            "task": "legal_financial_overview",
            "capabilities": self.get_capabilities(),
            "note": "Full financial modeling, patent search, and compliance analysis available",
        }
