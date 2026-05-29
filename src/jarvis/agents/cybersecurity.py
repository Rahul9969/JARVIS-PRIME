"""
JARVIS-PRIME Cybersecurity Agent
==================================

Capabilities:
- CVE vulnerability lookup via NVD API (free, no key needed)
- MITRE ATT&CK technique mapping
- Threat modeling and risk assessment
- Security audit report generation

All data sources are free public APIs.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from jarvis.agents.base_agent import BaseAgent


# MITRE ATT&CK tactics (static mapping for offline use)
MITRE_TACTICS = {
    "TA0001": "Initial Access",
    "TA0002": "Execution",
    "TA0003": "Persistence",
    "TA0004": "Privilege Escalation",
    "TA0005": "Defense Evasion",
    "TA0006": "Credential Access",
    "TA0007": "Discovery",
    "TA0008": "Lateral Movement",
    "TA0009": "Collection",
    "TA0010": "Exfiltration",
    "TA0011": "Command and Control",
    "TA0040": "Impact",
    "TA0042": "Resource Development",
    "TA0043": "Reconnaissance",
}


class CybersecurityAgent(BaseAgent):
    """Domain agent for cybersecurity analysis."""

    SUPPORTED_TASKS = [
        "cve_lookup",
        "threat_model",
        "attack_surface",
        "mitre_mapping",
        "security_audit",
    ]

    def __init__(self):
        super().__init__(name="CybersecurityAgent", domain="cybersecurity")

    def get_capabilities(self) -> list[str]:
        return [
            "cve_vulnerability_lookup",
            "mitre_attack_mapping",
            "threat_modeling",
            "risk_assessment",
            "security_audit_generation",
        ]

    async def execute(self, task: dict[str, Any], tools: list[Any]) -> dict[str, Any]:
        task_type = task.get("type", "threat_model")

        if task_type == "cve_lookup":
            return await self._cve_lookup(task)
        elif task_type == "mitre_mapping":
            return self._mitre_mapping(task)
        elif task_type == "threat_model":
            return self._threat_model(task)
        elif task_type == "attack_surface":
            return self._attack_surface(task)
        elif task_type == "security_audit":
            return self._security_audit(task)
        else:
            return self._threat_model(task)

    async def _cve_lookup(self, task: dict[str, Any]) -> dict[str, Any]:
        """Query NIST NVD for CVE data (free, no API key needed)."""
        keyword = task.get("parameters", {}).get("keyword", task.get("description", ""))
        # Extract CVE ID if present
        cve_id = None
        for word in keyword.upper().split():
            if word.startswith("CVE-"):
                cve_id = word
                break

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                if cve_id:
                    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
                else:
                    search = keyword[:100].replace(" ", "+")
                    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch={search}&resultsPerPage=5"

                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()

                vulns = []
                for item in data.get("vulnerabilities", [])[:5]:
                    cve = item.get("cve", {})
                    descriptions = cve.get("descriptions", [])
                    desc = next((d["value"] for d in descriptions if d["lang"] == "en"), "N/A")
                    metrics = cve.get("metrics", {})

                    # Extract CVSS score
                    cvss_score = None
                    for version in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
                        if version in metrics and metrics[version]:
                            cvss_data = metrics[version][0].get("cvssData", {})
                            cvss_score = cvss_data.get("baseScore")
                            break

                    vulns.append({
                        "cve_id": cve.get("id", "Unknown"),
                        "description": desc[:300],
                        "cvss_score": cvss_score,
                        "published": cve.get("published", "Unknown"),
                    })

                return {
                    "task": "cve_lookup",
                    "query": keyword,
                    "total_results": data.get("totalResults", 0),
                    "vulnerabilities": vulns,
                }

        except Exception as e:
            return {
                "task": "cve_lookup",
                "query": keyword,
                "error": str(e),
                "note": "NVD API may be rate-limited. Try again in 30 seconds.",
            }

    def _mitre_mapping(self, task: dict[str, Any]) -> dict[str, Any]:
        """Map attack scenarios to MITRE ATT&CK framework."""
        return {
            "task": "mitre_mapping",
            "framework": "MITRE ATT&CK v15",
            "tactics": MITRE_TACTICS,
            "common_techniques": {
                "T1566": {"name": "Phishing", "tactic": "Initial Access"},
                "T1059": {"name": "Command and Scripting Interpreter", "tactic": "Execution"},
                "T1053": {"name": "Scheduled Task/Job", "tactic": "Persistence"},
                "T1548": {"name": "Abuse Elevation Control", "tactic": "Privilege Escalation"},
                "T1027": {"name": "Obfuscated Files", "tactic": "Defense Evasion"},
                "T1003": {"name": "OS Credential Dumping", "tactic": "Credential Access"},
                "T1046": {"name": "Network Service Discovery", "tactic": "Discovery"},
                "T1021": {"name": "Remote Services", "tactic": "Lateral Movement"},
            },
        }

    def _threat_model(self, task: dict[str, Any]) -> dict[str, Any]:
        """Generate a threat model using STRIDE methodology."""
        return {
            "task": "threat_model",
            "methodology": "STRIDE",
            "categories": {
                "Spoofing": "Impersonating a user or system component",
                "Tampering": "Modifying data or code without authorization",
                "Repudiation": "Denying actions without accountability",
                "Information Disclosure": "Exposing data to unauthorized parties",
                "Denial of Service": "Making system unavailable",
                "Elevation of Privilege": "Gaining unauthorized access levels",
            },
            "recommendation": "Apply defense-in-depth: input validation, encryption, RBAC, rate limiting, audit logging",
        }

    def _attack_surface(self, task: dict[str, Any]) -> dict[str, Any]:
        """Analyze attack surface for a system."""
        return {
            "task": "attack_surface",
            "areas": [
                {"surface": "Network", "risk": "high", "vectors": ["Open ports", "Unencrypted traffic", "DNS poisoning"]},
                {"surface": "Application", "risk": "high", "vectors": ["SQL injection", "XSS", "CSRF", "SSRF"]},
                {"surface": "Authentication", "risk": "critical", "vectors": ["Weak passwords", "Missing MFA", "Session hijacking"]},
                {"surface": "Supply Chain", "risk": "high", "vectors": ["Dependency vulnerabilities", "Compromised packages"]},
                {"surface": "Physical", "risk": "medium", "vectors": ["Device theft", "USB attacks", "Social engineering"]},
            ],
        }

    def _security_audit(self, task: dict[str, Any]) -> dict[str, Any]:
        """Generate a security audit checklist."""
        return {
            "task": "security_audit",
            "checklist": [
                {"item": "TLS 1.3 on all endpoints", "category": "Transport", "priority": "critical"},
                {"item": "Input validation on all user inputs", "category": "Application", "priority": "critical"},
                {"item": "Rate limiting on authentication endpoints", "category": "Authentication", "priority": "high"},
                {"item": "Dependency vulnerability scanning (Snyk/Dependabot)", "category": "Supply Chain", "priority": "high"},
                {"item": "Secrets stored in vault, not environment variables", "category": "Configuration", "priority": "high"},
                {"item": "Audit logging for all privileged operations", "category": "Monitoring", "priority": "medium"},
                {"item": "Regular penetration testing schedule", "category": "Process", "priority": "medium"},
            ],
        }
