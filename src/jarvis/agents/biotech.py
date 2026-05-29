"""
JARVIS-PRIME Biotech Agent
============================

Capabilities:
- Protein sequence analysis (computed locally)
- Drug-target interaction analysis
- CRISPR guide RNA design (basic off-target scoring)
- Clinical trial search (ClinicalTrials.gov API — free)
- Molecular weight and property calculations

All computations are local or use free public APIs.
"""
from __future__ import annotations

import math
from typing import Any

from jarvis.agents.base_agent import BaseAgent


# Amino acid molecular weights (Da)
AA_WEIGHTS = {
    'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.16,
    'E': 147.13, 'Q': 146.15, 'G': 75.03, 'H': 155.16, 'I': 131.17,
    'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
    'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15,
}

# DNA/RNA complement
DNA_COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}


class BiotechAgent(BaseAgent):
    """Domain agent for biotechnology and life sciences."""

    SUPPORTED_TASKS = [
        "protein_analysis",
        "crispr_design",
        "drug_target",
        "sequence_stats",
        "clinical_trials",
    ]

    def __init__(self):
        super().__init__(name="BiotechAgent", domain="biotech")

    def get_capabilities(self) -> list[str]:
        return [
            "protein_sequence_analysis",
            "crispr_guide_rna_design",
            "drug_target_interaction",
            "molecular_weight_calculation",
            "clinical_trial_search",
            "sequence_composition",
        ]

    async def execute(self, task: dict[str, Any], tools: list[Any]) -> dict[str, Any]:
        task_type = task.get("type", "sequence_stats")
        params = task.get("parameters", {})

        if task_type == "protein_analysis":
            return self._protein_analysis(params.get("sequence", ""))
        elif task_type == "crispr_design":
            return self._crispr_design(params.get("target_sequence", ""))
        elif task_type == "sequence_stats":
            return self._sequence_stats(params.get("sequence", ""))
        elif task_type == "drug_target":
            return self._drug_target_analysis(params)
        elif task_type == "clinical_trials":
            return self._clinical_trials_info(params)
        else:
            return self._biotech_overview(task)

    def _protein_analysis(self, sequence: str) -> dict[str, Any]:
        """Analyze a protein sequence."""
        seq = sequence.upper().replace(" ", "")
        if not seq:
            seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"  # Hemoglobin alpha

        length = len(seq)
        mw = sum(AA_WEIGHTS.get(aa, 0) for aa in seq) - (length - 1) * 18.015

        # Amino acid composition
        composition = {}
        for aa in seq:
            composition[aa] = composition.get(aa, 0) + 1

        # Hydrophobicity (Kyte-Doolittle)
        hydro_scale = {
            'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
            'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
            'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'D': -3.5,
            'E': -3.5, 'N': -3.5, 'Q': -3.5, 'K': -3.9, 'R': -4.5,
        }
        avg_hydro = sum(hydro_scale.get(aa, 0) for aa in seq) / max(length, 1)

        # Isoelectric point estimation (simplified)
        pos_charge = seq.count('K') + seq.count('R') + seq.count('H') * 0.1
        neg_charge = seq.count('D') + seq.count('E')
        estimated_pI = 7.0 + (pos_charge - neg_charge) * 0.1

        return {
            "task": "protein_analysis",
            "length": length,
            "molecular_weight_Da": round(mw, 2),
            "molecular_weight_kDa": round(mw / 1000, 2),
            "composition": dict(sorted(composition.items())),
            "avg_hydrophobicity": round(avg_hydro, 3),
            "estimated_pI": round(min(max(estimated_pI, 3.0), 12.0), 1),
            "is_likely_membrane": avg_hydro > 0.5,
            "charged_residues_pct": round((pos_charge + neg_charge) / max(length, 1) * 100, 1),
        }

    def _crispr_design(self, target: str) -> dict[str, Any]:
        """Design CRISPR guide RNAs for a target DNA sequence."""
        seq = target.upper().replace(" ", "")
        if not seq:
            seq = "ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTG"

        # Find PAM sites (NGG for SpCas9)
        guides = []
        for i in range(len(seq) - 22):
            if seq[i + 21:i + 23] == "GG":
                guide = seq[i:i + 20]
                # GC content
                gc = (guide.count('G') + guide.count('C')) / 20 * 100
                # Simple off-target score (GC in 40-60% is best)
                score = 100 - abs(gc - 50) * 2
                # Penalize poly-T (Pol III terminator)
                if "TTTT" in guide:
                    score -= 30

                guides.append({
                    "position": i,
                    "sequence_20nt": guide,
                    "pam": seq[i + 20:i + 23],
                    "gc_content_pct": round(gc, 1),
                    "on_target_score": max(round(score, 1), 0),
                    "strand": "sense",
                })

        guides.sort(key=lambda g: g["on_target_score"], reverse=True)

        return {
            "task": "crispr_design",
            "cas_protein": "SpCas9",
            "pam_motif": "NGG",
            "target_length": len(seq),
            "guides_found": len(guides),
            "top_guides": guides[:5],
            "note": "Scores are simplified. Use Benchling/CRISPOR for clinical-grade design.",
        }

    def _sequence_stats(self, sequence: str) -> dict[str, Any]:
        """Compute basic statistics for DNA/RNA/protein sequence."""
        seq = sequence.upper().replace(" ", "")
        if not seq:
            seq = "ATGCATGCATGCATGC"

        is_dna = all(c in "ATGCN" for c in seq)
        is_rna = all(c in "AUGCN" for c in seq)
        is_protein = all(c in AA_WEIGHTS for c in seq) and not is_dna

        result: dict[str, Any] = {
            "task": "sequence_stats",
            "length": len(seq),
            "type": "DNA" if is_dna else ("RNA" if is_rna else "protein"),
        }

        if is_dna or is_rna:
            result["gc_content_pct"] = round(
                (seq.count('G') + seq.count('C')) / max(len(seq), 1) * 100, 1
            )
            result["composition"] = {n: seq.count(n) for n in set(seq)}
            if is_dna:
                result["complement"] = "".join(DNA_COMPLEMENT.get(c, 'N') for c in seq)
                result["reverse_complement"] = result["complement"][::-1]
        elif is_protein:
            result.update(self._protein_analysis(seq))

        return result

    def _drug_target_analysis(self, params: dict[str, Any]) -> dict[str, Any]:
        """Provide drug-target interaction framework."""
        return {
            "task": "drug_target",
            "databases": [
                {"name": "ChEMBL", "url": "https://www.ebi.ac.uk/chembl/", "type": "Bioactivity"},
                {"name": "DrugBank", "url": "https://go.drugbank.com/", "type": "Drug data"},
                {"name": "PDB", "url": "https://www.rcsb.org/", "type": "3D structures"},
                {"name": "UniProt", "url": "https://www.uniprot.org/", "type": "Protein function"},
            ],
            "pipeline": [
                "1. Identify target protein (UniProt ID)",
                "2. Retrieve 3D structure (PDB/AlphaFold)",
                "3. Find known ligands (ChEMBL)",
                "4. Virtual screening (docking)",
                "5. ADMET prediction",
                "6. Lead optimization",
            ],
        }

    def _clinical_trials_info(self, params: dict[str, Any]) -> dict[str, Any]:
        """Provide clinical trials search guidance."""
        return {
            "task": "clinical_trials",
            "api": "ClinicalTrials.gov API v2",
            "url": "https://clinicaltrials.gov/api/v2/studies",
            "note": "Free public API, no authentication required",
            "example_query": "https://clinicaltrials.gov/api/v2/studies?query.cond=cancer&pageSize=5",
        }

    def _biotech_overview(self, task: dict[str, Any]) -> dict[str, Any]:
        """General biotech overview."""
        return {
            "task": "biotech_overview",
            "frontier_areas_2026": [
                "Prime editing clinical trials (CGD patient — protein restored)",
                "In-vivo CRISPR (hereditary angioedema, TTR disease)",
                "Epigenetic reprogramming human trials (glaucoma, skin aging)",
                "AI-driven drug discovery (AlphaFold 3, diffusion models)",
                "mRNA therapeutics beyond vaccines",
            ],
            "tools_available": self.get_capabilities(),
        }
