"""
JARVIS-PRIME Creative Studio Agent
=====================================

Capabilities:
- Research paper drafting and formatting
- Technical writing assistance
- Data visualization descriptions
- Presentation outlines
- Markdown report generation

Uses LLM when available, structured templates as fallback.
"""
from __future__ import annotations

import time
from typing import Any

from jarvis.agents.base_agent import BaseAgent


class CreativeAgent(BaseAgent):
    """Domain agent for creative and writing tasks."""

    SUPPORTED_TASKS = [
        "draft_paper",
        "write_report",
        "create_presentation",
        "format_results",
        "summarize",
    ]

    def __init__(self):
        super().__init__(name="CreativeAgent", domain="creative")

    def get_capabilities(self) -> list[str]:
        return [
            "research_paper_drafting",
            "technical_report_writing",
            "presentation_creation",
            "results_formatting",
            "executive_summary",
            "markdown_generation",
        ]

    async def execute(self, task: dict[str, Any], tools: list[Any]) -> dict[str, Any]:
        task_type = task.get("type", "write_report")

        if task_type == "draft_paper":
            return self._draft_paper(task)
        elif task_type == "write_report":
            return self._write_report(task)
        elif task_type == "create_presentation":
            return self._create_presentation(task)
        elif task_type == "format_results":
            return self._format_results(task)
        elif task_type == "summarize":
            return self._summarize(task)
        else:
            return self._write_report(task)

    def _draft_paper(self, task: dict[str, Any]) -> dict[str, Any]:
        """Generate a research paper template."""
        topic = task.get("description", "JARVIS-PRIME System Architecture")
        return {
            "task": "draft_paper",
            "template": {
                "title": f"Research Paper: {topic}",
                "sections": [
                    {"name": "Abstract", "content": "[150-250 word summary of key findings]"},
                    {"name": "1. Introduction", "content": "[Background, motivation, problem statement]"},
                    {"name": "2. Related Work", "content": "[Prior art, gap analysis, positioning]"},
                    {"name": "3. Methodology", "content": "[Technical approach, algorithms, architecture]"},
                    {"name": "4. Results", "content": "[Quantitative results, comparisons, ablations]"},
                    {"name": "5. Discussion", "content": "[Interpretation, limitations, implications]"},
                    {"name": "6. Conclusion", "content": "[Summary, future work, open questions]"},
                    {"name": "References", "content": "[BibTeX formatted references]"},
                ],
            },
            "format": "LaTeX/Markdown",
            "style_guide": "IEEE/ACM conference format",
        }

    def _write_report(self, task: dict[str, Any]) -> dict[str, Any]:
        """Generate a technical report."""
        topic = task.get("description", "System Analysis Report")
        return {
            "task": "write_report",
            "report": {
                "title": topic,
                "date": time.strftime("%Y-%m-%d"),
                "executive_summary": f"This report presents analysis of: {topic}",
                "sections": [
                    "1. Overview and Objectives",
                    "2. Methodology",
                    "3. Findings and Analysis",
                    "4. Recommendations",
                    "5. Next Steps",
                ],
                "format": "Markdown",
            },
        }

    def _create_presentation(self, task: dict[str, Any]) -> dict[str, Any]:
        """Generate a presentation outline."""
        topic = task.get("description", "Research Presentation")
        return {
            "task": "create_presentation",
            "slides": [
                {"number": 1, "title": topic, "type": "title_slide"},
                {"number": 2, "title": "Problem Statement", "bullets": 3},
                {"number": 3, "title": "Our Approach", "bullets": 4},
                {"number": 4, "title": "Architecture Overview", "type": "diagram"},
                {"number": 5, "title": "Key Results", "type": "data_visualization"},
                {"number": 6, "title": "Comparison", "type": "table"},
                {"number": 7, "title": "Discussion", "bullets": 3},
                {"number": 8, "title": "Future Work & Questions", "type": "closing"},
            ],
        }

    def _format_results(self, task: dict[str, Any]) -> dict[str, Any]:
        """Format computation results into readable output."""
        return {
            "task": "format_results",
            "formats_available": [
                "Markdown table",
                "LaTeX table",
                "CSV",
                "JSON (structured)",
                "Plain text summary",
            ],
            "visualization_types": [
                "Line chart (trend data)",
                "Bar chart (comparisons)",
                "Heatmap (2D parameter sweeps)",
                "Scatter plot (correlations)",
            ],
        }

    def _summarize(self, task: dict[str, Any]) -> dict[str, Any]:
        """Generate an executive summary."""
        return {
            "task": "summarize",
            "note": "Use with LLM provider for AI-generated summaries",
            "template": {
                "headline": "[One-sentence key finding]",
                "key_points": ["Point 1", "Point 2", "Point 3"],
                "implications": "[What this means for the field]",
                "action_items": ["Next step 1", "Next step 2"],
            },
        }
