"""
UI/UX PRO MAX PLUGIN
=====================
Design intelligence skill for Mel — provides BM25-powered search across
67 UI styles, 96 color palettes, 57 font pairings, 25 chart types, 99 UX
guidelines, and 13 stack-specific guideline sets.

Based on: https://github.com/nextlevelbuilder/ui-ux-pro-max-skill
"""

import csv
import re
import os
import logging
from math import log
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional

logger = logging.getLogger("ui_ux")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "ui_ux_data"
MAX_RESULTS = 3

CSV_CONFIG = {
    "style": {
        "file": "styles.csv",
        "search_cols": ["Style Category", "Keywords", "Best For", "Type", "AI Prompt Keywords"],
        "output_cols": ["Style Category", "Type", "Keywords", "Primary Colors", "Effects & Animation", "Best For", "Performance", "Accessibility", "Framework Compatibility", "Complexity", "AI Prompt Keywords", "CSS/Technical Keywords", "Implementation Checklist", "Design System Variables"],
    },
    "color": {
        "file": "colors.csv",
        "search_cols": ["Product Type", "Notes"],
        "output_cols": ["Product Type", "Primary", "On Primary", "Secondary", "On Secondary", "Accent", "On Accent", "Background", "Foreground", "Card", "Card Foreground", "Muted", "Muted Foreground", "Border", "Destructive", "On Destructive", "Ring", "Notes"],
    },
    "chart": {
        "file": "charts.csv",
        "search_cols": ["Data Type", "Keywords", "Best Chart Type", "When to Use", "When NOT to Use", "Accessibility Notes"],
        "output_cols": ["Data Type", "Keywords", "Best Chart Type", "Secondary Options", "When to Use", "When NOT to Use", "Data Volume Threshold", "Color Guidance", "Accessibility Grade", "Accessibility Notes", "A11y Fallback", "Library Recommendation", "Interactive Level"],
    },
    "landing": {
        "file": "landing.csv",
        "search_cols": ["Pattern Name", "Keywords", "Conversion Optimization", "Section Order"],
        "output_cols": ["Pattern Name", "Keywords", "Section Order", "Primary CTA Placement", "Color Strategy", "Conversion Optimization"],
    },
    "product": {
        "file": "products.csv",
        "search_cols": ["Product Type", "Keywords", "Primary Style Recommendation", "Key Considerations"],
        "output_cols": ["Product Type", "Keywords", "Primary Style Recommendation", "Secondary Styles", "Landing Page Pattern", "Dashboard Style (if applicable)", "Color Palette Focus"],
    },
    "ux": {
        "file": "ux-guidelines.csv",
        "search_cols": ["Category", "Issue", "Description", "Platform"],
        "output_cols": ["Category", "Issue", "Platform", "Description", "Do", "Don't", "Code Example Good", "Code Example Bad", "Severity"],
    },
    "typography": {
        "file": "typography.csv",
        "search_cols": ["Font Pairing Name", "Category", "Mood/Style Keywords", "Best For", "Heading Font", "Body Font"],
        "output_cols": ["Font Pairing Name", "Category", "Heading Font", "Body Font", "Mood/Style Keywords", "Best For", "Google Fonts URL", "CSS Import", "Tailwind Config", "Notes"],
    },
    "icons": {
        "file": "icons.csv",
        "search_cols": ["Category", "Icon Name", "Keywords", "Best For"],
        "output_cols": ["Category", "Icon Name", "Keywords", "Library", "Import Code", "Usage", "Best For", "Style"],
    },
    "react": {
        "file": "react-performance.csv",
        "search_cols": ["Category", "Issue", "Keywords", "Description"],
        "output_cols": ["Category", "Issue", "Platform", "Description", "Do", "Don't", "Code Example Good", "Code Example Bad", "Severity"],
    },
    "web": {
        "file": "app-interface.csv",
        "search_cols": ["Category", "Issue", "Keywords", "Description"],
        "output_cols": ["Category", "Issue", "Platform", "Description", "Do", "Don't", "Code Example Good", "Code Example Bad", "Severity"],
    },
    "google-fonts": {
        "file": "google-fonts.csv",
        "search_cols": ["Family", "Category", "Stroke", "Classifications", "Keywords", "Subsets", "Designers"],
        "output_cols": ["Family", "Category", "Stroke", "Classifications", "Styles", "Variable Axes", "Subsets", "Designers", "Popularity Rank", "Google Fonts URL"],
    },
}

STACK_CONFIG = {
    "react-native": {"file": "stacks/react-native.csv"},
}
_STACK_COLS = {
    "search_cols": ["Category", "Guideline", "Description", "Do", "Don't"],
    "output_cols": ["Category", "Guideline", "Description", "Do", "Don't", "Code Good", "Code Bad", "Severity", "Docs URL"],
}
AVAILABLE_STACKS = list(STACK_CONFIG.keys())
AVAILABLE_DOMAINS = list(CSV_CONFIG.keys())


# ─────────────────────────────────────────────
# BM25 Search Engine
# ─────────────────────────────────────────────
class BM25:
    """BM25 ranking algorithm for text search."""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_lengths = []
        self.avgdl = 0
        self.idf = {}
        self.doc_freqs = defaultdict(int)
        self.N = 0

    def tokenize(self, text):
        text = re.sub(r'[^\w\s]', ' ', str(text).lower())
        return [w for w in text.split() if len(w) > 2]

    def fit(self, documents):
        self.corpus = [self.tokenize(doc) for doc in documents]
        self.N = len(self.corpus)
        if self.N == 0:
            return
        self.doc_lengths = [len(doc) for doc in self.corpus]
        self.avgdl = sum(self.doc_lengths) / self.N

        for doc in self.corpus:
            seen = set()
            for word in doc:
                if word not in seen:
                    self.doc_freqs[word] += 1
                    seen.add(word)

        for word, freq in self.doc_freqs.items():
            self.idf[word] = log((self.N - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, query):
        query_tokens = self.tokenize(query)
        scores = []

        for idx, doc in enumerate(self.corpus):
            score = 0
            doc_len = self.doc_lengths[idx]
            term_freqs = defaultdict(int)
            for word in doc:
                term_freqs[word] += 1

            for token in query_tokens:
                if token in self.idf:
                    tf = term_freqs[token]
                    idf = self.idf[token]
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    score += idf * numerator / denominator

            scores.append((idx, score))

        return sorted(scores, key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────
# Search Functions
# ─────────────────────────────────────────────
def _load_csv(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _search_csv(filepath, search_cols, output_cols, query, max_results):
    if not filepath.exists():
        return []

    data = _load_csv(filepath)
    documents = [" ".join(str(row.get(col, "")) for col in search_cols) for row in data]

    bm25 = BM25()
    bm25.fit(documents)
    ranked = bm25.score(query)

    results = []
    for idx, score in ranked[:max_results]:
        if score > 0:
            row = data[idx]
            results.append({col: row.get(col, "") for col in output_cols if col in row})

    return results


def detect_domain(query):
    """Auto-detect the most relevant domain from query."""
    query_lower = query.lower()

    domain_keywords = {
        "color": ["color", "palette", "hex", "#", "rgb", "token", "semantic", "accent", "destructive", "muted", "foreground"],
        "chart": ["chart", "graph", "visualization", "trend", "bar", "pie", "scatter", "heatmap", "funnel"],
        "landing": ["landing", "page", "cta", "conversion", "hero", "testimonial", "pricing", "section"],
        "product": ["saas", "ecommerce", "e-commerce", "fintech", "healthcare", "gaming", "portfolio", "crypto", "dashboard", "fitness", "restaurant", "hotel", "travel", "music", "education", "learning", "legal", "insurance", "medical", "beauty", "pharmacy", "dental", "pet", "dating", "wedding", "recipe", "delivery", "ride", "booking", "calendar", "timer", "tracker", "diary", "note", "chat", "messenger", "crm", "invoice", "parking", "transit", "vpn", "alarm", "weather", "sleep", "meditation", "fasting", "habit", "grocery", "meme", "wardrobe", "plant care", "reading", "flashcard", "puzzle", "trivia", "arcade", "photography", "streaming", "podcast", "newsletter", "marketplace", "freelancer", "coworking", "airline", "museum", "theater", "church", "non-profit", "charity", "kindergarten", "daycare", "senior care", "veterinary", "florist", "bakery", "brewery", "construction", "automotive", "real estate", "logistics", "agriculture", "coding bootcamp"],
        "style": ["style", "design", "ui", "minimalism", "glassmorphism", "neumorphism", "brutalism", "dark mode", "flat", "aurora", "prompt", "css", "implementation", "variable", "checklist", "tailwind"],
        "ux": ["ux", "usability", "accessibility", "wcag", "touch", "scroll", "animation", "keyboard", "navigation", "mobile"],
        "typography": ["font pairing", "typography pairing", "heading font", "body font"],
        "google-fonts": ["google font", "font family", "font weight", "font style", "variable font", "noto", "font for", "find font", "font subset", "font language", "monospace font", "serif font", "sans serif font", "display font", "handwriting font", "font", "typography", "serif", "sans"],
        "icons": ["icon", "icons", "lucide", "heroicons", "symbol", "glyph", "pictogram", "svg icon"],
        "react": ["react", "next.js", "nextjs", "suspense", "memo", "usecallback", "useeffect", "rerender", "bundle", "waterfall", "barrel", "dynamic import", "rsc", "server component"],
        "web": ["aria", "focus", "outline", "semantic", "virtualize", "autocomplete", "form", "input type", "preconnect"],
    }

    scores = {
        domain: sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', query_lower))
        for domain, keywords in domain_keywords.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "style"


def search(query, domain=None, max_results=MAX_RESULTS):
    """Main search function with auto-domain detection."""
    if domain is None:
        domain = detect_domain(query)

    config = CSV_CONFIG.get(domain, CSV_CONFIG["style"])
    filepath = DATA_DIR / config["file"]

    if not filepath.exists():
        return {"error": f"Data file not found: {config['file']}", "domain": domain}

    results = _search_csv(filepath, config["search_cols"], config["output_cols"], query, max_results)

    return {
        "domain": domain,
        "query": query,
        "file": config["file"],
        "count": len(results),
        "results": results,
    }


def search_stack(query, stack, max_results=MAX_RESULTS):
    """Search stack-specific guidelines."""
    if stack not in STACK_CONFIG:
        return {"error": f"Unknown stack: {stack}. Available: {', '.join(AVAILABLE_STACKS)}"}

    filepath = DATA_DIR / STACK_CONFIG[stack]["file"]

    if not filepath.exists():
        return {"error": f"Stack file not found: {STACK_CONFIG[stack]['file']}", "stack": stack}

    results = _search_csv(filepath, _STACK_COLS["search_cols"], _STACK_COLS["output_cols"], query, max_results)

    return {
        "domain": "stack",
        "stack": stack,
        "query": query,
        "file": STACK_CONFIG[stack]["file"],
        "count": len(results),
        "results": results,
    }


# ─────────────────────────────────────────────
# Design System Generator
# ─────────────────────────────────────────────
REASONING_FILE = "ui-reasoning.csv"

DESIGN_SEARCH_CONFIG = {
    "product": {"max_results": 1},
    "style": {"max_results": 3},
    "color": {"max_results": 2},
    "landing": {"max_results": 2},
    "typography": {"max_results": 2},
}


class DesignSystemGenerator:
    """Generates design system recommendations from aggregated searches."""

    def __init__(self):
        self.reasoning_data = self._load_reasoning()

    def _load_reasoning(self) -> list:
        filepath = DATA_DIR / REASONING_FILE
        if not filepath.exists():
            return []
        with open(filepath, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))

    def _multi_domain_search(self, query: str, style_priority: list = None) -> dict:
        results = {}
        for domain, config in DESIGN_SEARCH_CONFIG.items():
            if domain == "style" and style_priority:
                priority_query = " ".join(style_priority[:2]) if style_priority else query
                combined_query = f"{query} {priority_query}"
                results[domain] = search(combined_query, domain, config["max_results"])
            else:
                results[domain] = search(query, domain, config["max_results"])
        return results

    def _find_reasoning_rule(self, category: str) -> dict:
        category_lower = category.lower()

        for rule in self.reasoning_data:
            if rule.get("UI_Category", "").lower() == category_lower:
                return rule

        for rule in self.reasoning_data:
            ui_cat = rule.get("UI_Category", "").lower()
            if ui_cat in category_lower or category_lower in ui_cat:
                return rule

        for rule in self.reasoning_data:
            ui_cat = rule.get("UI_Category", "").lower()
            keywords = ui_cat.replace("/", " ").replace("-", " ").split()
            if any(kw in category_lower for kw in keywords):
                return rule

        return {}

    def _apply_reasoning(self, category: str, search_results: dict) -> dict:
        rule = self._find_reasoning_rule(category)

        if not rule:
            return {
                "pattern": "Hero + Features + CTA",
                "style_priority": ["Minimalism", "Flat Design"],
                "color_mood": "Professional",
                "typography_mood": "Clean",
                "key_effects": "Subtle hover transitions",
                "anti_patterns": "",
                "decision_rules": {},
            }

        return {
            "pattern": rule.get("Landing_Page_Pattern", "Hero + Features + CTA"),
            "style_priority": [s.strip() for s in rule.get("Primary_Style", "Minimalism").split(",")],
            "color_mood": rule.get("Color_Mood", "Professional"),
            "typography_mood": rule.get("Typography_Mood", "Clean"),
            "key_effects": rule.get("Key_Effects", ""),
            "anti_patterns": rule.get("Anti_Patterns", ""),
            "decision_rules": {
                "primary_style": rule.get("Primary_Style", ""),
                "secondary_styles": rule.get("Secondary_Styles", ""),
                "dashboard_style": rule.get("Dashboard_Style", ""),
            },
        }

    def generate(self, query: str, project_name: str = None) -> dict:
        """Generate a complete design system recommendation."""
        # Step 1: Search product domain to identify category
        product_results = search(query, "product", 1)
        category = query
        style_priority = None

        if product_results.get("results"):
            product = product_results["results"][0]
            category = product.get("Product Type", query)
            style_rec = product.get("Primary Style Recommendation", "")
            if style_rec:
                style_priority = [s.strip() for s in style_rec.split(",")]

        # Step 2: Apply reasoning
        reasoning = self._apply_reasoning(category, {})

        # Step 3: Multi-domain search
        all_results = self._multi_domain_search(query, style_priority or reasoning.get("style_priority"))

        # Step 4: Compile design system
        design_system = {
            "project_name": project_name or query,
            "category": category,
            "generated_at": datetime.now().isoformat(),
            "reasoning": reasoning,
            "product": product_results.get("results", []),
            "styles": all_results.get("style", {}).get("results", []),
            "colors": all_results.get("color", {}).get("results", []),
            "landing_patterns": all_results.get("landing", {}).get("results", []),
            "typography": all_results.get("typography", {}).get("results", []),
        }

        return design_system


# ─────────────────────────────────────────────
# Format Helpers
# ─────────────────────────────────────────────
def format_search_results(result: dict) -> str:
    """Format search results for human-readable output."""
    if "error" in result:
        return f"Error: {result['error']}"

    output = []
    if result.get("stack"):
        output.append(f"## UI Pro Max Stack Guidelines")
        output.append(f"**Stack:** {result['stack']} | **Query:** {result['query']}")
    else:
        output.append(f"## UI Pro Max Search Results")
        output.append(f"**Domain:** {result['domain']} | **Query:** {result['query']}")
    output.append(f"**Source:** {result['file']} | **Found:** {result['count']} results\n")

    for i, row in enumerate(result['results'], 1):
        output.append(f"### Result {i}")
        for key, value in row.items():
            value_str = str(value)
            if len(value_str) > 300:
                value_str = value_str[:300] + "..."
            output.append(f"- **{key}:** {value_str}")
        output.append("")

    return "\n".join(output)


def format_design_system(ds: dict) -> str:
    """Format a design system dict into readable markdown."""
    lines = []
    lines.append(f"# Design System: {ds['project_name']}")
    lines.append(f"**Category:** {ds['category']}")
    lines.append(f"**Generated:** {ds['generated_at']}\n")

    reasoning = ds.get("reasoning", {})
    if reasoning:
        lines.append("## Reasoning")
        lines.append(f"- **Pattern:** {reasoning.get('pattern', 'N/A')}")
        lines.append(f"- **Style Priority:** {', '.join(reasoning.get('style_priority', []))}")
        lines.append(f"- **Color Mood:** {reasoning.get('color_mood', 'N/A')}")
        lines.append(f"- **Typography Mood:** {reasoning.get('typography_mood', 'N/A')}")
        if reasoning.get("key_effects"):
            lines.append(f"- **Key Effects:** {reasoning['key_effects']}")
        if reasoning.get("anti_patterns"):
            lines.append(f"- **Anti-Patterns:** {reasoning['anti_patterns']}")
        lines.append("")

    for section, key in [("Styles", "styles"), ("Colors", "colors"), ("Landing Patterns", "landing_patterns"), ("Typography", "typography")]:
        items = ds.get(key, [])
        if items:
            lines.append(f"## {section}")
            for i, item in enumerate(items, 1):
                lines.append(f"### {section} {i}")
                for k, v in item.items():
                    v_str = str(v)
                    if len(v_str) > 300:
                        v_str = v_str[:300] + "..."
                    lines.append(f"- **{k}:** {v_str}")
                lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Plugin Class (mel-agent integration)
# ─────────────────────────────────────────────
class UIUXPlugin:
    """UI/UX design intelligence plugin for Mel agent."""

    def __init__(self):
        self.generator = DesignSystemGenerator()
        logger.info(f"UI/UX Pro Max loaded: {len(AVAILABLE_DOMAINS)} domains, {len(AVAILABLE_STACKS)} stacks")

    async def search_designs(self, params: dict) -> str:
        """Search UI/UX design knowledge base.

        params:
            query (str): Search query
            domain (str, optional): Specific domain to search
            max_results (int, optional): Max results (default 3)
        """
        query = params.get("query", "")
        if not query:
            return "Please provide a search query."

        domain = params.get("domain")
        max_results = params.get("max_results", MAX_RESULTS)

        result = search(query, domain, max_results)
        return format_search_results(result)

    async def search_stack_guidelines(self, params: dict) -> str:
        """Search stack-specific guidelines.

        params:
            query (str): Search query
            stack (str): Stack name (e.g., react-native)
            max_results (int, optional): Max results (default 3)
        """
        query = params.get("query", "")
        stack = params.get("stack", "")

        if not query:
            return "Please provide a search query."
        if not stack:
            return f"Please specify a stack. Available: {', '.join(AVAILABLE_STACKS)}"

        result = search_stack(query, stack, params.get("max_results", MAX_RESULTS))
        return format_search_results(result)

    async def generate_design_system(self, params: dict) -> str:
        """Generate a complete design system recommendation.

        params:
            query (str): Product/project description (e.g., "SaaS dashboard")
            project_name (str, optional): Project name for the output
        """
        query = params.get("query", "")
        if not query:
            return "Please provide a project description."

        project_name = params.get("project_name")
        ds = self.generator.generate(query, project_name)
        return format_design_system(ds)

    async def list_domains(self, params: dict) -> str:
        """List all available search domains and stacks."""
        lines = ["## Available Domains"]
        for domain, config in CSV_CONFIG.items():
            lines.append(f"- **{domain}**: {config['file']}")
        lines.append("\n## Available Stacks")
        for stack in AVAILABLE_STACKS:
            lines.append(f"- **{stack}**")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Plugin Registration
# ─────────────────────────────────────────────
def register_ui_ux_plugins(action_registry):
    """Register UI/UX plugins with the action registry."""
    plugin = UIUXPlugin()

    action_registry.register(
        "ui_ux_search", plugin.search_designs,
        "Search UI/UX styles, colors, charts, typography, icons, and more"
    )
    action_registry.register(
        "ui_ux_stack", plugin.search_stack_guidelines,
        "Search stack-specific UI/UX guidelines (React Native, etc.)"
    )
    action_registry.register(
        "ui_ux_design_system", plugin.generate_design_system,
        "Generate a complete design system recommendation"
    )
    action_registry.register(
        "ui_ux_domains", plugin.list_domains,
        "List available UI/UX search domains and stacks"
    )

    logger.info(f"Registered UI/UX Pro Max plugin ({len(AVAILABLE_DOMAINS)} domains)")
    return plugin
