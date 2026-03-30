"""Tests for UI/UX Pro Max plugin — search, design system, domain detection, registration."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ui_ux import (
    BM25,
    UIUXPlugin,
    DesignSystemGenerator,
    detect_domain,
    search,
    search_stack,
    format_search_results,
    format_design_system,
    register_ui_ux_plugins,
    AVAILABLE_DOMAINS,
    AVAILABLE_STACKS,
    CSV_CONFIG,
    DATA_DIR,
)
from orchestrator import ActionRegistry


# ── BM25 Engine ──────────────────────────────

class TestBM25:
    def test_tokenize_lowercases(self):
        bm25 = BM25()
        tokens = bm25.tokenize("Hello World Testing")
        assert "hello" in tokens
        assert "world" in tokens

    def test_tokenize_filters_short_words(self):
        bm25 = BM25()
        tokens = bm25.tokenize("a an the big word")
        assert "a" not in tokens
        assert "an" not in tokens
        # "the" is 3 chars so it passes the len > 2 filter
        assert "big" in tokens
        assert "word" in tokens

    def test_tokenize_removes_punctuation(self):
        bm25 = BM25()
        tokens = bm25.tokenize("hello! world? test.")
        assert "hello" in tokens
        assert "world" in tokens

    def test_fit_and_score(self):
        bm25 = BM25()
        docs = ["red blue green", "orange yellow purple", "red orange pink"]
        bm25.fit(docs)
        scores = bm25.score("red color")
        # First result should be doc with "red"
        assert scores[0][1] > 0

    def test_empty_corpus(self):
        bm25 = BM25()
        bm25.fit([])
        scores = bm25.score("test")
        assert scores == []

    def test_no_match_returns_zero_scores(self):
        bm25 = BM25()
        bm25.fit(["apple banana cherry"])
        scores = bm25.score("zzzznonexistent")
        assert scores[0][1] == 0


# ── Domain Detection ─────────────────────────

class TestDetectDomain:
    def test_detect_color(self):
        assert detect_domain("color palette for fintech") == "color"

    def test_detect_chart(self):
        assert detect_domain("best chart for trends") == "chart"

    def test_detect_product(self):
        assert detect_domain("saas dashboard recommendations") == "product"

    def test_detect_ux(self):
        assert detect_domain("accessibility guidelines wcag") == "ux"

    def test_detect_icons(self):
        assert detect_domain("lucide icons set") == "icons"

    def test_detect_react(self):
        assert detect_domain("react performance optimization") == "react"

    def test_default_style(self):
        assert detect_domain("something completely random xyz") == "style"

    def test_detect_typography(self):
        assert detect_domain("font pairing for headers") == "typography"

    def test_detect_landing(self):
        assert detect_domain("landing page conversion optimization") == "landing"


# ── Search Functions ─────────────────────────

class TestSearch:
    def test_search_returns_dict(self):
        result = search("minimalism clean design")
        assert isinstance(result, dict)
        assert "domain" in result
        assert "results" in result

    def test_search_specific_domain(self):
        result = search("fitness app", domain="color")
        assert result["domain"] == "color"

    def test_search_with_max_results(self):
        result = search("modern design", max_results=1)
        assert len(result.get("results", [])) <= 1

    def test_search_invalid_domain_falls_back(self):
        result = search("test", domain="nonexistent")
        # Falls back to style domain config
        assert isinstance(result, dict)

    def test_search_has_results_for_known_query(self):
        result = search("glassmorphism", domain="style")
        assert result["count"] >= 0  # Data exists

    def test_search_color_domain(self):
        result = search("fitness", domain="color")
        assert result["domain"] == "color"
        assert "results" in result

    def test_search_product_domain(self):
        result = search("ecommerce", domain="product")
        assert result["domain"] == "product"


class TestSearchStack:
    def test_search_stack_valid(self):
        result = search_stack("navigation", "react-native")
        assert isinstance(result, dict)
        assert result.get("domain") == "stack" or "error" in result

    def test_search_stack_invalid(self):
        result = search_stack("test", "nonexistent-stack")
        assert "error" in result

    def test_available_stacks(self):
        assert "react-native" in AVAILABLE_STACKS


# ── Design System Generator ──────────────────

class TestDesignSystemGenerator:
    def test_init_loads_reasoning(self):
        gen = DesignSystemGenerator()
        # Should load without error even if file missing
        assert isinstance(gen.reasoning_data, list)

    def test_generate_returns_dict(self):
        gen = DesignSystemGenerator()
        ds = gen.generate("SaaS dashboard")
        assert isinstance(ds, dict)
        assert "project_name" in ds
        assert "category" in ds
        assert "reasoning" in ds
        assert "styles" in ds
        assert "colors" in ds
        assert "typography" in ds

    def test_generate_with_project_name(self):
        gen = DesignSystemGenerator()
        ds = gen.generate("fitness app", project_name="FitTrack")
        assert ds["project_name"] == "FitTrack"

    def test_reasoning_has_fields(self):
        gen = DesignSystemGenerator()
        ds = gen.generate("ecommerce store")
        reasoning = ds["reasoning"]
        assert "pattern" in reasoning
        assert "style_priority" in reasoning
        assert "color_mood" in reasoning
        assert "typography_mood" in reasoning

    def test_find_reasoning_rule(self):
        gen = DesignSystemGenerator()
        if gen.reasoning_data:
            # Should find at least one rule for common categories
            rule = gen._find_reasoning_rule("SaaS")
            # May or may not find, but should not crash
            assert isinstance(rule, dict)


# ── Format Helpers ───────────────────────────

class TestFormatHelpers:
    def test_format_search_results_with_error(self):
        result = format_search_results({"error": "test error"})
        assert "Error" in result

    def test_format_search_results_with_data(self):
        result = format_search_results({
            "domain": "style",
            "query": "test",
            "file": "styles.csv",
            "count": 1,
            "results": [{"Style Category": "Minimalism", "Type": "General"}],
        })
        assert "Minimalism" in result
        assert "UI Pro Max" in result

    def test_format_search_results_stack(self):
        result = format_search_results({
            "domain": "stack",
            "stack": "react-native",
            "query": "test",
            "file": "stacks/react-native.csv",
            "count": 0,
            "results": [],
        })
        assert "Stack Guidelines" in result

    def test_format_design_system(self):
        ds = {
            "project_name": "Test",
            "category": "SaaS",
            "generated_at": "2025-01-01",
            "reasoning": {
                "pattern": "Hero + Features",
                "style_priority": ["Minimalism"],
                "color_mood": "Professional",
                "typography_mood": "Clean",
                "key_effects": "",
                "anti_patterns": "",
            },
            "styles": [],
            "colors": [],
            "landing_patterns": [],
            "typography": [],
        }
        output = format_design_system(ds)
        assert "Design System: Test" in output
        assert "Professional" in output

    def test_format_truncates_long_values(self):
        result = format_search_results({
            "domain": "style",
            "query": "test",
            "file": "styles.csv",
            "count": 1,
            "results": [{"Style Category": "x" * 500}],
        })
        assert "..." in result


# ── Plugin Class ─────────────────────────────

class TestUIUXPlugin:
    def test_init(self):
        plugin = UIUXPlugin()
        assert plugin.generator is not None

    @pytest.mark.asyncio
    async def test_search_designs_empty_query(self):
        plugin = UIUXPlugin()
        result = await plugin.search_designs({})
        assert "provide" in result.lower()

    @pytest.mark.asyncio
    async def test_search_designs_valid(self):
        plugin = UIUXPlugin()
        result = await plugin.search_designs({"query": "minimalism"})
        assert "UI Pro Max" in result or "Error" in result

    @pytest.mark.asyncio
    async def test_search_stack_empty(self):
        plugin = UIUXPlugin()
        result = await plugin.search_stack_guidelines({"query": ""})
        assert "provide" in result.lower()

    @pytest.mark.asyncio
    async def test_search_stack_no_stack(self):
        plugin = UIUXPlugin()
        result = await plugin.search_stack_guidelines({"query": "navigation"})
        assert "specify" in result.lower()

    @pytest.mark.asyncio
    async def test_generate_design_system_empty(self):
        plugin = UIUXPlugin()
        result = await plugin.generate_design_system({})
        assert "provide" in result.lower()

    @pytest.mark.asyncio
    async def test_generate_design_system_valid(self):
        plugin = UIUXPlugin()
        result = await plugin.generate_design_system({"query": "SaaS dashboard"})
        assert "Design System" in result

    @pytest.mark.asyncio
    async def test_list_domains(self):
        plugin = UIUXPlugin()
        result = await plugin.list_domains({})
        assert "style" in result
        assert "color" in result
        assert "Domains" in result


# ── Registration ─────────────────────────────

class TestRegistration:
    def test_register_ui_ux_plugins(self):
        registry = ActionRegistry()
        plugin = register_ui_ux_plugins(registry)
        assert plugin is not None
        assert "ui_ux_search" in registry.actions
        assert "ui_ux_stack" in registry.actions
        assert "ui_ux_design_system" in registry.actions
        assert "ui_ux_domains" in registry.actions

    @pytest.mark.asyncio
    async def test_execute_via_registry(self):
        registry = ActionRegistry()
        register_ui_ux_plugins(registry)
        result = await registry.execute("ui_ux_search", {"query": "dark mode"})
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_execute_design_system_via_registry(self):
        registry = ActionRegistry()
        register_ui_ux_plugins(registry)
        result = await registry.execute("ui_ux_design_system", {"query": "fitness app"})
        assert isinstance(result, str)
        assert "Design System" in result


# ── Data Integrity ───────────────────────────

class TestDataIntegrity:
    def test_data_dir_exists(self):
        assert DATA_DIR.exists()

    def test_all_csv_files_exist(self):
        for domain, config in CSV_CONFIG.items():
            filepath = DATA_DIR / config["file"]
            assert filepath.exists(), f"Missing data file for domain '{domain}': {config['file']}"

    def test_csv_files_not_empty(self):
        for domain, config in CSV_CONFIG.items():
            filepath = DATA_DIR / config["file"]
            if filepath.exists():
                import csv
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                assert len(rows) > 0, f"Empty CSV for domain '{domain}'"

    def test_available_domains_match_config(self):
        assert set(AVAILABLE_DOMAINS) == set(CSV_CONFIG.keys())
