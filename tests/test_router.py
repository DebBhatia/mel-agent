"""Tests for router.py — ModelRouter backend selection policy."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from router import ModelRouter, Backend


class TestModelRouter:
    def setup_method(self):
        self.router = ModelRouter()

    def test_defaults_to_local(self):
        assert self.router.choose("INFORMATION", {"requires_cloud": False}, "how's it going") == Backend.LOCAL

    def test_no_classification_defaults_to_local(self):
        assert self.router.choose("PERSONAL", None, "what's my name") == Backend.LOCAL

    def test_code_category_routes_to_claude(self):
        assert self.router.choose("CODE", {"requires_cloud": True}, "build me a website") == Backend.CLAUDE

    def test_devops_category_routes_to_claude(self):
        assert self.router.choose("DEVOPS", {"requires_cloud": False}, "restart the server") == Backend.CLAUDE

    def test_requires_cloud_flag_routes_to_claude_even_outside_complex_categories(self):
        assert self.router.choose("INFORMATION", {"requires_cloud": True}, "what's the meaning of life") == Backend.CLAUDE

    def test_openai_search_stub_category(self):
        assert self.router.choose("OPENAI_SEARCH", {}, "search for something") == Backend.OPENAI_SEARCH

    def test_reservation_with_requires_cloud_routes_to_claude(self):
        assert self.router.choose("RESERVATION", {"requires_cloud": True}, "book a table") == Backend.CLAUDE

    def test_communication_without_requires_cloud_routes_to_local(self):
        assert self.router.choose("COMMUNICATION", {"requires_cloud": False}, "text mom") == Backend.LOCAL
