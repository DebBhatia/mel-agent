"""Tests for router.py — ModelRouter backend selection policy."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from router import ModelRouter, Backend, _is_complex_task


class TestModelRouter:
    def setup_method(self):
        self.router = ModelRouter()

    def test_defaults_to_local(self):
        assert self.router.choose("INFORMATION", {"requires_cloud": False}, "how's it going") == Backend.LOCAL

    def test_no_classification_defaults_to_local(self):
        assert self.router.choose("PERSONAL", None, "what's my name") == Backend.LOCAL

    def test_simple_code_request_stays_local(self):
        # Being CODE-categorized is no longer sufficient on its own --
        # a plain "build me a website" has no complexity signal.
        assert self.router.choose("CODE", {"requires_cloud": True}, "build me a website") == Backend.LOCAL

    def test_simple_devops_request_stays_local(self):
        assert self.router.choose("DEVOPS", {"requires_cloud": False}, "restart the server") == Backend.LOCAL

    def test_complex_code_request_routes_to_claude(self):
        assert self.router.choose(
            "CODE", {}, "refactor this module to remove the circular dependency"
        ) == Backend.CLAUDE

    def test_complex_devops_request_routes_to_claude(self):
        assert self.router.choose(
            "DEVOPS", {}, "help me redesign our CI/CD pipeline for faster deploys"
        ) == Backend.CLAUDE

    def test_requires_cloud_flag_alone_does_not_route_to_claude(self):
        # requires_cloud is an IntentClassifier guess, not a complexity signal --
        # it must not override the local-first default.
        assert self.router.choose("INFORMATION", {"requires_cloud": True}, "what's the meaning of life") == Backend.LOCAL

    def test_requires_cloud_does_not_substitute_for_complexity_on_code(self):
        # A CODE category with requires_cloud=True but no real complexity
        # signal in the text must still stay local.
        assert self.router.choose("CODE", {"requires_cloud": True}, "make me a simple todo app") == Backend.LOCAL

    def test_search_category_routes_to_openai_search(self):
        assert self.router.choose("SEARCH", {}, "what's happening in the news today") == Backend.OPENAI_SEARCH

    def test_openai_search_stub_category_still_supported(self):
        assert self.router.choose("OPENAI_SEARCH", {}, "search for something") == Backend.OPENAI_SEARCH

    def test_reservation_with_requires_cloud_still_routes_to_local(self):
        assert self.router.choose("RESERVATION", {"requires_cloud": True}, "book a table") == Backend.LOCAL

    def test_communication_without_requires_cloud_routes_to_local(self):
        assert self.router.choose("COMMUNICATION", {"requires_cloud": False}, "text mom") == Backend.LOCAL


class TestComplexityHeuristic:
    """Pure function, no router/LLM involvement -- deterministic keyword/regex only."""

    def test_empty_input_is_not_complex(self):
        assert _is_complex_task("") is False

    def test_none_like_input_is_not_complex(self):
        assert _is_complex_task(None) is False

    def test_plain_build_request_is_not_complex(self):
        assert _is_complex_task("build me a website for my dog walking business") is False

    def test_typo_fix_is_not_complex(self):
        assert _is_complex_task("fix this typo in the readme") is False

    def test_refactor_is_complex(self):
        assert _is_complex_task("refactor this service to use dependency injection") is True

    def test_architecture_is_complex(self):
        assert _is_complex_task("review the architecture of this microservice") is True

    def test_migration_is_complex(self):
        assert _is_complex_task("plan the database migration from postgres to cockroachdb") is True

    def test_concurrency_is_complex(self):
        assert _is_complex_task("there's a race condition in this concurrent worker pool") is True

    def test_distributed_system_is_complex(self):
        assert _is_complex_task("design a distributed system for this workload") is True

    def test_repo_wide_change_is_complex(self):
        assert _is_complex_task("make a repository-wide change to rename this package") is True

    def test_security_audit_is_complex(self):
        assert _is_complex_task("do a security audit of this auth module") is True

    def test_is_case_insensitive(self):
        assert _is_complex_task("REFACTOR THIS MODULE") is True
