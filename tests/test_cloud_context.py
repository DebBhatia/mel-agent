"""Tests for cloud_context.py — CloudContextPolicy sanitization/capping."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cloud_context import CloudContextPolicy, sanitize_cloud_context, DEFAULT_MAX_CHARS


class TestCloudContextPolicySanitize:
    def setup_method(self):
        self.policy = CloudContextPolicy()

    def test_redacts_email(self):
        text, removed = self.policy.sanitize("contact me at deb@example.com please")
        assert "deb@example.com" not in text
        assert "[EMAIL_REDACTED]" in text
        assert "EMAIL_REDACTED" in removed

    def test_redacts_phone(self):
        text, removed = self.policy.sanitize("call 555-123-4567 tomorrow")
        assert "555-123-4567" not in text
        assert "PHONE_REDACTED" in removed

    def test_redacts_api_key(self):
        text, removed = self.policy.sanitize("here's the key sk-ant-abc123def456ghi789")
        assert "sk-ant-abc123def456ghi789" not in text
        assert "ANTHROPIC_KEY_REDACTED" in removed

    def test_redacts_password_assignment(self):
        text, removed = self.policy.sanitize("password: hunter2")
        assert "hunter2" not in text
        assert "PASSWORD_REDACTED" in removed

    def test_clean_text_passes_through_unchanged(self):
        text, removed = self.policy.sanitize("what's the weather like today")
        assert text == "what's the weather like today"
        assert removed == []

    def test_empty_text_returns_empty(self):
        text, removed = self.policy.sanitize("")
        assert text == ""
        assert removed == []

    def test_does_not_mutate_original_string(self):
        original = "email me at deb@example.com"
        text, removed = self.policy.sanitize(original)
        assert original == "email me at deb@example.com"
        assert text != original

    def test_removed_categories_contain_no_raw_values(self):
        text, removed = self.policy.sanitize("email me at deb@example.com, call 555-123-4567")
        joined = " ".join(removed)
        assert "deb@example.com" not in joined
        assert "555-123-4567" not in joined


class TestCloudContextPolicySizeCap:
    def test_long_text_is_truncated(self):
        policy = CloudContextPolicy(max_chars=50)
        text, removed = policy.sanitize("x" * 500)
        assert len(text) <= 50 + len("...[TRUNCATED]")
        assert "TRUNCATED_FOR_SIZE" in removed

    def test_short_text_is_not_truncated(self):
        policy = CloudContextPolicy(max_chars=DEFAULT_MAX_CHARS)
        text, removed = policy.sanitize("a short message")
        assert text == "a short message"
        assert "TRUNCATED_FOR_SIZE" not in removed

    def test_default_max_chars_is_reasonable(self):
        assert DEFAULT_MAX_CHARS > 0


class TestPrepareCloudRequest:
    def test_builds_sanitized_payload(self):
        policy = CloudContextPolicy()
        payload = policy.prepare_cloud_request("call me at 555-123-4567", extra_context="notes")
        assert "555-123-4567" not in payload["message"]
        assert payload["extra_context"] == "notes"
        assert "PHONE_REDACTED" in payload["redacted_categories"]

    def test_empty_extra_context_stays_empty(self):
        policy = CloudContextPolicy()
        payload = policy.prepare_cloud_request("hello")
        assert payload["extra_context"] == ""
        assert payload["redacted_categories"] == []


class TestSanitizeCloudContextConvenience:
    def test_module_level_wrapper_matches_policy_behavior(self):
        text, removed = sanitize_cloud_context("email me at deb@example.com")
        assert "deb@example.com" not in text
        assert "EMAIL_REDACTED" in removed
