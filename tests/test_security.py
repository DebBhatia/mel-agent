"""Tests for the security module — encryption, vault, audit, PII, network guard, rate limiter."""

import os
import sys
import json
import time
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from security import (
    EncryptionEngine,
    SecretVault,
    AuditLogger,
    EnhancedPIISanitizer,
    NetworkGuard,
    RateLimiter,
    SecurityManager,
)


# ── EncryptionEngine ──────────────────────────

def _crypto_available():
    try:
        from cryptography.fernet import Fernet
        return True
    except BaseException:
        return False


class TestEncryptionEngine:
    @pytest.mark.skipif(not _crypto_available(), reason="cryptography package not available in this env")
    def test_encrypt_decrypt_roundtrip(self, tmp_path):
        key_path = str(tmp_path / "test.key")
        engine = EncryptionEngine(key_path=key_path)
        plaintext = "Hello, secret world!"
        cipher = engine.encrypt(plaintext)
        assert cipher != plaintext
        assert engine.decrypt(cipher) == plaintext

    @pytest.mark.skipif(not _crypto_available(), reason="cryptography package not available in this env")
    def test_encrypt_empty_string(self, tmp_path):
        key_path = str(tmp_path / "test.key")
        engine = EncryptionEngine(key_path=key_path)
        assert engine.decrypt(engine.encrypt("")) == ""

    def test_passthrough_when_no_crypto(self):
        engine = EncryptionEngine.__new__(EncryptionEngine)
        engine.fernet = None
        assert engine.encrypt("raw") == "raw"
        assert engine.decrypt("raw") == "raw"

    @pytest.mark.skipif(not _crypto_available(), reason="cryptography package not available in this env")
    def test_key_file_created(self, tmp_path):
        key_path = str(tmp_path / "new.key")
        EncryptionEngine(key_path=key_path)
        assert os.path.exists(key_path)

    @pytest.mark.skipif(not _crypto_available(), reason="cryptography package not available in this env")
    def test_encrypt_file(self, tmp_path):
        key_path = str(tmp_path / "test.key")
        engine = EncryptionEngine(key_path=key_path)
        src = tmp_path / "plain.txt"
        src.write_text("file content here")
        engine.encrypt_file(str(src))
        enc_path = str(src) + ".enc"
        assert os.path.exists(enc_path)
        engine.decrypt_file(enc_path, str(tmp_path / "restored.txt"))
        assert (tmp_path / "restored.txt").read_text() == "file content here"


# ── SecretVault ───────────────────────────────

@pytest.mark.skipif(not _crypto_available(), reason="cryptography package not available in this env")
class TestSecretVault:
    def test_set_get_delete(self, tmp_path):
        vault = SecretVault(vault_path=str(tmp_path / "vault.enc"))
        vault.set("my_key", "my_value")
        assert vault.get("my_key") == "my_value"
        assert vault.has("my_key")
        assert "my_key" in vault.list_keys()
        vault.delete("my_key")
        assert not vault.has("my_key")
        assert vault.get("my_key", "default") == "default"

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "vault.enc")
        v1 = SecretVault(vault_path=path)
        v1.set("persist_test", "value123")
        v2 = SecretVault(vault_path=path)
        assert v2.get("persist_test") == "value123"


# ── AuditLogger ───────────────────────────────

class TestAuditLogger:
    def test_log_creates_file(self, tmp_path):
        audit = AuditLogger(log_dir=str(tmp_path))
        audit.log("test_event", {"key": "value"})
        assert os.path.exists(audit.current_log)

    def test_log_chain_integrity(self, tmp_path):
        audit = AuditLogger(log_dir=str(tmp_path))
        audit.log("event1", {"x": 1})
        audit.log("event2", {"x": 2})
        entries = audit.get_recent(10)
        assert len(entries) == 2
        assert entries[0]["prev_hash"] == "genesis"
        assert entries[1]["prev_hash"] == entries[0]["hash"]

    def test_log_api_call(self, tmp_path):
        audit = AuditLogger(log_dir=str(tmp_path))
        audit.log_api_call("claude", "https://api.anthropic.com/v1/messages", "test preview")
        entries = audit.get_recent()
        assert entries[-1]["event_type"] == "api_call"
        assert entries[-1]["sensitivity"] == "high"

    def test_log_shell_command(self, tmp_path):
        audit = AuditLogger(log_dir=str(tmp_path))
        audit.log_shell_command("ls -la", True)
        entries = audit.get_recent()
        assert entries[-1]["details"]["command"] == "ls -la"

    def test_log_code_generation(self, tmp_path):
        audit = AuditLogger(log_dir=str(tmp_path))
        audit.log_code_generation("my-project", 5)
        entries = audit.get_recent()
        assert entries[-1]["details"]["files"] == 5

    def test_get_summary(self, tmp_path):
        audit = AuditLogger(log_dir=str(tmp_path))
        audit.log_api_call("s", "u")
        audit.log_shell_command("ls", True)
        audit.log_code_generation("p", 1)
        summary = audit.get_summary()
        assert summary["api_calls"] == 1
        assert summary["shell_commands"] == 1
        assert summary["code_generations"] == 1


# ── EnhancedPIISanitizer ─────────────────────

class TestEnhancedPIISanitizer:
    def setup_method(self):
        self.sanitizer = EnhancedPIISanitizer(mappings_path="/nonexistent")

    def test_ssn_redacted(self):
        assert "[SSN_REDACTED]" in self.sanitizer.sanitize("My SSN is 123-45-6789")

    def test_credit_card_redacted(self):
        result = self.sanitizer.sanitize("Card: 4111111111111111")
        assert "[CARD_REDACTED]" in result

    def test_credit_card_with_spaces(self):
        result = self.sanitizer.sanitize("Card: 4111 1111 1111 1111")
        assert "[CARD_REDACTED]" in result

    def test_email_redacted(self):
        result = self.sanitizer.sanitize("Email me at user@example.com")
        assert "[EMAIL_REDACTED]" in result

    def test_phone_redacted(self):
        result = self.sanitizer.sanitize("Call 555-123-4567")
        assert "[PHONE_REDACTED]" in result

    def test_password_redacted(self):
        result = self.sanitizer.sanitize("password: s3cret123!")
        assert "[PASSWORD_REDACTED]" in result

    def test_anthropic_key_redacted(self):
        result = self.sanitizer.sanitize("key is sk-ant-api03-ABCDEF12345678")
        assert "[ANTHROPIC_KEY_REDACTED]" in result

    def test_openai_key_redacted(self):
        result = self.sanitizer.sanitize("key is sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890")
        assert "[OPENAI_KEY_REDACTED]" in result

    def test_aws_key_redacted(self):
        result = self.sanitizer.sanitize("key is AKIAIOSFODNN7EXAMPLE")
        assert "[AWS_KEY_REDACTED]" in result

    def test_github_token_redacted(self):
        result = self.sanitizer.sanitize("token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij")
        assert "[GITHUB_TOKEN_REDACTED]" in result

    def test_scan_returns_findings(self):
        findings = self.sanitizer.scan("SSN: 123-45-6789 and email: x@y.com")
        types = [f["type"] for f in findings]
        assert any("SSN" in t for t in types)
        assert any("EMAIL" in t for t in types)

    def test_clean_text_unchanged(self):
        text = "This is perfectly safe text with no PII."
        assert self.sanitizer.sanitize(text) == text

    def test_custom_mappings(self, tmp_path):
        mappings = {"John Doe": "[NAME]", "123 Main St": "[ADDRESS]"}
        p = tmp_path / "pii.json"
        p.write_text(json.dumps(mappings))
        s = EnhancedPIISanitizer(mappings_path=str(p))
        result = s.sanitize("Contact John Doe at 123 Main St")
        assert "[NAME]" in result
        assert "[ADDRESS]" in result

    def test_desanitize_restores_custom(self, tmp_path):
        mappings = {"John": "[NAME]"}
        p = tmp_path / "pii.json"
        p.write_text(json.dumps(mappings))
        s = EnhancedPIISanitizer(mappings_path=str(p))
        sanitized = s.sanitize("Hello John")
        assert "John" not in sanitized
        restored = s.desanitize(sanitized)
        assert "John" in restored


# ── NetworkGuard ──────────────────────────────

class TestNetworkGuard:
    def test_allowed_domains(self):
        assert NetworkGuard.is_allowed("https://api.anthropic.com/v1/messages")
        assert NetworkGuard.is_allowed("https://api.elevenlabs.io/v1/tts")
        assert NetworkGuard.is_allowed("https://api.twilio.com/send")
        assert NetworkGuard.is_allowed("https://www.googleapis.com/calendar")
        assert NetworkGuard.is_allowed("http://localhost:8000/health")
        assert NetworkGuard.is_allowed("http://127.0.0.1:8000")

    def test_blocked_domains(self):
        assert not NetworkGuard.is_allowed("https://pastebin.com/raw/abc")
        assert not NetworkGuard.is_allowed("https://hastebin.com/doc")
        assert not NetworkGuard.is_allowed("https://webhook.site/test")
        assert not NetworkGuard.is_allowed("https://requestbin.com/r/abc")
        assert not NetworkGuard.is_allowed("https://abc.ngrok.io/tunnel")

    def test_private_ranges_allowed(self):
        assert NetworkGuard.is_allowed("http://192.168.1.100:3000")
        assert NetworkGuard.is_allowed("http://10.0.0.5/api")
        assert NetworkGuard.is_allowed("http://172.16.0.1:80")

    def test_unknown_domain_blocked(self):
        assert not NetworkGuard.is_allowed("https://evil-site.com/steal")
        assert not NetworkGuard.is_allowed("https://random-unknown.org")

    def test_add_allowed(self):
        NetworkGuard.add_allowed("custom.example.com")
        assert NetworkGuard.is_allowed("https://custom.example.com/api")
        # Cleanup
        NetworkGuard.ALLOWED_DOMAINS.remove("custom.example.com")


# ── RateLimiter ───────────────────────────────

class TestRateLimiter:
    def test_within_limit(self):
        rl = RateLimiter()
        assert rl.check("claude_api") is True

    def test_exceeds_limit(self):
        rl = RateLimiter()
        rl.limits["test_svc"] = {"max_calls": 3, "window_seconds": 3600}
        assert rl.check("test_svc") is True
        assert rl.check("test_svc") is True
        assert rl.check("test_svc") is True
        assert rl.check("test_svc") is False  # 4th call blocked

    def test_unknown_service_allowed(self):
        rl = RateLimiter()
        assert rl.check("nonexistent_service") is True

    def test_get_usage(self):
        rl = RateLimiter()
        rl.check("claude_api")
        rl.check("claude_api")
        usage = rl.get_usage()
        assert usage["claude_api"]["used"] == 2
        assert usage["claude_api"]["remaining"] == 98

    def test_window_expiry(self):
        rl = RateLimiter()
        rl.limits["fast_svc"] = {"max_calls": 1, "window_seconds": 1}
        assert rl.check("fast_svc") is True
        assert rl.check("fast_svc") is False
        time.sleep(1.1)
        assert rl.check("fast_svc") is True
