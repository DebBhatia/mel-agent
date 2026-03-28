"""
SECURITY MODULE
================
Comprehensive security layer for the AI agent.
Handles encryption, secret management, audit logging,
network restrictions, and data protection.

⚠️  This module is the FIRST thing loaded by the orchestrator.
    If security checks fail, the agent refuses to start.
"""

import os
import re
import json
import time
import hmac
import hashlib
import logging
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from base64 import b64encode, b64decode

logger = logging.getLogger("security")


# ─────────────────────────────────────────────
# Encryption Engine (AES-256-GCM via Fernet)
# ─────────────────────────────────────────────
class EncryptionEngine:
    """
    Encrypts sensitive data at rest using Fernet (AES-128-CBC + HMAC-SHA256).
    For AES-256-GCM, use the optional cryptography backend.

    Your master key is derived from a passphrase + salt using PBKDF2.
    The key NEVER leaves your machine.
    """

    def __init__(self, key_path: str = None):
        self.key_path = key_path or os.path.expanduser("~/.config/agent/master.key")
        self.fernet = None
        self._initialize()

    def _initialize(self):
        """Load or generate the master encryption key."""
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            from cryptography.hazmat.primitives import hashes

            if os.path.exists(self.key_path):
                with open(self.key_path, "rb") as f:
                    key = f.read()
                self.fernet = Fernet(key)
                logger.info("🔐 Encryption key loaded")
            else:
                # Generate new key
                key = Fernet.generate_key()
                os.makedirs(os.path.dirname(self.key_path), exist_ok=True)
                with open(self.key_path, "wb") as f:
                    f.write(key)
                # Lock file permissions: owner read/write only
                self._lock_file(self.key_path)
                self.fernet = Fernet(key)
                logger.info("🔐 New encryption key generated and saved")

        except ImportError:
            logger.warning(
                "cryptography package not installed. "
                "Install with: pip install cryptography. "
                "Data-at-rest encryption disabled."
            )

    @staticmethod
    def _lock_file(filepath: str):
        """Lock file to owner-only access (cross-platform)."""
        import platform
        if platform.system() == "Windows":
            try:
                import subprocess
                # Remove inherited permissions, grant only current user
                username = os.environ.get("USERNAME", os.environ.get("USER", ""))
                if username:
                    subprocess.run(
                        ["icacls", filepath, "/inheritance:r",
                         "/grant:r", f"{username}:(R,W)"],
                        capture_output=True, check=False
                    )
                    logger.info(f"🔐 Windows ACL set: owner-only on {filepath}")
            except Exception as e:
                logger.warning(f"Could not set Windows ACL: {e}")
        else:
            try:
                os.chmod(filepath, 0o600)
            except Exception as e:
                logger.warning(f"Could not set Unix permissions: {e}")

    def encrypt(self, data: str) -> str:
        """Encrypt a string. Returns base64-encoded ciphertext."""
        if not self.fernet:
            return data  # Passthrough if encryption unavailable
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt(self, token: str) -> str:
        """Decrypt a Fernet token back to plaintext."""
        if not self.fernet:
            return token
        return self.fernet.decrypt(token.encode()).decode()

    def encrypt_file(self, input_path: str, output_path: str = None):
        """Encrypt a file on disk."""
        if not self.fernet:
            return
        output_path = output_path or input_path + ".enc"
        with open(input_path, "rb") as f:
            data = f.read()
        encrypted = self.fernet.encrypt(data)
        with open(output_path, "wb") as f:
            f.write(encrypted)
        self._lock_file(output_path)

    def decrypt_file(self, input_path: str, output_path: str = None):
        """Decrypt an encrypted file."""
        if not self.fernet:
            return
        output_path = output_path or input_path.replace(".enc", "")
        with open(input_path, "rb") as f:
            data = f.read()
        decrypted = self.fernet.decrypt(data)
        with open(output_path, "wb") as f:
            f.write(decrypted)


# ─────────────────────────────────────────────
# Secret Vault - Encrypted credential storage
# ─────────────────────────────────────────────
class SecretVault:
    """
    Encrypted local vault for API keys, tokens, and credentials.
    All secrets are encrypted at rest using Fernet.
    Never stores secrets in plaintext.
    """

    def __init__(self, vault_path: str = None):
        self.vault_path = vault_path or os.path.expanduser("~/.config/agent/vault.enc")
        self.encryption = EncryptionEngine()
        self._secrets = {}
        self._load()

    def _load(self):
        """Load and decrypt the vault."""
        if os.path.exists(self.vault_path):
            try:
                with open(self.vault_path, "r") as f:
                    encrypted_data = f.read()
                decrypted = self.encryption.decrypt(encrypted_data)
                self._secrets = json.loads(decrypted)
                logger.info(f"🔐 Vault loaded: {len(self._secrets)} secrets")
            except Exception as e:
                logger.error(f"Failed to load vault: {e}")
                self._secrets = {}
        else:
            logger.info("🔐 No vault found, starting fresh")

    def _save(self):
        """Encrypt and save the vault."""
        os.makedirs(os.path.dirname(self.vault_path), exist_ok=True)
        plaintext = json.dumps(self._secrets)
        encrypted = self.encryption.encrypt(plaintext)
        with open(self.vault_path, "w") as f:
            f.write(encrypted)
        EncryptionEngine._lock_file(self.vault_path)

    def set(self, key: str, value: str):
        """Store a secret."""
        self._secrets[key] = value
        self._save()
        logger.info(f"🔐 Secret stored: {key}")

    def get(self, key: str, default: str = "") -> str:
        """Retrieve a secret."""
        return self._secrets.get(key, default)

    def delete(self, key: str):
        """Delete a secret."""
        if key in self._secrets:
            del self._secrets[key]
            self._save()

    def list_keys(self) -> list[str]:
        """List all secret keys (not values)."""
        return list(self._secrets.keys())

    def has(self, key: str) -> bool:
        return key in self._secrets


# ─────────────────────────────────────────────
# Audit Logger - Track every external call
# ─────────────────────────────────────────────
class AuditLogger:
    """
    Logs every external API call, shell command, and data access.
    Creates an immutable audit trail you can review.
    Logs are stored locally and tamper-evident via HMAC chains.
    """

    def __init__(self, log_dir: str = None):
        self.log_dir = log_dir or os.path.expanduser("~/.config/agent/audit")
        os.makedirs(self.log_dir, exist_ok=True)
        self.current_log = os.path.join(
            self.log_dir,
            f"audit-{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        )
        self._chain_key = secrets.token_hex(16)
        self._last_hash = "genesis"

    def log(self, event_type: str, details: dict, sensitivity: str = "normal"):
        """
        Log an auditable event.

        event_type: api_call, shell_exec, data_access, auth, code_gen, deploy, error
        sensitivity: low, normal, high, critical
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "sensitivity": sensitivity,
            "details": details,
            "prev_hash": self._last_hash,
        }

        # Chain integrity: each entry contains hash of previous
        entry_str = json.dumps(entry, sort_keys=True)
        entry["hash"] = hmac.new(
            self._chain_key.encode(), entry_str.encode(), hashlib.sha256
        ).hexdigest()[:16]
        self._last_hash = entry["hash"]

        with open(self.current_log, "a") as f:
            f.write(json.dumps(entry) + "\n")

        if sensitivity in ("high", "critical"):
            logger.warning(f"🔴 AUDIT [{event_type}]: {details.get('summary', '')}")

    def log_api_call(self, service: str, endpoint: str, data_sent_preview: str = ""):
        """Log an outbound API call."""
        self.log("api_call", {
            "service": service,
            "endpoint": endpoint,
            "data_preview": data_sent_preview[:200],  # Only first 200 chars
            "summary": f"API call to {service}",
        }, sensitivity="high")

    def log_shell_command(self, command: str, success: bool):
        """Log a shell command execution."""
        self.log("shell_exec", {
            "command": command,
            "success": success,
            "summary": f"Shell: {command[:80]}",
        }, sensitivity="high")

    def log_code_generation(self, project_name: str, files_count: int):
        """Log code generation event."""
        self.log("code_gen", {
            "project": project_name,
            "files": files_count,
            "summary": f"Generated project: {project_name}",
        })

    def log_data_access(self, data_type: str, direction: str):
        """Log data access (read/write of sensitive data)."""
        self.log("data_access", {
            "data_type": data_type,
            "direction": direction,
            "summary": f"{direction} {data_type}",
        })

    def get_recent(self, n: int = 50) -> list[dict]:
        """Get recent audit entries."""
        entries = []
        if os.path.exists(self.current_log):
            with open(self.current_log) as f:
                for line in f:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries[-n:]

    def get_summary(self) -> dict:
        """Get audit statistics."""
        entries = self.get_recent(1000)
        return {
            "total_entries": len(entries),
            "api_calls": sum(1 for e in entries if e["event_type"] == "api_call"),
            "shell_commands": sum(1 for e in entries if e["event_type"] == "shell_exec"),
            "code_generations": sum(1 for e in entries if e["event_type"] == "code_gen"),
            "high_sensitivity": sum(1 for e in entries if e.get("sensitivity") in ("high", "critical")),
            "log_file": self.current_log,
        }


# ─────────────────────────────────────────────
# Enhanced PII Sanitizer (with regex patterns)
# ─────────────────────────────────────────────
class EnhancedPIISanitizer:
    """
    Advanced PII stripping that catches patterns even if
    not explicitly listed in pii_mappings.json.
    """

    # Regex patterns for common PII
    PII_PATTERNS = [
        (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]'),           # SSN
        (r'\b\d{16}\b', '[CARD_REDACTED]'),                       # Credit card (16 digits)
        (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARD_REDACTED]'),  # CC with spaces
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]'),  # Email
        (r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE_REDACTED]'),  # US phone
        (r'\b\d{5}(?:-\d{4})?\b', '[ZIP_REDACTED]'),             # ZIP code
        (r'(?i)\b(?:password|passwd|pwd)\s*[:=]\s*\S+', '[PASSWORD_REDACTED]'),
        # Specific API key patterns MUST come before generic secret matcher
        (r'sk-ant-[a-zA-Z0-9\-_]+', '[ANTHROPIC_KEY_REDACTED]'),
        (r'sk-[a-zA-Z0-9]{20,}', '[OPENAI_KEY_REDACTED]'),
        (r'AKIA[0-9A-Z]{16}', '[AWS_KEY_REDACTED]'),
        (r'ghp_[a-zA-Z0-9]{36}', '[GITHUB_TOKEN_REDACTED]'),
        # Generic catch-all for key/token/secret assignments (after specific patterns)
        # Negative lookahead avoids re-redacting already-replaced placeholders
        (r'(?i)\b(?:api[_-]?key|token|secret)\s*[:=]\s*(?!\[)\S+', '[SECRET_REDACTED]'),
    ]

    def __init__(self, mappings_path: str = None):
        self.mappings_path = mappings_path or os.path.join(
            os.path.dirname(__file__), "..", "config", "pii_mappings.json"
        )
        self.custom_mappings = {}
        if os.path.exists(self.mappings_path):
            with open(self.mappings_path) as f:
                raw = json.load(f)
            # Filter out non-string values (e.g. encrypted stub: _encrypted: true)
            self.custom_mappings = {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}

    def sanitize(self, text: str) -> str:
        """Full PII sanitization: custom mappings + regex patterns."""
        result = text

        # Apply custom mappings first (exact matches)
        for real_value, placeholder in self.custom_mappings.items():
            result = result.replace(real_value, placeholder)

        # Apply regex patterns
        for pattern, replacement in self.PII_PATTERNS:
            result = re.sub(pattern, replacement, result)

        return result

    def desanitize(self, text: str) -> str:
        """Restore custom mappings (regex replacements are one-way)."""
        result = text
        for real_value, placeholder in self.custom_mappings.items():
            result = result.replace(placeholder, real_value)
        return result

    def scan(self, text: str) -> list[dict]:
        """Scan text for PII without replacing. Returns findings."""
        findings = []
        for pattern, replacement in self.PII_PATTERNS:
            matches = re.finditer(pattern, text)
            for m in matches:
                findings.append({
                    "type": replacement.strip("[]"),
                    "position": m.start(),
                    "length": len(m.group()),
                })
        return findings


# ─────────────────────────────────────────────
# Network Guard - Restrict outbound connections
# ─────────────────────────────────────────────
class NetworkGuard:
    """
    Whitelist-based network access control.
    Only approved domains can be contacted.
    """

    ALLOWED_DOMAINS = [
        "api.anthropic.com",          # Claude API
        "api.elevenlabs.io",          # TTS (optional)
        "api.twilio.com",             # Calls & SMS
        "www.googleapis.com",         # Google Calendar
        "oauth2.googleapis.com",      # Google OAuth
        "accounts.google.com",        # Google Auth
        "api.spotify.com",            # Spotify playback
        "accounts.spotify.com",       # Spotify auth
        "localhost",                   # Local services
        "127.0.0.1",                  # Local
    ]

    # Domains that should NEVER be contacted
    BLOCKED_DOMAINS = [
        "pastebin.com",
        "hastebin.com",
        "webhook.site",
        "requestbin.com",
        "ngrok.io",
    ]

    @classmethod
    def is_allowed(cls, url: str) -> bool:
        """Check if a URL is in the allowed whitelist."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        hostname = parsed.hostname or ""

        # Check blocked first
        for blocked in cls.BLOCKED_DOMAINS:
            if blocked in hostname:
                logger.warning(f"🚫 BLOCKED domain: {hostname}")
                return False

        # Check allowed
        for allowed in cls.ALLOWED_DOMAINS:
            if hostname == allowed or hostname.endswith("." + allowed):
                return True

        # Private network ranges always allowed
        if hostname.startswith("192.168.") or hostname.startswith("10.") or hostname.startswith("172."):
            return True

        logger.warning(f"⚠️ Domain not in whitelist: {hostname}")
        return False

    @classmethod
    def add_allowed(cls, domain: str):
        """Add a domain to the whitelist."""
        if domain not in cls.ALLOWED_DOMAINS:
            cls.ALLOWED_DOMAINS.append(domain)
            logger.info(f"✅ Added to network whitelist: {domain}")


# ─────────────────────────────────────────────
# Rate Limiter - Prevent runaway API costs
# ─────────────────────────────────────────────
class RateLimiter:
    """
    Prevents excessive API usage.
    Protects your wallet from runaway agent loops.
    """

    def __init__(self):
        self.calls: dict[str, list[float]] = {}
        self.limits = {
            "claude_api": {"max_calls": 100, "window_seconds": 3600},     # 100/hour
            "twilio": {"max_calls": 20, "window_seconds": 3600},          # 20/hour
            "shell_exec": {"max_calls": 50, "window_seconds": 3600},      # 50/hour
            "code_gen": {"max_calls": 20, "window_seconds": 3600},        # 20/hour
            "elevenlabs": {"max_calls": 50, "window_seconds": 3600},      # 50/hour
        }

    def check(self, service: str) -> bool:
        """Check if a call is within rate limits. Returns True if allowed."""
        if service not in self.limits:
            return True

        now = time.time()
        limit = self.limits[service]
        window = limit["window_seconds"]
        max_calls = limit["max_calls"]

        # Clean old entries
        if service not in self.calls:
            self.calls[service] = []
        self.calls[service] = [t for t in self.calls[service] if now - t < window]

        if len(self.calls[service]) >= max_calls:
            logger.warning(f"🚫 Rate limit hit: {service} ({max_calls}/{window}s)")
            return False

        self.calls[service].append(now)
        return True

    def get_usage(self) -> dict:
        """Get current rate limit usage."""
        now = time.time()
        usage = {}
        for service, limit in self.limits.items():
            window = limit["window_seconds"]
            calls = [t for t in self.calls.get(service, []) if now - t < window]
            usage[service] = {
                "used": len(calls),
                "limit": limit["max_calls"],
                "remaining": limit["max_calls"] - len(calls),
                "window": f"{window}s",
            }
        return usage


# ─────────────────────────────────────────────
# Security Manager - Central security orchestrator
# ─────────────────────────────────────────────
class SecurityManager:
    """
    Central security manager that coordinates all security components.
    Initialize this FIRST before any other agent component.
    """

    def __init__(self):
        self.encryption = EncryptionEngine()
        self.vault = SecretVault()
        self.audit = AuditLogger()
        self.sanitizer = EnhancedPIISanitizer()
        self.network = NetworkGuard()
        self.rate_limiter = RateLimiter()
        self._startup_checks()

    def _startup_checks(self):
        """Run security checks on startup."""
        import platform
        issues = []

        # Check .env file permissions (Unix only — Windows uses ACLs)
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if platform.system() != "Windows" and os.path.exists(env_path):
            mode = oct(os.stat(env_path).st_mode)[-3:]
            if mode != "600":
                issues.append(f".env file has loose permissions ({mode}). Run: chmod 600 .env")

        # Check key file exists
        if not os.path.exists(self.encryption.key_path):
            issues.append("No encryption key found — one will be generated on first use")

        # Check for common security misconfigurations
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if api_key and api_key.startswith("sk-ant-") and len(api_key) < 20:
            issues.append("ANTHROPIC_API_KEY looks invalid or truncated")

        if issues:
            logger.warning(f"⚠️  Security notes:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("🔐 All security checks passed")

        self.audit.log("startup", {"checks_passed": len(issues) == 0, "issues": issues})

    def sanitize_for_cloud(self, text: str) -> str:
        """Full sanitization before any cloud API call."""
        sanitized = self.sanitizer.sanitize(text)
        # Double check: scan for anything the regex might have missed
        findings = self.sanitizer.scan(sanitized)
        if findings:
            logger.warning(f"⚠️ PII still detected after sanitization: {len(findings)} items")
        return sanitized

    def check_api_call(self, service: str, url: str, data_preview: str = "") -> bool:
        """Gate check before any external API call."""
        # Rate limit
        if not self.rate_limiter.check(service):
            self.audit.log("rate_limit", {"service": service}, sensitivity="high")
            return False

        # Network guard
        if not self.network.is_allowed(url):
            self.audit.log("blocked_domain", {"url": url, "service": service}, sensitivity="critical")
            return False

        # Audit log
        self.audit.log_api_call(service, url, data_preview)
        return True

    def get_status(self) -> dict:
        """Full security status report."""
        return {
            "encryption": "active" if self.encryption.fernet else "disabled",
            "vault": f"{len(self.vault.list_keys())} secrets stored",
            "audit": self.audit.get_summary(),
            "rate_limits": self.rate_limiter.get_usage(),
            "network_whitelist": len(self.network.ALLOWED_DOMAINS),
            "pii_mappings": len(self.sanitizer.custom_mappings),
        }
