"""
MEL AI AGENT - Cloud Context Policy
=====================================
Smallest practical abstraction for minimizing what leaves the machine on
any non-local (Claude/OpenAI) call. Reuses security.py's existing
EnhancedPIISanitizer regex patterns rather than reimplementing redaction --
this module only adds size capping and a no-raw-values-logged removal
summary on top of it.

Not a security framework: no perfect PII detection is claimed. This
covers the patterns EnhancedPIISanitizer already covers (email, phone,
SSN, credit card, ZIP, password, common API key/token shapes) plus a
context-size cap so a runaway prompt can't be forwarded wholesale.
"""

import logging

logger = logging.getLogger("cloud_context")

DEFAULT_MAX_CHARS = 4000


class CloudContextPolicy:
    """Sanitizes and caps text before it is sent to any cloud provider."""

    def __init__(self, max_chars: int = DEFAULT_MAX_CHARS):
        self.max_chars = max_chars
        try:
            from security import EnhancedPIISanitizer
            self._sanitizer = EnhancedPIISanitizer()
        except Exception:
            self._sanitizer = None

    def sanitize(self, text: str) -> tuple[str, list[str]]:
        """Returns (sanitized_text, removed_categories). removed_categories
        lists which kinds of things were redacted (e.g. "EMAIL_REDACTED"),
        never the raw values themselves -- safe to log."""
        if not text:
            return text, []

        removed: list[str] = []
        result = text

        if self._sanitizer is not None:
            findings = self._sanitizer.scan(result)
            removed = sorted({f["type"] for f in findings})
            result = self._sanitizer.sanitize(result)

        truncated = len(result) > self.max_chars
        if truncated:
            result = result[: self.max_chars] + "...[TRUNCATED]"
            removed.append("TRUNCATED_FOR_SIZE")

        return result, removed

    def prepare_cloud_request(self, message: str, extra_context: str = "") -> dict:
        """Builds a cloud-safe payload from task-relevant fields only.
        Callers should pass narrowly-scoped strings (a single task's
        message/extra_context), never a full local history array or
        stored profile/email/calendar object -- this function sanitizes
        and caps what it's given, it doesn't decide what's relevant."""
        safe_message, removed_msg = self.sanitize(message)
        safe_context, removed_ctx = self.sanitize(extra_context) if extra_context else ("", [])

        removed = sorted(set(removed_msg) | set(removed_ctx))
        if removed:
            logger.info(f"Cloud request sanitized, redacted categories: {removed}")

        return {
            "message": safe_message,
            "extra_context": safe_context,
            "redacted_categories": removed,
        }


def sanitize_cloud_context(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> tuple[str, list[str]]:
    """Module-level convenience wrapper -- creates a policy instance so
    callers that just want a quick sanitize don't need to build one."""
    return CloudContextPolicy(max_chars=max_chars).sanitize(text)
