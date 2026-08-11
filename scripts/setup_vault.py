"""
setup_vault.py
==============
Migrates plaintext secrets from .env into the encrypted SecretVault.
Run once: python scripts/setup_vault.py

After running:
  - Secrets are AES-encrypted in ~/.config/agent/vault.enc
  - Master key is owner-only at ~/.config/agent/master.key
  - .env is rewritten with secrets REMOVED (only non-sensitive config stays)
  - A temporary, owner-only .env.bak backup is created before rewriting and
    removed again once the rewrite succeeds (secrets live only in the vault
    afterward, never as a lingering plaintext copy on disk)
"""

import os
import sys
import shutil

# Add src/ to path so we can import security
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from security import SecretVault, EncryptionEngine

# ── Secrets to migrate from .env → vault ──────────────────────────────────────
# Keys that contain sensitive credentials and should NOT remain in .env plaintext
SECRET_KEYS = {
    "ANTHROPIC_API_KEY",
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_VOICE_ID",
    "AGENT_API_KEY",
    "GOOGLE_CALENDAR_ID",        # email address
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "TWILIO_PHONE_NUMBER",
    "SMTP_USER",
    "SMTP_PASS",
    "SPOTIFY_CLIENT_ID",
    "SPOTIFY_CLIENT_SECRET",
    "OPENWEATHER_API_KEY",
    "HA_TOKEN",
    "AGENT_ENCRYPTION_KEY",
    "GMAIL_USER_EMAIL",          # email address
    "NTFY_TOPIC",                # notification channel identifier
    "NTFY_TOKEN",                # optional ntfy auth token
    "PUSHOVER_USER_KEY",
    "PUSHOVER_API_TOKEN",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
}

ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
PII_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "pii_mappings.json")

def parse_env(path):
    """Parse .env file into list of (line, key, value) tuples."""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n")
            if stripped.startswith("#") or "=" not in stripped or stripped.startswith(" "):
                lines.append((stripped, None, None))
            else:
                key, _, value = stripped.partition("=")
                key = key.strip()
                value = value.strip()
                # Skip commented-out lines (start with #KEY)
                if key.startswith("#"):
                    lines.append((stripped, None, None))
                else:
                    lines.append((stripped, key, value))
    return lines

def main():
    if not os.path.exists(ENV_PATH):
        print(f"❌ .env not found at {ENV_PATH}")
        sys.exit(1)

    print("[ENC] Initializing encrypted vault...")
    vault = SecretVault()
    enc = vault.encryption

    # ── 1. Parse .env and store secrets in vault ──────────────────────────────
    parsed = parse_env(ENV_PATH)
    migrated = []
    for (original_line, key, value) in parsed:
        if key and key in SECRET_KEYS and value:
            vault.set(key, value)
            migrated.append(key)
            print(f"  [OK] Migrated: {key}")

    # ── 2. Backup original .env ───────────────────────────────────────────────
    backup_path = ENV_PATH + ".bak"
    shutil.copy2(ENV_PATH, backup_path)
    os.chmod(backup_path, 0o600)  # owner-only: backup still contains plaintext secrets
    print(f"\n[BAK] Backup saved (owner-only, temporary): {backup_path}")

    # ── 3. Rewrite .env — strip secrets, leave non-sensitive config ───────────
    new_lines = []
    for (original_line, key, value) in parsed:
        if key and key in SECRET_KEYS and value:
            # Replace value with a placeholder comment
            new_lines.append(f"# {key}=<encrypted — stored in vault>")
        else:
            new_lines.append(original_line)

    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")
    print(f"[OK] .env rewritten -- {len(migrated)} secrets removed from plaintext")

    # ── 3b. Remove the plaintext backup now that secrets are safely in the vault ──
    # The backup's only purpose was to guard against failure during steps 1-3 above;
    # keeping it around afterward would leave a permanent unencrypted copy of every
    # secret the vault was just used to protect.
    os.remove(backup_path)
    print(f"[OK] Temporary backup removed: {backup_path}")

    # ── 4. Encrypt pii_mappings.json ──────────────────────────────────────────
    if os.path.exists(PII_PATH) and enc.fernet:
        enc_pii_path = PII_PATH + ".enc"
        enc.encrypt_file(PII_PATH, enc_pii_path)
        # Overwrite original with a notice
        with open(PII_PATH, "w") as f:
            f.write('{"_encrypted": true, "_note": "PII data is in pii_mappings.json.enc"}\n')
        print(f"[OK] PII mappings encrypted: {enc_pii_path}")
    else:
        print("[INFO] pii_mappings.json not found or encryption unavailable -- skipped")

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print(f"""
VAULT MIGRATION COMPLETE
  Encrypted vault : ~/.config/agent/vault.enc
  Master key      : ~/.config/agent/master.key (owner-only)
  Secrets migrated: {len(migrated)}

IMPORTANT: Restart the server after running this script.
The master key at ~/.config/agent/master.key is the ONLY
way to decrypt your secrets. Back it up securely.
""")

if __name__ == "__main__":
    main()
