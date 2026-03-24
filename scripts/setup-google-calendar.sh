#!/bin/bash
# ═══════════════════════════════════════════════
# Google Calendar Setup for Mel Agent
# ═══════════════════════════════════════════════
#
# TWO OPTIONS (pick one):
#
# ── Option A: Service Account (RECOMMENDED — never expires) ──
#
# 1. Go to https://console.cloud.google.com
# 2. Create a project (or select existing)
# 3. Enable "Google Calendar API"
# 4. Go to Credentials → Create Credentials → Service Account
#    - Name it "mel-agent" or similar
#    - No roles needed → Done
# 5. Click the service account → Keys → Add Key → JSON → Download
# 6. Note the service account email (looks like mel-agent@project.iam.gserviceaccount.com)
# 7. In Google Calendar (web), go to Settings → your calendar → Share with specific people
#    - Add the service account email with "Make changes to events"
# 8. Run: ./scripts/setup-google-calendar.sh --service-account ~/Downloads/key.json
# 9. Add to your .env: GOOGLE_CALENDAR_ID=your-email@gmail.com
#
# ── Option B: OAuth2 (legacy — requires browser login, token can expire) ──
#
# 1. Same project → Credentials → Create Credentials → OAuth 2.0 Client ID
#    - Application type: Desktop app → Download JSON
# 2. Run: ./scripts/setup-google-calendar.sh ~/Downloads/credentials.json
#
# ═══════════════════════════════════════════════

set -e

CONFIG_DIR="$HOME/.config/agent"
SA_FILE="$CONFIG_DIR/google_service_account.json"
CREDS_FILE="$CONFIG_DIR/google_credentials.json"
TOKEN_FILE="$CONFIG_DIR/google_token.json"

mkdir -p "$CONFIG_DIR"
chmod 700 "$CONFIG_DIR"

# ── Status check (no args) ──
if [ -z "$1" ]; then
    echo "Usage:"
    echo "  $0 --service-account <path-to-service-account-key.json>  (recommended)"
    echo "  $0 <path-to-oauth-credentials.json>                      (legacy)"
    echo ""
    echo "Current status:"
    if [ -f "$SA_FILE" ]; then
        SA_EMAIL=$(python3 -c "import json; print(json.load(open('$SA_FILE')).get('client_email','?'))" 2>/dev/null)
        echo "  Service Account: ✅ Found ($SA_EMAIL)"
    else
        echo "  Service Account: ❌ Not found"
    fi
    if [ -f "$CREDS_FILE" ]; then
        echo "  OAuth Creds:     ✅ Found at $CREDS_FILE"
    else
        echo "  OAuth Creds:     ❌ Not found"
    fi
    if [ -f "$TOKEN_FILE" ]; then
        echo "  OAuth Token:     ✅ Found at $TOKEN_FILE"
    else
        echo "  OAuth Token:     ❌ Not found"
    fi
    exit 1
fi

# ── Service Account setup ──
if [ "$1" = "--service-account" ]; then
    if [ -z "$2" ] || [ ! -f "$2" ]; then
        echo "Error: Provide path to service account JSON key file"
        echo "Usage: $0 --service-account ~/Downloads/key.json"
        exit 1
    fi

    # Validate it's a service account JSON
    python3 -c "
import json, sys
data = json.load(open('$2'))
if 'client_email' not in data or 'private_key' not in data:
    print('Error: This does not look like a service account key file.')
    print('Expected fields: client_email, private_key')
    sys.exit(1)
print(f\"Service account: {data['client_email']}\")
" || exit 1

    cp "$2" "$SA_FILE"
    chmod 600 "$SA_FILE"

    SA_EMAIL=$(python3 -c "import json; print(json.load(open('$SA_FILE'))['client_email'])")

    echo ""
    echo "✅ Service account key installed to $SA_FILE"
    echo ""
    echo "IMPORTANT — Complete these steps:"
    echo ""
    echo "  1. Open Google Calendar → Settings → your calendar → Share with specific people"
    echo "  2. Add: $SA_EMAIL"
    echo "  3. Set permission: 'Make changes to events'"
    echo "  4. Add to your .env file:"
    echo "       GOOGLE_CALENDAR_ID=your-email@gmail.com"
    echo "       GOOGLE_SERVICE_ACCOUNT_PATH=$SA_FILE"
    echo "  5. Restart Mel server"
    echo ""
    echo "That's it — no browser login needed, no tokens to expire."
    exit 0
fi

# ── OAuth2 setup (legacy) ──
if [ ! -f "$1" ]; then
    echo "Error: File not found: $1"
    exit 1
fi

python3 -c "import json; json.load(open('$1'))" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Invalid JSON file"
    exit 1
fi

cp "$1" "$CREDS_FILE"
chmod 600 "$CREDS_FILE"

echo "✅ Google Calendar OAuth credentials installed to $CREDS_FILE"
echo ""
echo "Next steps:"
echo "  1. Restart Mel server: cd src && python server.py"
echo "  2. First calendar request will open a browser for OAuth consent"
echo "  3. After consent, a token is saved to $TOKEN_FILE"
echo ""
echo "NOTE: For zero-maintenance auth, consider using a service account instead:"
echo "  $0 --service-account <path-to-key.json>"
