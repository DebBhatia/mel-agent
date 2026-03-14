#!/bin/bash
# ═══════════════════════════════════════════════
# Google Calendar Setup for Mel Agent
# ═══════════════════════════════════════════════
#
# Steps:
# 1. Go to https://console.cloud.google.com
# 2. Create a project (or select existing)
# 3. Enable "Google Calendar API"
# 4. Go to Credentials → Create Credentials → OAuth 2.0 Client ID
#    - Application type: Desktop app
#    - Download the JSON file
# 5. Run this script with the path to that JSON file:
#    ./scripts/setup-google-calendar.sh ~/Downloads/credentials.json
#
# ═══════════════════════════════════════════════

set -e

CONFIG_DIR="$HOME/.config/agent"
CREDS_FILE="$CONFIG_DIR/google_credentials.json"
TOKEN_FILE="$CONFIG_DIR/google_token.json"

mkdir -p "$CONFIG_DIR"
chmod 700 "$CONFIG_DIR"

if [ -z "$1" ]; then
    echo "Usage: $0 <path-to-google-credentials.json>"
    echo ""
    echo "Current status:"
    if [ -f "$CREDS_FILE" ]; then
        echo "  Credentials: ✅ Found at $CREDS_FILE"
    else
        echo "  Credentials: ❌ Not found"
    fi
    if [ -f "$TOKEN_FILE" ]; then
        echo "  Token:       ✅ Found at $TOKEN_FILE"
    else
        echo "  Token:       ❌ Not found (will be created on first auth)"
    fi
    exit 1
fi

if [ ! -f "$1" ]; then
    echo "Error: File not found: $1"
    exit 1
fi

# Validate it's JSON
python3 -c "import json; json.load(open('$1'))" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Invalid JSON file"
    exit 1
fi

cp "$1" "$CREDS_FILE"
chmod 600 "$CREDS_FILE"

echo "✅ Google Calendar credentials installed to $CREDS_FILE"
echo ""
echo "Next steps:"
echo "  1. Restart Mel server: cd src && python server.py"
echo "  2. First calendar request will open a browser for OAuth consent"
echo "  3. After consent, a token is saved to $TOKEN_FILE"
echo ""
echo "Environment variables (optional):"
echo "  export GOOGLE_CREDENTIALS_PATH=$CREDS_FILE"
echo "  export GOOGLE_TOKEN_PATH=$TOKEN_FILE"
