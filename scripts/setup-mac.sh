#!/bin/bash
# ─────────────────────────────────────────────
# Mel Agent - macOS Setup Script (Mac mini / any Mac)
# ─────────────────────────────────────────────
# Run this on your Mac after cloning the repo:
#   chmod +x scripts/setup-mac.sh && ./scripts/setup-mac.sh
#
# This sets up EVERYTHING on one machine: the orchestrator
# server (server.py) and the voice wake-word listener
# (wake_listener.py), since a Mac mini is powerful enough
# to run both at once.
#
# Prerequisites:
#   - macOS with Homebrew installed (https://brew.sh)
#   - Built-in mic/speakers, or a USB mic/speaker
# ─────────────────────────────────────────────

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "================================================"
echo "  Mel Agent - macOS Setup"
echo "  Repo: $REPO_DIR"
echo "================================================"

# 0. Check for Homebrew
if ! command -v brew >/dev/null 2>&1; then
    echo ""
    echo "Homebrew not found. Install it first:"
    echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
fi

# 1. System packages (audio + whisper needs ffmpeg)
echo ""
echo "[1/7] Installing system packages via Homebrew..."
brew install portaudio ffmpeg python@3.11 >/dev/null

# 2. Python virtual environment
echo ""
echo "[2/7] Setting up Python virtual environment..."
cd "$REPO_DIR"
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
echo ""
echo "[3/7] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements-mac.txt -q

# 4. Install Piper TTS voice model (local, no cloud needed)
echo ""
echo "[4/7] Setting up Piper TTS voice model..."
PIPER_MODEL_DIR="$HOME/.local/share/piper-voices"
mkdir -p "$PIPER_MODEL_DIR"
if [ ! -f "$PIPER_MODEL_DIR/en_US-lessac-medium.onnx" ]; then
    echo "Downloading Piper voice model..."
    curl -sL "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" \
        -o "$PIPER_MODEL_DIR/en_US-lessac-medium.onnx"
    curl -sL "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" \
        -o "$PIPER_MODEL_DIR/en_US-lessac-medium.onnx.json"
fi

# 5. Check Ollama (local LLM used by the orchestrator for private/simple requests)
echo ""
echo "[5/7] Checking Ollama..."
if ! command -v ollama >/dev/null 2>&1; then
    echo "  Ollama not found. Installing via Homebrew..."
    brew install ollama >/dev/null
    echo "  Starting Ollama service..."
    brew services start ollama >/dev/null
    sleep 2
fi
if ! ollama list >/dev/null 2>&1; then
    brew services start ollama >/dev/null
    sleep 2
fi
echo "  Pulling llama3.1 model (this may take a few minutes)..."
ollama pull llama3.1 || echo "  !! Could not pull model automatically — run 'ollama pull llama3.1' manually."

# 6. Create .env if not exists
echo ""
echo "[6/7] Checking environment configuration..."
if [ ! -f "$REPO_DIR/.env" ]; then
    echo "Creating .env file — you need to fill in your values!"
    cat > "$REPO_DIR/.env" << 'ENVEOF'
# ── Mel Agent Configuration (macOS all-in-one) ──

# REQUIRED: Claude API key (heavy reasoning) — https://console.anthropic.com/
ANTHROPIC_API_KEY=

# Orchestrator server (running locally on this same Mac)
ORCHESTRATOR_URL=http://localhost:8000
AGENT_HOST=127.0.0.1
AGENT_PORT=8000

# REQUIRED: shared secret between the listener and the server.
# Generate one with: python -c "import secrets; print('mel-' + secrets.token_hex(16))"
AGENT_API_KEY=

# Local LLM via Ollama (private/simple requests)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

# REQUIRED: Picovoice access key for wake word detection (free)
# Get yours at https://console.picovoice.ai/
PICOVOICE_ACCESS_KEY=

# Custom wake word models (optional — train at https://console.picovoice.ai/)
# Without these, built-in "Jarvis" (command) and "Computer" (homecoming) are used
# PORCUPINE_KEYWORD_COMMAND=path/to/mel.ppn
# PORCUPINE_KEYWORD_HOMECOMING=path/to/wake-up-daddy-is-home.ppn

# TTS engine: "piper" (local, free) or "elevenlabs" (cloud, better quality)
TTS_ENGINE=piper

# Whisper model: "tiny" (fastest), "base" (balanced), "small" (best quality)
WHISPER_MODEL=base

# Only needed if using ElevenLabs TTS
# ELEVENLABS_API_KEY=
# ELEVENLABS_VOICE_ID=

USER_NAME=Deb
TIMEZONE=America/Chicago
ENVEOF
    echo ""
    echo "  !! IMPORTANT: Edit .env with your keys before starting !!"
    echo "  Run: nano $REPO_DIR/.env"
    echo ""
fi

# 7. Install launchd services (auto-start on login, no sudo needed)
echo ""
echo "[7/7] Installing launchd services..."
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs/mel-agent"

for svc in server listener; do
    TEMPLATE="$REPO_DIR/scripts/com.melagent.${svc}.plist.template"
    DEST="$HOME/Library/LaunchAgents/com.melagent.${svc}.plist"
    sed -e "s|__REPO_DIR__|$REPO_DIR|g" -e "s|__HOME__|$HOME|g" "$TEMPLATE" > "$DEST"
done

echo ""
echo "================================================"
echo "  Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Edit your .env:  nano $REPO_DIR/.env"
echo "     - Set ANTHROPIC_API_KEY"
echo "     - Set AGENT_API_KEY (generate with the command in the .env comments)"
echo "     - Set PICOVOICE_ACCESS_KEY (free at https://console.picovoice.ai/)"
echo ""
echo "  2. (Optional) Set up Google Calendar:"
echo "     ./scripts/setup-google-calendar.sh --service-account ~/Downloads/key.json"
echo ""
echo "  3. Test the mic (grant Terminal mic access if macOS prompts you):"
echo "     cd $REPO_DIR/src && ../venv/bin/python -c \"import pyaudio; print('mic OK, devices:', pyaudio.PyAudio().get_device_count())\""
echo ""
echo "  4. Try it manually first, before enabling auto-start:"
echo "     Terminal 1: cd $REPO_DIR/src && ../venv/bin/python server.py"
echo "     Terminal 2: cd $REPO_DIR/src && ../venv/bin/python wake_listener.py"
echo "     Say 'Jarvis' (command mode) or 'Computer' (homecoming mode) once both are running."
echo ""
echo "  5. Once it works, enable auto-start on login/boot:"
echo "     launchctl load ~/Library/LaunchAgents/com.melagent.server.plist"
echo "     launchctl load ~/Library/LaunchAgents/com.melagent.listener.plist"
echo ""
echo "  6. Check logs:"
echo "     tail -f ~/Library/Logs/mel-agent/server.log"
echo "     tail -f ~/Library/Logs/mel-agent/listener.log"
echo ""
echo "  7. To stop the auto-start services:"
echo "     launchctl unload ~/Library/LaunchAgents/com.melagent.server.plist"
echo "     launchctl unload ~/Library/LaunchAgents/com.melagent.listener.plist"
echo ""
