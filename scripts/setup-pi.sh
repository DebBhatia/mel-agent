#!/bin/bash
# ─────────────────────────────────────────────
# Mel Agent - Raspberry Pi Setup Script
# ─────────────────────────────────────────────
# Run this on your Raspberry Pi after cloning the repo:
#   chmod +x scripts/setup-pi.sh && ./scripts/setup-pi.sh
#
# Prerequisites:
#   - Raspberry Pi 4 (4GB+) or Pi 5 with Raspberry Pi OS 64-bit
#   - USB microphone plugged in
#   - Speaker connected (3.5mm or Bluetooth)
#   - Network connection to reach your Mel server
# ─────────────────────────────────────────────

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "================================================"
echo "  Mel Agent - Raspberry Pi Setup"
echo "  Repo: $REPO_DIR"
echo "================================================"

# 1. System packages for audio
echo ""
echo "[1/6] Installing system audio packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip python3-venv \
    portaudio19-dev \
    libsndfile1 \
    alsa-utils \
    mpg123 \
    ffmpeg

# 2. Python virtual environment
echo ""
echo "[2/6] Setting up Python virtual environment..."
cd "$REPO_DIR"
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
echo ""
echo "[3/6] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements-pi.txt -q

# 4. Install Piper TTS (local, no cloud needed)
echo ""
echo "[4/6] Installing Piper TTS..."
pip install piper-tts -q
# Download default voice model if not present
PIPER_MODEL_DIR="$HOME/.local/share/piper-voices"
mkdir -p "$PIPER_MODEL_DIR"
if [ ! -f "$PIPER_MODEL_DIR/en_US-lessac-medium.onnx" ]; then
    echo "Downloading Piper voice model..."
    curl -sL "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" \
        -o "$PIPER_MODEL_DIR/en_US-lessac-medium.onnx"
    curl -sL "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" \
        -o "$PIPER_MODEL_DIR/en_US-lessac-medium.onnx.json"
fi

# 5. Create .env if not exists
echo ""
echo "[5/6] Checking environment configuration..."
if [ ! -f "$REPO_DIR/.env" ]; then
    echo "Creating .env file — you need to fill in your values!"
    cat > "$REPO_DIR/.env" << 'ENVEOF'
# ── Mel Agent Pi Configuration ──

# REQUIRED: Your orchestrator server URL (your Windows laptop or AWS)
ORCHESTRATOR_URL=http://YOUR_SERVER_IP:8000

# REQUIRED: API key from your server's .env file
AGENT_API_KEY=mel-YOUR_KEY_HERE

# Wake word — must match the server's WAKE_PHRASE
WAKE_PHRASE=wake up daddy is home

# TTS engine: "piper" (local, free) or "elevenlabs" (cloud, better quality)
TTS_ENGINE=piper

# Whisper model: "tiny" (fastest), "base" (balanced), "small" (best quality)
WHISPER_MODEL=base

# Only needed if using ElevenLabs TTS
# ELEVENLABS_API_KEY=
# ELEVENLABS_VOICE_ID=
ENVEOF
    echo ""
    echo "  !! IMPORTANT: Edit .env with your server IP and API key !!"
    echo "  Run: nano $REPO_DIR/.env"
    echo ""
fi

# 6. Install systemd service
echo ""
echo "[6/6] Installing systemd service..."
sudo cp "$REPO_DIR/scripts/mel-listener.service" /etc/systemd/system/
sudo systemctl daemon-reload

echo ""
echo "================================================"
echo "  Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Edit your .env:  nano $REPO_DIR/.env"
echo "     - Set ORCHESTRATOR_URL to your server IP"
echo "     - Set AGENT_API_KEY from your server's .env"
echo ""
echo "  2. Test the mic:    arecord -d 3 test.wav && aplay test.wav"
echo ""
echo "  3. Test the listener:"
echo "     cd $REPO_DIR/src && ../venv/bin/python wake_listener.py"
echo ""
echo "  4. Enable auto-start on boot:"
echo "     sudo systemctl enable mel-listener"
echo "     sudo systemctl start mel-listener"
echo ""
echo "  5. Check logs:"
echo "     journalctl -u mel-listener -f"
echo ""
