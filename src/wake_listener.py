"""
WAKE WORD LISTENER - Raspberry Pi
===================================
Listens for "wake up daddy is home" using OpenWakeWord,
then activates the agent and starts voice interaction.

Runs 24/7 on minimal resources. Only processes audio
after wake word is detected.
"""

import os
import json
import asyncio
import logging
import wave
import struct
import numpy as np
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("wake_listener")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
WAKE_PHRASE = os.getenv("WAKE_PHRASE", "wake up daddy is home")
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms at 16kHz
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 2.0  # seconds of silence = end of command
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")
TTS_ENGINE = os.getenv("TTS_ENGINE", "piper")  # piper (local) or elevenlabs (cloud)


class WakeWordDetector:
    """
    Detects custom wake word using OpenWakeWord.
    Runs entirely locally - no audio sent anywhere.
    """

    def __init__(self):
        self.model = None
        self.is_listening = True

    def initialize(self):
        """Load the wake word model."""
        try:
            from openwakeword.model import Model
            # Use pre-trained model or train custom one
            self.model = Model(
                wakeword_models=["hey_jarvis"],  # Closest built-in, or use custom
                inference_framework="onnx",
            )
            logger.info("Wake word model loaded successfully")
        except ImportError:
            logger.warning(
                "OpenWakeWord not installed. Install with: "
                "pip install openwakeword"
            )
            # Fallback: simple keyword detection via STT
            self.model = None

    def detect(self, audio_chunk: np.ndarray) -> bool:
        """Check if wake word was spoken. Returns True if detected."""
        if self.model is None:
            return False

        prediction = self.model.predict(audio_chunk)
        # Check all model scores
        for model_name, score in prediction.items():
            if score > 0.5:  # Confidence threshold
                logger.info(f"Wake word detected! (score: {score:.2f})")
                return True
        return False


class AudioCapture:
    """
    Captures audio from USB microphone on Raspberry Pi.
    Uses PyAudio for cross-platform compatibility.
    """

    def __init__(self):
        self.stream = None
        self.audio = None

    def start(self):
        """Initialize audio capture from USB mic."""
        try:
            import pyaudio
            self.audio = pyaudio.PyAudio()

            # Find USB microphone
            mic_index = self._find_usb_mic()

            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=mic_index,
                frames_per_buffer=CHUNK_SIZE,
            )
            logger.info(f"Audio capture started (device: {mic_index})")
        except Exception as e:
            logger.error(f"Audio capture failed: {e}")
            raise

    def _find_usb_mic(self) -> int:
        """Auto-detect USB microphone."""
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                name = info["name"].lower()
                if any(kw in name for kw in ["usb", "respeaker", "seeed", "minimic"]):
                    logger.info(f"Found USB mic: {info['name']} (index: {i})")
                    return i
        # Default to first input device
        logger.warning("No USB mic found, using default input device")
        return None

    def read_chunk(self) -> np.ndarray:
        """Read one audio chunk."""
        data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
        return np.frombuffer(data, dtype=np.int16)

    def stop(self):
        """Clean up audio resources."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()


class SpeechToText:
    """
    Converts speech to text using local Whisper model.
    All processing stays on your device.
    """

    def __init__(self):
        self.model = None

    def initialize(self):
        """Load Whisper model locally."""
        try:
            import whisper
            # Use 'tiny' or 'base' for Raspberry Pi, 'small' for more powerful hardware
            model_size = os.getenv("WHISPER_MODEL", "base")
            self.model = whisper.load_model(model_size)
            logger.info(f"Whisper {model_size} model loaded")
        except ImportError:
            logger.warning("Whisper not installed. Install: pip install openai-whisper")

    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio to text locally."""
        if self.model is None:
            return ""

        # Convert to float32 for Whisper
        audio_float = audio_data.astype(np.float32) / 32768.0
        result = self.model.transcribe(audio_float, language="en", fp16=False)
        return result["text"].strip()


class TextToSpeech:
    """
    Converts text to speech. Supports local (Piper) and cloud (ElevenLabs).
    """

    def __init__(self):
        self.engine = TTS_ENGINE

    async def speak(self, text: str):
        """Convert text to speech and play it."""
        if self.engine == "piper":
            await self._speak_piper(text)
        elif self.engine == "elevenlabs":
            await self._speak_elevenlabs(text)
        else:
            logger.warning(f"Unknown TTS engine: {self.engine}")

    async def _speak_piper(self, text: str):
        """Use Piper TTS (runs locally, no cloud needed)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "echo", text, "|",
                "piper", "--model", "en_US-lessac-medium.onnx", "--output-raw", "|",
                "aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()
        except Exception as e:
            logger.error(f"Piper TTS error: {e}")

    async def _speak_elevenlabs(self, text: str):
        """Use ElevenLabs API (cloud, better quality)."""
        import httpx
        api_key = os.getenv("ELEVENLABS_API_KEY", "")
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

        if not api_key:
            logger.warning("ElevenLabs API key not set")
            return

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={"xi-api-key": api_key},
                json={"text": text, "model_id": "eleven_turbo_v2"},
            )
            if response.status_code == 200:
                # Save and play audio
                audio_path = "/tmp/agent_response.mp3"
                with open(audio_path, "wb") as f:
                    f.write(response.content)
                proc = await asyncio.create_subprocess_exec(
                    "mpg123", audio_path,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()


class VoiceAgent:
    """
    Main voice agent loop for Raspberry Pi.
    Listens for wake word → captures command → processes → responds.
    """

    def __init__(self):
        self.wake_detector = WakeWordDetector()
        self.audio = AudioCapture()
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.is_active = False

    async def initialize(self):
        """Set up all components."""
        logger.info("Initializing Voice Agent...")
        self.wake_detector.initialize()
        self.stt.initialize()
        self.audio.start()
        logger.info("Voice Agent ready! Listening for wake word...")

    async def run(self):
        """Main 24/7 loop."""
        await self.initialize()

        while True:
            try:
                # Phase 1: Listen for wake word (low power)
                audio_chunk = self.audio.read_chunk()

                if not self.is_active:
                    if self.wake_detector.detect(audio_chunk):
                        self.is_active = True
                        await self.tts.speak("I'm here. What do you need?")
                        await self._process_command()
                        self.is_active = False

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)

        self.audio.stop()

    async def _process_command(self):
        """Capture full voice command after wake word."""
        logger.info("Listening for command...")
        audio_buffer = []
        silence_count = 0
        max_silence_chunks = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)

        while silence_count < max_silence_chunks:
            chunk = self.audio.read_chunk()
            audio_buffer.append(chunk)

            # Detect silence (end of speech)
            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
            if rms < SILENCE_THRESHOLD:
                silence_count += 1
            else:
                silence_count = 0

        # Transcribe locally
        full_audio = np.concatenate(audio_buffer)
        command = self.stt.transcribe(full_audio)
        logger.info(f"Command: {command}")

        if command:
            # Send to orchestrator
            response = await self._send_to_orchestrator(command)
            await self.tts.speak(response)

    async def _send_to_orchestrator(self, command: str) -> str:
        """Send command to the orchestrator API."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{ORCHESTRATOR_URL}/process",
                    json={"input": command},
                )
                return resp.json().get("response", "I couldn't process that.")
        except Exception as e:
            logger.error(f"Orchestrator connection failed: {e}")
            return "I'm having trouble connecting to my brain. Please try again."


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    agent = VoiceAgent()
    asyncio.run(agent.run())
