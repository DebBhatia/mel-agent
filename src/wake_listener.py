"""
WAKE WORD LISTENER - Raspberry Pi / macOS
===========================================
Listens for a wake word, then activates the agent and starts voice
interaction. Tries Porcupine (Picovoice) first, falls back to
openWakeWord (free, offline, no account needed) if Picovoice isn't
configured, and falls back further to keyboard/text mode if neither
wake word engine is available.

Runs 24/7 on minimal resources. Only processes audio
after wake word is detected. Works on Raspberry Pi (Linux/ALSA)
and macOS (via afplay) without changes — TTS playback and
microphone capture are both platform-aware.

Security: API key required to talk to orchestrator.
No voice data or commands are logged.
"""

from __future__ import annotations

import os
import re
import json
import asyncio
import logging
import platform
import tempfile
import wave
from datetime import datetime
from enum import Enum
import numpy as np
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("wake_listener")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms at 16kHz
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 2.0  # seconds of silence = end of command
MAX_COMMAND_DURATION = 30.0  # max seconds to record a single command
# Phrases (matched as a substring, case-insensitive) that end a conversation
# session and put the listener back to sleep, waiting for the wake word again.
SLEEP_PHRASES = ("goodbye mel", "good bye mel", "bye mel", "bye, mel")
# Consecutive empty (no speech heard) turns in a conversation session before
# auto-sleeping, in case the user walked away without saying a sleep phrase.
MAX_CONSECUTIVE_SILENT_TURNS = 3
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "")
TTS_ENGINE = os.getenv("TTS_ENGINE", "piper")  # piper (local) or elevenlabs (cloud)
PIPER_MODEL = os.getenv("PIPER_MODEL", "en_US-lessac-medium.onnx")
USER_NAME = os.getenv("USER_NAME", "Deb")
TIMEZONE = os.getenv("TIMEZONE", "America/Chicago")
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY", "")
# Paths to custom .ppn wake word model files (train at https://console.picovoice.ai/)
PORCUPINE_KEYWORD_COMMAND = os.getenv("PORCUPINE_KEYWORD_COMMAND", "")  # custom "Mel" .ppn file
PORCUPINE_KEYWORD_HOMECOMING = os.getenv("PORCUPINE_KEYWORD_HOMECOMING", "")  # custom homecoming .ppn file

# openWakeWord — free, offline, no account required. Used automatically when
# Picovoice isn't configured. Set WAKE_ENGINE=openwakeword to force it even if
# a Picovoice key is present, or WAKE_ENGINE=porcupine to disable this fallback.
WAKE_ENGINE = os.getenv("WAKE_ENGINE", "").strip().lower()
# Pretrained model name (or path to a custom-trained .onnx/.tflite model) per mode.
# There's no stock "computer" model, so homecoming defaults to "alexa" — train a
# custom model at https://github.com/dscripka/openWakeWord to use a different phrase.
OWW_KEYWORD_COMMAND = os.getenv("OWW_KEYWORD_COMMAND", "hey_jarvis")
OWW_KEYWORD_HOMECOMING = os.getenv("OWW_KEYWORD_HOMECOMING", "alexa")
OWW_THRESHOLD = float(os.getenv("OWW_THRESHOLD", "0.5"))

_SENTENCE_END_RE = re.compile(r'[.!?](?:\s|$)')


def _split_ready_sentences(buffer: str) -> tuple[str, str]:
    """
    Split buffer at the last complete sentence boundary.

    Returns (ready_to_speak, remainder) -- ready_to_speak holds zero or more
    complete sentences, remainder is the trailing incomplete one still
    waiting for more streamed tokens.
    """
    last_end = 0
    for m in _SENTENCE_END_RE.finditer(buffer):
        last_end = m.end()
    return buffer[:last_end], buffer[last_end:]


class WakeMode(Enum):
    """Which wake word was spoken determines the interaction mode."""
    HOMECOMING = "homecoming"  # "wake up daddy is home" → greeting + calendar
    COMMAND = "command"        # "Mel" → ready for any command


class WakeWordDetector:
    """
    Detects wake words. Tries Porcupine (Picovoice) first, then
    openWakeWord (free, offline, no account needed), then falls back
    to keyboard mode if neither engine is available.

    Supports two wake words:
      1. Command wake word (e.g. "Mel") → standard command mode
      2. Homecoming wake word → greeting + calendar briefing

    Porcupine: train custom wake words at https://console.picovoice.ai/
    and set PICOVOICE_ACCESS_KEY and PORCUPINE_KEYWORD_* in .env.

    openWakeWord: set OWW_KEYWORD_COMMAND / OWW_KEYWORD_HOMECOMING to a
    pretrained model name or a path to a custom-trained model. Used
    automatically when Picovoice isn't configured.

    Runs entirely locally — no audio sent anywhere.
    """

    def __init__(self):
        self.porcupine = None
        self.keyword_mode_map: list[WakeMode] = []
        self.oww_model = None
        self.oww_mode_map: dict = {}
        self.is_listening = True
        self.use_keyboard_fallback = False
        self._keyboard_triggered = None

    def initialize(self):
        """Load a wake word engine, trying each tier in order."""
        if WAKE_ENGINE != "openwakeword" and self._try_porcupine():
            return
        if WAKE_ENGINE != "porcupine" and self._try_openwakeword():
            return
        self._enable_keyboard_fallback()

    def _try_porcupine(self) -> bool:
        """Attempt to initialize Porcupine. Returns True on success."""
        try:
            import pvporcupine
        except ImportError:
            logger.info("pvporcupine not installed — trying next wake word engine")
            return False

        if not PICOVOICE_ACCESS_KEY:
            logger.info(
                "PICOVOICE_ACCESS_KEY not set — trying next wake word engine "
                "(get a free key at https://console.picovoice.ai/)"
            )
            return False

        try:
            # Build keyword lists — custom .ppn files or built-in fallbacks
            keyword_paths = []
            keywords = []
            self.keyword_mode_map = []

            if PORCUPINE_KEYWORD_COMMAND and Path(PORCUPINE_KEYWORD_COMMAND).exists():
                keyword_paths.append(PORCUPINE_KEYWORD_COMMAND)
                self.keyword_mode_map.append(WakeMode.COMMAND)
                logger.info(f"Custom command wake word: {PORCUPINE_KEYWORD_COMMAND}")
            else:
                keywords.append("jarvis")
                self.keyword_mode_map.append(WakeMode.COMMAND)
                logger.info("Using built-in 'Jarvis' as command wake word")

            if PORCUPINE_KEYWORD_HOMECOMING and Path(PORCUPINE_KEYWORD_HOMECOMING).exists():
                keyword_paths.append(PORCUPINE_KEYWORD_HOMECOMING)
                self.keyword_mode_map.append(WakeMode.HOMECOMING)
                logger.info(f"Custom homecoming wake word: {PORCUPINE_KEYWORD_HOMECOMING}")
            else:
                keywords.append("computer")
                self.keyword_mode_map.append(WakeMode.HOMECOMING)
                logger.info("Using built-in 'Computer' as homecoming wake word")

            create_kwargs = {"access_key": PICOVOICE_ACCESS_KEY}
            if keyword_paths:
                create_kwargs["keyword_paths"] = keyword_paths
            if keywords:
                create_kwargs["keywords"] = keywords

            self.porcupine = pvporcupine.create(**create_kwargs)
            logger.info("Porcupine wake word engine initialized")
            return True

        except Exception as e:
            logger.error(f"Porcupine init failed: {e}")
            return False

    def _try_openwakeword(self) -> bool:
        """Attempt to initialize openWakeWord. Returns True on success."""
        try:
            from openwakeword.model import Model
            from openwakeword.utils import download_models
        except ImportError:
            logger.info(
                "openwakeword not installed — trying next wake word engine. "
                "Install with: pip install openwakeword"
            )
            return False

        try:
            keyword_config = {
                OWW_KEYWORD_COMMAND: WakeMode.COMMAND,
                OWW_KEYWORD_HOMECOMING: WakeMode.HOMECOMING,
            }
            if OWW_KEYWORD_HOMECOMING == "alexa":
                logger.info(
                    "No stock 'computer' openWakeWord model exists — using 'alexa' "
                    "for homecoming mode. Train a custom model at "
                    "https://github.com/dscripka/openWakeWord and set "
                    "OWW_KEYWORD_HOMECOMING to its path for a different phrase."
                )

            # Only fetch pretrained models by name — custom paths are the user's own files.
            names_to_download = [k for k in keyword_config if not Path(k).exists()]
            if names_to_download:
                download_models(names_to_download)

            try:
                import tflite_runtime  # noqa: F401
                framework = "tflite"
            except ImportError:
                framework = "onnx"

            self.oww_model = Model(
                wakeword_models=list(keyword_config.keys()),
                inference_framework=framework,
            )
            # openWakeWord keys predictions by basename-without-extension for custom
            # model paths, but by the literal name for pretrained models -- map our
            # config keys to whatever key the Model instance actually produces.
            self.oww_mode_map = {
                (Path(k).stem if Path(k).exists() else k): mode
                for k, mode in keyword_config.items()
            }
            logger.info(
                f"openWakeWord engine initialized ({framework}) — "
                f"command: '{OWW_KEYWORD_COMMAND}', homecoming: '{OWW_KEYWORD_HOMECOMING}'"
            )
            return True

        except Exception as e:
            logger.error(f"openWakeWord init failed: {e}")
            self.oww_model = None
            return False

    def _enable_keyboard_fallback(self):
        """Enable keyboard-based wake word trigger."""
        import threading
        self.use_keyboard_fallback = True
        logger.info(
            "Keyboard mode: press Enter for command mode, "
            "type 'home' + Enter for homecoming mode."
        )

        def _listen():
            while self.is_listening:
                try:
                    line = input().strip().lower()
                    if line in ("home", "h"):
                        self._keyboard_triggered = WakeMode.HOMECOMING
                    else:
                        self._keyboard_triggered = WakeMode.COMMAND
                except EOFError:
                    break

        thread = threading.Thread(target=_listen, daemon=True)
        thread.start()

    @property
    def frame_length(self) -> int:
        """Audio frame length required by Porcupine."""
        if self.porcupine:
            return self.porcupine.frame_length
        return CHUNK_SIZE

    def detect(self, audio_chunk: np.ndarray) -> WakeMode | None:
        """
        Check if any wake word was spoken.
        Returns WakeMode if detected, None otherwise.
        """
        if self.use_keyboard_fallback:
            if self._keyboard_triggered is not None:
                mode = self._keyboard_triggered
                self._keyboard_triggered = None
                logger.info(f"Keyboard trigger → {mode.value} mode")
                return mode
            return None

        if self.porcupine is not None:
            keyword_index = self.porcupine.process(audio_chunk)
            if keyword_index >= 0:
                mode = self.keyword_mode_map[keyword_index]
                logger.info(f"Wake word detected (index {keyword_index}) → {mode.value} mode")
                return mode
            return None

        if self.oww_model is not None:
            predictions = self.oww_model.predict(audio_chunk)
            for name, mode in self.oww_mode_map.items():
                if predictions.get(name, 0.0) > OWW_THRESHOLD:
                    logger.info(f"Wake word '{name}' detected → {mode.value} mode")
                    return mode
            return None

        return None

    def cleanup(self):
        """Release wake word engine resources."""
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None
        self.oww_model = None

    def reset(self):
        """
        Clear internal detector state after a busy period.

        predict() isn't called at all while we're speaking/listening/waiting
        on the orchestrator, so openWakeWord's rolling feature-window buffer
        goes stale. Left alone, the next predict() call blends that old
        pre-interaction audio with fresh audio into a meaningless window,
        which can misfire as a detection the moment we resume listening.
        """
        if self.oww_model is not None:
            self.oww_model.reset()


class AudioCapture:
    """
    Captures audio from a USB (or built-in) microphone.
    Uses PyAudio for cross-platform compatibility (Raspberry Pi, macOS, etc).
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

    def drain(self):
        """
        Discard any audio already buffered by the OS/driver.

        The input stream keeps filling its buffer while we're busy doing
        something else (TTS playback, Whisper transcription, the HTTP round
        trip to the orchestrator) since none of that reads from it. Left
        alone, that backlog gets replayed instantly (not in real time) the
        next time we call read_chunk(), which both looks like silence before
        the user's real speech arrives (cutting off command capture early)
        and can surface a stale wake-word hit the moment we resume listening.
        """
        if not self.stream:
            return
        try:
            available = self.stream.get_read_available()
            while available > 0:
                self.stream.read(min(available, CHUNK_SIZE), exception_on_overflow=False)
                available = self.stream.get_read_available()
        except Exception:
            pass

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


PIPER_SAMPLE_RATE = int(os.getenv("PIPER_SAMPLE_RATE", "22050"))


class TextToSpeech:
    """
    Converts text to speech. Supports local (Piper) and cloud (ElevenLabs).
    Playback works on macOS (afplay), Linux (aplay/mpg123), and falls back
    to xdg-open elsewhere.
    """

    def __init__(self):
        self.engine = TTS_ENGINE

    async def speak(self, text: str):
        """Convert text to speech and play it."""
        path = await self.synthesize(text)
        if path:
            try:
                await self._play_file(path)
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    async def play_chime(self):
        """
        Play a short local tone instead of a spoken acknowledgment.

        A generated tone has no network round trip (unlike ElevenLabs) and is
        far shorter than a spoken sentence, so it won't talk over the user if
        they keep speaking right after the wake word.
        """
        duration = 0.15
        freq = 880
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        tone = (np.sin(2 * np.pi * freq * t) * 0.3 * 32767).astype(np.int16)

        fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="mel_chime_")
        try:
            os.close(fd)
            with wave.open(wav_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(tone.tobytes())
            await self._play_file(wav_path)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    @staticmethod
    async def _play_file(path: str):
        """Play an audio file using the platform's native player."""
        system = platform.system()
        if system == "Darwin":
            cmd = ["afplay", path]
        elif system == "Windows":
            cmd = ["cmd", "/c", "start", "/wait", "", path]
        else:
            cmd = ["aplay", path] if path.endswith(".wav") else ["mpg123", path]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except FileNotFoundError:
            logger.error(f"Audio player not found ({cmd[0]}). Install it to enable playback.")

    async def synthesize(self, text: str) -> str | None:
        """
        Convert text to speech and return the path to the resulting audio
        file, WITHOUT playing it. Caller owns playback and cleanup.

        Split out from speak() so a streaming response can synthesize the
        next sentence while the current one is still playing, instead of
        waiting for the whole reply before saying anything.
        """
        if self.engine == "piper":
            return await self._synthesize_piper(text)
        elif self.engine == "elevenlabs":
            return await self._synthesize_elevenlabs(text)
        logger.warning(f"Unknown TTS engine: {self.engine}")
        return None

    async def _synthesize_piper(self, text: str) -> str | None:
        """Use Piper TTS (runs locally, no cloud needed)."""
        try:
            # Piper reads stdin text → outputs raw 16-bit PCM audio
            piper_proc = await asyncio.create_subprocess_exec(
                "piper", "--model", PIPER_MODEL, "--output-raw",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            raw_audio, _ = await piper_proc.communicate(input=text.encode("utf-8"))

            if raw_audio:
                # Wrap the raw PCM in a WAV header so afplay/aplay can both play
                # it directly as a file (afplay does not accept headerless PCM).
                fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="mel_")
                os.close(fd)
                with wave.open(wav_path, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(PIPER_SAMPLE_RATE)
                    wav_file.writeframes(raw_audio)
                return wav_path
        except FileNotFoundError:
            logger.error("Piper not installed. Install: pip install piper-tts")
        except Exception as e:
            logger.error(f"Piper TTS error: {e}")
        return None

    async def _synthesize_elevenlabs(self, text: str) -> str | None:
        """Use ElevenLabs API (cloud, better quality)."""
        import httpx
        api_key = os.getenv("ELEVENLABS_API_KEY", "")
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

        if not api_key:
            logger.warning("ElevenLabs API key not set")
            return None

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    headers={"xi-api-key": api_key},
                    json={"text": text, "model_id": "eleven_multilingual_v2"},
                )
                if response.status_code == 200:
                    # Use secure temp file instead of predictable path
                    fd, audio_path = tempfile.mkstemp(suffix=".mp3", prefix="mel_")
                    os.write(fd, response.content)
                    os.close(fd)
                    return audio_path
                else:
                    logger.warning(f"ElevenLabs returned status {response.status_code}")
        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {e}")
        return None


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
        """Set up all components and verify orchestrator connection."""
        logger.info("Initializing Voice Agent...")

        # Verify we can reach the orchestrator before starting
        if not AGENT_API_KEY:
            logger.error("AGENT_API_KEY not set. Set it in .env or environment.")
            raise SystemExit(1)

        connected = await self._check_orchestrator()
        if not connected:
            logger.error(f"Cannot reach orchestrator at {ORCHESTRATOR_URL}")
            logger.error("Start the server first, then restart the listener.")
            raise SystemExit(1)

        self.wake_detector.initialize()
        self.keyboard_mode = self.wake_detector.use_keyboard_fallback

        if self.keyboard_mode:
            logger.info("Running in keyboard/text mode (no wake word engine).")
        else:
            self.stt.initialize()
            self.audio.start()
            logger.info("Voice Agent ready! Listening for wake word...")

    async def _check_orchestrator(self) -> bool:
        """Verify orchestrator is reachable and API key works."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{ORCHESTRATOR_URL}/health/pi")
                if resp.status_code != 200:
                    logger.error(f"Orchestrator health check returned {resp.status_code}")
                    return False
                # Now verify API key works on an authenticated endpoint
                resp = await client.get(
                    f"{ORCHESTRATOR_URL}/health/detail",
                    headers={"Authorization": f"Bearer {AGENT_API_KEY}"},
                )
                if resp.status_code == 401:
                    logger.error("API key is invalid — check AGENT_API_KEY in .env")
                    return False
                logger.info(f"Connected to orchestrator at {ORCHESTRATOR_URL}")
                return True
        except Exception as e:
            logger.error(f"Cannot connect to orchestrator: {e}")
            return False

    async def run(self):
        """Main 24/7 loop."""
        await self.initialize()

        try:
            if self.keyboard_mode:
                await self._run_keyboard_mode()
            else:
                await self._run_voice_mode()
        finally:
            self.wake_detector.cleanup()

    async def _run_voice_mode(self):
        """Standard voice mode with Porcupine wake word detection."""
        while True:
            try:
                audio_chunk = self.audio.read_chunk()

                if not self.is_active:
                    wake_mode = self.wake_detector.detect(audio_chunk)
                    if wake_mode is not None:
                        self.is_active = True

                        if wake_mode == WakeMode.HOMECOMING:
                            await self._homecoming_greeting()
                        else:
                            await self.tts.play_chime()
                            await self._converse_until_goodbye()

                        self.is_active = False
                        self.audio.drain()
                        self.wake_detector.reset()

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)

        self.audio.stop()

    async def _run_keyboard_mode(self):
        """Text-based fallback when wake word engine is unavailable."""
        print("\n" + "=" * 50)
        print("  MEL VOICE AGENT — Keyboard Mode")
        print("=" * 50)
        print("  Type your command and press Enter")
        print("  Type 'home' for homecoming greeting")
        print("  Type 'quit' to exit")
        print("=" * 50 + "\n")

        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("You > ").strip()
                )

                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "q"):
                    logger.info("Shutting down...")
                    break
                if user_input.lower() in ("home", "h"):
                    await self._homecoming_greeting()
                    continue

                response = await self._send_to_orchestrator(user_input)
                print(f"Mel > {response}\n")
                await self.tts.speak(response)

            except (KeyboardInterrupt, EOFError):
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                await asyncio.sleep(1)

    def _get_time_greeting(self) -> tuple[str, str]:
        """Return time-appropriate greeting and a friendly follow-up."""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "Good morning", "hope you had a great night"
        elif 12 <= hour < 17:
            return "Good afternoon", "hope your day is going well"
        elif 17 <= hour < 21:
            return "Good evening", "welcome home"
        else:
            return "Hey there", "burning the midnight oil I see"

    async def _homecoming_greeting(self):
        """
        Full homecoming flow — delegates to /homecoming server endpoint.
        Server handles: time-appropriate greeting, weather + smart advice,
        calendar briefing, and silently opens Chrome tabs (news + stocks).
        Pi just speaks the returned text, then waits for follow-up commands.
        """
        speech = await self._fetch_homecoming_speech()
        await self.tts.speak(speech)
        await self._process_command()

    async def _fetch_homecoming_speech(self) -> str:
        """Call /homecoming on the orchestrator server and return the speech text."""
        import httpx
        headers = {"Authorization": f"Bearer {AGENT_API_KEY}"}
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    f"{ORCHESTRATOR_URL}/homecoming",
                    headers=headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("speech", f"Welcome home, {USER_NAME}!")
                else:
                    logger.warning(f"/homecoming returned {resp.status_code}, using fallback")
        except Exception as e:
            logger.error(f"Homecoming fetch failed: {e}")

        # Fallback if server unreachable — basic local greeting
        greeting, followup = self._get_time_greeting()
        return f"{greeting} {USER_NAME}, {followup}. I couldn't reach the server for your full briefing right now."

    async def _fetch_today_calendar(self) -> str:
        """Ask the orchestrator for today's calendar events."""
        import httpx
        headers = {"Authorization": f"Bearer {AGENT_API_KEY}"}
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Use the calendar/events endpoint directly
                resp = await client.get(
                    f"{ORCHESTRATOR_URL}/calendar/events",
                    headers=headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    events = data.get("result", data.get("events", []))
                    if not events:
                        return ""
                    return self._format_calendar_events(events)
                else:
                    logger.warning(f"Calendar fetch returned {resp.status_code}")
                    # Fallback: ask via natural language through /process
                    resp = await client.post(
                        f"{ORCHESTRATOR_URL}/process",
                        json={"input": "What's on my calendar for today?"},
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        return resp.json().get("response", "")
        except Exception as e:
            logger.error(f"Calendar fetch failed: {e}")
        return ""

    @staticmethod
    def _format_calendar_events(events) -> str:
        """Format calendar events into a natural spoken summary."""
        if isinstance(events, str):
            return events

        if not isinstance(events, list) or len(events) == 0:
            return ""

        lines = []
        for event in events:
            if isinstance(event, dict):
                summary = event.get("summary", "Untitled event")
                start = event.get("start", {})
                time_str = start.get("dateTime", start.get("date", ""))
                if "T" in time_str:
                    try:
                        dt = datetime.fromisoformat(time_str)
                        time_str = dt.strftime("%-I:%M %p")
                    except (ValueError, TypeError):
                        pass
                if time_str:
                    lines.append(f"{summary} at {time_str}")
                else:
                    lines.append(summary)
            elif isinstance(event, str):
                lines.append(event)

        count = len(lines)
        if count == 1:
            return f"You have one thing today: {lines[0]}."
        else:
            items = ", ".join(lines[:-1]) + f", and {lines[-1]}"
            return f"You have {count} things today: {items}."

    async def _capture_command_text(self, drain_first: bool = True) -> str:
        """
        Record until the user stops talking and transcribe it locally.

        drain_first: discard any already-buffered audio before recording.
        Skip this right after wake-word detection -- if the user kept
        talking in the same breath as the wake word, that continuation is
        sitting in the buffer and we want to keep it. Do drain between
        turns in a conversation, where the buffered audio is stale leftover
        from Mel's own spoken response, not something to capture.
        """
        logger.info("Listening for command...")
        if drain_first:
            self.audio.drain()
        audio_buffer = []
        silence_count = 0
        max_silence_chunks = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
        max_total_chunks = int(MAX_COMMAND_DURATION * SAMPLE_RATE / CHUNK_SIZE)
        total_chunks = 0

        while silence_count < max_silence_chunks and total_chunks < max_total_chunks:
            chunk = self.audio.read_chunk()
            audio_buffer.append(chunk)
            total_chunks += 1

            # Detect silence (end of speech)
            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
            if rms < SILENCE_THRESHOLD:
                silence_count += 1
            else:
                silence_count = 0

        if total_chunks >= max_total_chunks:
            logger.info("Max command duration reached, processing what we have")

        # Transcribe locally
        full_audio = np.concatenate(audio_buffer)
        command = self.stt.transcribe(full_audio)
        # SECURITY: Don't log the actual command — only log that we got one
        logger.info(f"Command received ({len(command)} chars)")
        return command

    async def _process_command(self):
        """Capture a single command after wake word and reply (used by the
        homecoming flow's one-shot follow-up)."""
        command = await self._capture_command_text(drain_first=True)
        if command:
            await self._stream_reply(command)

    async def _converse_until_goodbye(self):
        """
        Stay awake after the wake word and keep listening for follow-up
        commands -- no need to repeat "Hey Mel" between turns -- replying to
        each right away, until the user says a sleep phrase (e.g. "goodbye
        mel") or goes quiet for a few turns in a row.
        """
        consecutive_silent = 0
        first_turn = True
        while True:
            command = await self._capture_command_text(drain_first=not first_turn)
            first_turn = False

            if not command:
                consecutive_silent += 1
                if consecutive_silent >= MAX_CONSECUTIVE_SILENT_TURNS:
                    logger.info("No follow-up heard, going back to sleep.")
                    return
                continue
            consecutive_silent = 0

            if any(phrase in command.lower() for phrase in SLEEP_PHRASES):
                await self.tts.speak("Goodbye!")
                return

            await self._stream_reply(command)

    async def _send_to_orchestrator(self, command: str) -> str:
        """Send command to the orchestrator API with auth and retry."""
        import httpx

        if not AGENT_API_KEY:
            logger.error("AGENT_API_KEY not set — cannot authenticate with orchestrator")
            return "I need an API key to connect. Please set AGENT_API_KEY in your environment."

        headers = {"Authorization": f"Bearer {AGENT_API_KEY}"}
        retries = [2, 4, 8]  # Retry delays in seconds

        for attempt in range(1 + len(retries)):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        f"{ORCHESTRATOR_URL}/process",
                        json={"input": command},
                        headers=headers,
                    )
                    if resp.status_code == 401:
                        logger.error("API key rejected by orchestrator")
                        return "Authentication failed. Check your API key."
                    if resp.status_code == 429:
                        logger.warning("Rate limited by orchestrator")
                        return "I'm being rate limited. Try again in a moment."
                    return resp.json().get("response", "I couldn't process that.")
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt < len(retries):
                    delay = retries[attempt]
                    logger.warning(f"Orchestrator connection failed, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("Orchestrator connection failed after retries")
                    return "I'm having trouble connecting. Is the server running?"
            except Exception as e:
                logger.error(f"Unexpected error contacting orchestrator: {e}")
                return "Something went wrong. Please try again."

        return "Connection failed."

    async def _stream_reply(self, command: str) -> str:
        """
        Stream the orchestrator's response and speak it sentence-by-sentence
        as it arrives, synthesizing each sentence's audio while the previous
        one is still playing -- instead of waiting for the entire reply
        before saying anything. Falls back to a spoken error message (never
        raises) so callers can treat this like _send_to_orchestrator().
        """
        import httpx

        if not AGENT_API_KEY:
            logger.error("AGENT_API_KEY not set — cannot authenticate with orchestrator")
            await self.tts.speak("I need an API key to connect. Please set AGENT_API_KEY in your environment.")
            return ""

        playback_queue: asyncio.Queue = asyncio.Queue()

        async def player():
            while True:
                path = await playback_queue.get()
                if path is None:
                    break
                try:
                    await self.tts._play_file(path)
                finally:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

        async def enqueue(text: str):
            text = text.strip()
            if not text:
                return
            path = await self.tts.synthesize(text)
            if path:
                await playback_queue.put(path)

        player_task = asyncio.create_task(player())
        full_response = ""
        buffer = ""
        headers = {"Authorization": f"Bearer {AGENT_API_KEY}"}

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST", f"{ORCHESTRATOR_URL}/process/stream",
                    json={"input": command}, headers=headers,
                ) as resp:
                    if resp.status_code == 401:
                        await enqueue("Authentication failed. Check your API key.")
                    else:
                        event_type = None
                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            if line.startswith("event:"):
                                event_type = line[len("event:"):].strip()
                            elif line.startswith("data:"):
                                try:
                                    data = json.loads(line[len("data:"):].strip())
                                except ValueError:
                                    continue
                                if event_type == "token":
                                    token = data.get("token", "")
                                    buffer += token
                                    full_response += token
                                    ready, buffer = _split_ready_sentences(buffer)
                                    if ready:
                                        await enqueue(ready)
                                elif event_type == "error":
                                    logger.error(f"Stream error: {data.get('error')}")
                                elif event_type == "done":
                                    break
        except (httpx.ConnectError, httpx.TimeoutException):
            logger.error("Orchestrator streaming connection failed")
            if not full_response:
                await enqueue("I'm having trouble connecting. Is the server running?")
        except Exception as e:
            logger.error(f"Unexpected error streaming from orchestrator: {e}")
            if not full_response:
                await enqueue("Something went wrong. Please try again.")

        if buffer.strip():
            await enqueue(buffer)

        await playback_queue.put(None)
        await player_task
        return full_response


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    agent = VoiceAgent()
    asyncio.run(agent.run())
