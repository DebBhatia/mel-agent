"""
SPOTIFY MUSIC PLAYER PLUGIN
=============================
Controls Spotify playback via the Spotify Web API.
Supports play, pause, skip, search, playlists, and now-playing.

Requires: SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI
in .env. User must complete OAuth2 flow once via /spotify/auth.

All tokens stored locally — never sent to cloud AI.
"""

import os
import json
import time
import asyncio
import logging
import secrets
from typing import Optional
from urllib.parse import urlencode

import httpx

logger = logging.getLogger("music")


def _load_secret(key: str, default: str = "") -> str:
    """Load a secret from the encrypted vault; fall back to env var."""
    try:
        from security import SecretVault
        val = SecretVault().get(key, "")
        if val:
            return val
    except Exception:
        pass
    return os.getenv(key, default)

SPOTIFY_API_BASE = "https://api.spotify.com/v1"
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_SCOPES = (
    "streaming user-read-playback-state user-modify-playback-state user-read-currently-playing "
    "playlist-read-private playlist-read-collaborative user-library-read user-read-email user-read-private"
)


class SpotifyAuth:
    """Handles Spotify OAuth2 PKCE flow with local token persistence."""

    def __init__(self):
        self.client_id = _load_secret("SPOTIFY_CLIENT_ID")
        self.client_secret = _load_secret("SPOTIFY_CLIENT_SECRET")
        self.redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8000/spotify/callback")
        self.token_path = os.path.expanduser("~/.config/agent/spotify_token.json")
        self.state_path = os.path.expanduser("~/.config/agent/spotify_oauth_state.json")
        self._token_data = {}
        self._load_token()

    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _load_token(self):
        if os.path.exists(self.token_path):
            try:
                with open(self.token_path) as f:
                    self._token_data = json.load(f)
            except Exception:
                self._token_data = {}

    def _save_token(self):
        os.makedirs(os.path.dirname(self.token_path), exist_ok=True)
        with open(self.token_path, "w") as f:
            json.dump(self._token_data, f)
        try:
            os.chmod(self.token_path, 0o600)
        except Exception:
            pass

    def get_auth_url(self) -> str:
        state = secrets.token_urlsafe(32)
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump({"state": state, "expires_at": time.time() + 600}, f)
            os.chmod(self.state_path, 0o600)
        except Exception:
            logger.error("Failed to persist Spotify OAuth state")
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": SPOTIFY_SCOPES,
            "show_dialog": "true",
            "state": state,
        }
        return f"{SPOTIFY_AUTH_URL}?{urlencode(params)}"

    def verify_state(self, state: str) -> bool:
        """Validate the CSRF `state` param from the OAuth callback against the
        one issued in get_auth_url(). One-time use: consumed on success or failure."""
        try:
            with open(self.state_path) as f:
                pending = json.load(f)
        except Exception:
            return False
        finally:
            try:
                os.remove(self.state_path)
            except Exception:
                pass
        expected = pending.get("state", "")
        if not state or not expected:
            return False
        if time.time() >= pending.get("expires_at", 0):
            return False
        return secrets.compare_digest(state, expected)

    async def exchange_code(self, code: str) -> bool:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                SPOTIFY_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": self.redirect_uri,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if resp.status_code == 200:
                data = resp.json()
                self._token_data = {
                    "access_token": data["access_token"],
                    "refresh_token": data.get("refresh_token", ""),
                    "expires_at": time.time() + data.get("expires_in", 3600),
                    "token_type": data.get("token_type", "Bearer"),
                }
                self._save_token()
                logger.info("Spotify authenticated successfully")
                return True
            logger.error(f"Spotify token exchange failed: {resp.status_code}")
            return False

    async def get_access_token(self) -> Optional[str]:
        if not self._token_data.get("access_token"):
            return None
        if time.time() >= self._token_data.get("expires_at", 0) - 60:
            if not await self._refresh():
                return None
        return self._token_data["access_token"]

    async def _refresh(self) -> bool:
        refresh_token = self._token_data.get("refresh_token")
        if not refresh_token:
            return False
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    SPOTIFY_TOKEN_URL,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    self._token_data["access_token"] = data["access_token"]
                    self._token_data["expires_at"] = time.time() + data.get("expires_in", 3600)
                    if "refresh_token" in data:
                        self._token_data["refresh_token"] = data["refresh_token"]
                    self._save_token()
                    return True
        except Exception as e:
            logger.error(f"Spotify token refresh failed: {e}")
        return False

    @property
    def is_authenticated(self) -> bool:
        return bool(self._token_data.get("access_token"))


class SpotifyPlayer:
    """Controls Spotify playback via Web API."""

    def __init__(self):
        self.auth = SpotifyAuth()
        self.dashboard_device_id = None  # Set by /spotify/register-device

    async def _api(self, method: str, endpoint: str, json_body: dict = None, params: dict = None) -> Optional[dict]:
        token = await self.auth.get_access_token()
        if not token:
            return None
        headers = {"Authorization": f"Bearer {token}"}
        url = f"{SPOTIFY_API_BASE}{endpoint}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                kwargs = {"headers": headers}
                if json_body is not None:
                    kwargs["json"] = json_body
                if params is not None:
                    kwargs["params"] = params

                if method == "GET":
                    resp = await client.get(url, **kwargs)
                elif method == "PUT":
                    if json_body is None:
                        resp = await client.put(url, headers=headers, content=b"")
                    else:
                        resp = await client.put(url, headers=headers, json=json_body)
                elif method == "POST":
                    if json_body is None:
                        resp = await client.post(url, headers=headers, content=b"")
                    else:
                        resp = await client.post(url, headers=headers, json=json_body)
                else:
                    return None

                if resp.status_code in (200, 201, 202):
                    if not resp.content:
                        return {}
                    try:
                        return resp.json()
                    except Exception:
                        return {}
                elif resp.status_code == 204:
                    return {}
                elif resp.status_code == 403:
                    logger.warning(f"Spotify API {method} {endpoint}: 403 Forbidden (Premium required?)")
                    return None
                else:
                    body = resp.text[:150] if resp.text else ""
                    logger.warning(f"Spotify API {method} {endpoint}: {resp.status_code} {body}")
                    return None
        except Exception as e:
            logger.error(f"Spotify API error: {e}")
            return None

    async def now_playing(self) -> dict:
        data = await self._api("GET", "/me/player/currently-playing")
        if not data or not data.get("item"):
            return {"is_playing": False, "track": None, "artist": None, "album": None, "progress_ms": 0, "duration_ms": 0, "album_art": None}
        item = data["item"]
        artists = ", ".join(a["name"] for a in item.get("artists", []))
        album_art = None
        images = item.get("album", {}).get("images", [])
        if images:
            album_art = images[0]["url"]
        return {
            "is_playing": data.get("is_playing", False),
            "track": item.get("name", "Unknown"),
            "artist": artists,
            "album": item.get("album", {}).get("name", ""),
            "progress_ms": data.get("progress_ms", 0),
            "duration_ms": item.get("duration_ms", 0),
            "album_art": album_art,
            "device": data.get("device", {}).get("name", "Unknown"),
        }

    async def play(self, uri: str = None, context_uri: str = None, device_id: str = None) -> str:
        body = {}
        if context_uri:
            body["context_uri"] = context_uri
        elif uri:
            body["uris"] = [uri]

        # Find a target device
        target_id = device_id
        target_name = "your device"
        if not target_id:
            # 1st priority: registered dashboard web player (Mel Dashboard SDK)
            if self.dashboard_device_id:
                target_id = self.dashboard_device_id
                target_name = "Mel Dashboard"
            else:
                # 2nd: discover from Spotify API
                devices = await self.get_devices()
                if devices:
                    mel_dev = next((d for d in devices if "mel" in d["name"].lower()), None)
                    active_dev = next((d for d in devices if d["is_active"]), None)
                    pick = mel_dev or active_dev or devices[0]
                    target_id = pick["id"]
                    target_name = pick["name"]
                else:
                    return "No Spotify device found. Open the dashboard Music page or Spotify app."

        # Transfer playback to target device, then play
        await self._api("PUT", "/me/player", json_body={"device_ids": [target_id], "play": False})
        await asyncio.sleep(0.3)
        result = await self._api("PUT", f"/me/player/play?device_id={target_id}",
                                 json_body=body if body else None)
        if result is not None:
            await asyncio.sleep(0.8)
            np = await self.now_playing()
            if np.get("track"):
                return f"Now playing: {np['track']} by {np['artist']} on {target_name}."
            return f"Playing on {target_name}."
        return f"Could not start playback on {target_name}. Try refreshing the dashboard."

    async def _ensure_active_device(self) -> Optional[str]:
        """Make sure there's an active device. Returns device_id or None."""
        # Prefer registered dashboard device
        if self.dashboard_device_id:
            return self.dashboard_device_id
        state = await self._api("GET", "/me/player")
        if state and state.get("device"):
            return state["device"].get("id")
        # No active session — find a device and transfer
        devices = await self.get_devices()
        if not devices:
            return None
        mel_dev = next((d for d in devices if "mel" in d["name"].lower()), None)
        pick = mel_dev or devices[0]
        await self._api("PUT", "/me/player", json_body={"device_ids": [pick["id"]], "play": False})
        await asyncio.sleep(0.3)
        return pick["id"]

    async def pause(self) -> str:
        device_id = await self._ensure_active_device()
        endpoint = "/me/player/pause"
        if device_id:
            endpoint = f"/me/player/pause?device_id={device_id}"
        result = await self._api("PUT", endpoint)
        if result is not None:
            return "Music paused."
        return "Could not pause. Is Spotify playing?"

    async def next_track(self) -> str:
        device_id = await self._ensure_active_device()
        if not device_id:
            return "No Spotify device found. Open Spotify or go to the Music page."
        # Pass device_id to target the correct device
        result = await self._api("POST", f"/me/player/next?device_id={device_id}")
        if result is not None:
            await asyncio.sleep(0.5)
            np = await self.now_playing()
            if np.get("track"):
                return f"Now playing: {np['track']} by {np['artist']}."
            return "Skipped to next track."
        # Retry once — Spotify API can be flaky
        await asyncio.sleep(0.3)
        result = await self._api("POST", f"/me/player/next?device_id={device_id}")
        if result is not None:
            await asyncio.sleep(0.5)
            np = await self.now_playing()
            if np.get("track"):
                return f"Now playing: {np['track']} by {np['artist']}."
            return "Skipped to next track."
        return "Could not skip track. Try playing something first."

    async def previous_track(self) -> str:
        device_id = await self._ensure_active_device()
        if not device_id:
            return "No Spotify device found. Open Spotify or go to the Music page."
        # Pass device_id to target the correct device
        result = await self._api("POST", f"/me/player/previous?device_id={device_id}")
        if result is not None:
            await asyncio.sleep(0.5)
            np = await self.now_playing()
            if np.get("track"):
                return f"Now playing: {np['track']} by {np['artist']}."
            return "Playing previous track."
        # Retry once
        await asyncio.sleep(0.3)
        result = await self._api("POST", f"/me/player/previous?device_id={device_id}")
        if result is not None:
            await asyncio.sleep(0.5)
            np = await self.now_playing()
            if np.get("track"):
                return f"Now playing: {np['track']} by {np['artist']}."
            return "Playing previous track."
        return "Could not go back. Try playing something first."

    async def set_volume(self, volume: int) -> str:
        volume = max(0, min(100, volume))
        token = await self.auth.get_access_token()
        if not token:
            return "Not authenticated with Spotify."
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.put(
                    f"{SPOTIFY_API_BASE}/me/player/volume",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"volume_percent": volume},
                )
                if resp.status_code in (200, 204):
                    return f"Volume set to {volume}%."
        except Exception:
            pass
        return "Could not set volume."

    async def shuffle(self, state: bool = True) -> str:
        device_id = await self._ensure_active_device()
        endpoint = f"/me/player/shuffle?state={'true' if state else 'false'}"
        if device_id:
            endpoint += f"&device_id={device_id}"
        result = await self._api("PUT", endpoint)
        if result is not None:
            return f"Shuffle {'on' if state else 'off'}."
        return "Could not toggle shuffle."

    async def repeat(self, state: str = "context") -> str:
        """Set repeat mode: 'track', 'context' (playlist/album), or 'off'."""
        device_id = await self._ensure_active_device()
        endpoint = f"/me/player/repeat?state={state}"
        if device_id:
            endpoint += f"&device_id={device_id}"
        result = await self._api("PUT", endpoint)
        if result is not None:
            return f"Repeat set to {state}."
        return "Could not set repeat mode."

    async def seek(self, position_ms: int) -> str:
        device_id = await self._ensure_active_device()
        endpoint = f"/me/player/seek?position_ms={position_ms}"
        if device_id:
            endpoint += f"&device_id={device_id}"
        result = await self._api("PUT", endpoint)
        if result is not None:
            return "Seeking..."
        return "Could not seek."

    async def search(self, query: str, search_type: str = "track", limit: int = 5) -> list[dict]:
        data = await self._api("GET", "/search", params={"q": query, "type": search_type, "limit": limit})
        if not data:
            return []
        results = []
        key = f"{search_type}s"
        for item in data.get(key, {}).get("items", []):
            entry = {"name": item.get("name", ""), "uri": item.get("uri", "")}
            if search_type == "track":
                entry["artist"] = ", ".join(a["name"] for a in item.get("artists", []))
                entry["album"] = item.get("album", {}).get("name", "")
                images = item.get("album", {}).get("images", [])
                entry["album_art"] = images[0]["url"] if images else None
            elif search_type == "playlist":
                entry["owner"] = item.get("owner", {}).get("display_name", "")
                entry["tracks"] = item.get("tracks", {}).get("total", 0)
            results.append(entry)
        return results

    async def get_devices(self) -> list[dict]:
        data = await self._api("GET", "/me/player/devices")
        if not data:
            return []
        return [
            {"id": d["id"], "name": d["name"], "type": d["type"], "is_active": d["is_active"], "volume": d.get("volume_percent", 0)}
            for d in data.get("devices", [])
        ]

    async def get_playlists(self, limit: int = 10) -> list[dict]:
        data = await self._api("GET", "/me/playlists", params={"limit": limit})
        if not data:
            return []
        return [
            {"name": p["name"], "uri": p["uri"], "tracks": p["tracks"]["total"],
             "image": p["images"][0]["url"] if p.get("images") else None}
            for p in data.get("items", [])
        ]


def register_music_plugins(action_registry):
    """Register music player actions."""
    player = SpotifyPlayer()

    async def play_music(params):
        query = params.get("query", "")
        if query:
            # Try track search first
            results = await player.search(query, "track", 1)
            if results:
                return await player.play(uri=results[0]["uri"])
            # Fall back to playlist search (e.g. "today's country", "chill vibes")
            playlists = await player.search(query, "playlist", 1)
            if playlists:
                return await player.play(context_uri=playlists[0]["uri"])
            return f"No results found for '{query}'."
        return await player.play()

    async def pause_music(params):
        return await player.pause()

    async def skip_track(params):
        return await player.next_track()

    async def now_playing(params):
        data = await player.now_playing()
        if not data.get("track"):
            return "Nothing is currently playing."
        status = "Playing" if data["is_playing"] else "Paused"
        return f"{status}: {data['track']} by {data['artist']} (Album: {data['album']})"

    async def search_music(params):
        query = params.get("query", "")
        if not query:
            return "What would you like to search for?"
        results = await player.search(query, params.get("type", "track"))
        if not results:
            return f"No results for '{query}'."
        lines = []
        for i, r in enumerate(results, 1):
            if "artist" in r:
                lines.append(f"{i}. {r['name']} — {r['artist']}")
            else:
                lines.append(f"{i}. {r['name']}")
        return "Search results:\n" + "\n".join(lines)

    action_registry.register("music_play", play_music, "Play music or a specific song")
    action_registry.register("music_pause", pause_music, "Pause music")
    action_registry.register("music_skip", skip_track, "Skip to next track")
    action_registry.register("music_now_playing", now_playing, "Show current track")
    action_registry.register("music_search", search_music, "Search for music")

    logger.info("Spotify music plugins registered")
