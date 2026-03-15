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
import logging
from typing import Optional
from urllib.parse import urlencode

import httpx

logger = logging.getLogger("music")

SPOTIFY_API_BASE = "https://api.spotify.com/v1"
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_SCOPES = (
    "user-read-playback-state user-modify-playback-state user-read-currently-playing "
    "playlist-read-private playlist-read-collaborative user-library-read"
)


class SpotifyAuth:
    """Handles Spotify OAuth2 PKCE flow with local token persistence."""

    def __init__(self):
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID", "")
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "")
        self.redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8000/spotify/callback")
        self.token_path = os.path.expanduser("~/.config/agent/spotify_token.json")
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
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": SPOTIFY_SCOPES,
            "show_dialog": "true",
        }
        return f"{SPOTIFY_AUTH_URL}?{urlencode(params)}"

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

    async def _api(self, method: str, endpoint: str, json_body: dict = None, params: dict = None) -> Optional[dict]:
        token = await self.auth.get_access_token()
        if not token:
            return None
        headers = {"Authorization": f"Bearer {token}"}
        url = f"{SPOTIFY_API_BASE}{endpoint}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if method == "GET":
                    resp = await client.get(url, headers=headers, params=params)
                elif method == "PUT":
                    resp = await client.put(url, headers=headers, json=json_body)
                elif method == "POST":
                    resp = await client.post(url, headers=headers, json=json_body)
                else:
                    return None
                if resp.status_code in (200, 201):
                    return resp.json() if resp.content else {}
                elif resp.status_code == 204:
                    return {}
                else:
                    logger.warning(f"Spotify API {method} {endpoint}: {resp.status_code}")
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

    async def play(self, uri: str = None, context_uri: str = None) -> str:
        body = {}
        if context_uri:
            body["context_uri"] = context_uri
        elif uri:
            body["uris"] = [uri]
        result = await self._api("PUT", "/me/player/play", json_body=body if body else None)
        if result is not None:
            return "Playing music."
        return "Could not start playback. Make sure Spotify is open on a device."

    async def pause(self) -> str:
        result = await self._api("PUT", "/me/player/pause")
        if result is not None:
            return "Music paused."
        return "Could not pause. Is Spotify playing?"

    async def next_track(self) -> str:
        result = await self._api("POST", "/me/player/next")
        if result is not None:
            return "Skipped to next track."
        return "Could not skip track."

    async def previous_track(self) -> str:
        result = await self._api("POST", "/me/player/previous")
        if result is not None:
            return "Playing previous track."
        return "Could not go back."

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
            results = await player.search(query, "track", 1)
            if results:
                return await player.play(uri=results[0]["uri"])
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
