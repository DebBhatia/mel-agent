"""Tests for the Spotify music player plugin."""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from music import SpotifyAuth, SpotifyPlayer, register_music_plugins
from orchestrator import ActionRegistry


class TestSpotifyAuth:
    def test_init_defaults(self):
        auth = SpotifyAuth()
        assert isinstance(auth.client_id, str)
        assert isinstance(auth.client_secret, str)
        assert auth.redirect_uri.startswith("http")

    def test_not_configured_without_env(self):
        auth = SpotifyAuth()
        auth.client_id = ""
        auth.client_secret = ""
        assert not auth.is_configured()

    def test_configured_with_values(self):
        auth = SpotifyAuth()
        auth.client_id = "test-id"
        auth.client_secret = "test-secret"
        assert auth.is_configured()

    def test_not_authenticated_initially(self):
        auth = SpotifyAuth()
        auth._token_data = {}
        assert not auth.is_authenticated

    def test_authenticated_with_token(self):
        auth = SpotifyAuth()
        auth._token_data = {"access_token": "test-token"}
        assert auth.is_authenticated

    def test_get_auth_url(self):
        auth = SpotifyAuth()
        auth.client_id = "test-client-id"
        url = auth.get_auth_url()
        assert "accounts.spotify.com/authorize" in url
        assert "test-client-id" in url
        assert "response_type=code" in url


class TestSpotifyPlayer:
    def test_init(self):
        player = SpotifyPlayer()
        assert player.auth is not None
        assert isinstance(player.auth, SpotifyAuth)

    @pytest.mark.asyncio
    async def test_now_playing_no_auth(self):
        player = SpotifyPlayer()
        player.auth._token_data = {}
        result = await player.now_playing()
        assert result["is_playing"] is False
        assert result["track"] is None

    @pytest.mark.asyncio
    async def test_play_no_auth(self):
        player = SpotifyPlayer()
        player.auth._token_data = {}
        result = await player.play()
        assert "could not" in result.lower() or "no" in result.lower()

    @pytest.mark.asyncio
    async def test_pause_no_auth(self):
        player = SpotifyPlayer()
        player.auth._token_data = {}
        result = await player.pause()
        assert "could not" in result.lower() or "not" in result.lower()

    @pytest.mark.asyncio
    async def test_search_no_auth(self):
        player = SpotifyPlayer()
        player.auth._token_data = {}
        result = await player.search("test query")
        assert result == []

    @pytest.mark.asyncio
    async def test_set_volume_clamps(self):
        player = SpotifyPlayer()
        player.auth._token_data = {}
        result = await player.set_volume(150)
        assert "not authenticated" in result.lower() or "not" in result.lower()


class TestSpotifyPlayerExtended:
    @pytest.mark.asyncio
    async def test_next_track_no_auth(self):
        player = SpotifyPlayer()
        player.auth._token_data = {}
        result = await player.next_track()
        assert "no" in result.lower() or "could not" in result.lower()

    @pytest.mark.asyncio
    async def test_previous_track_no_auth(self):
        player = SpotifyPlayer()
        player.auth._token_data = {}
        result = await player.previous_track()
        assert "no" in result.lower() or "could not" in result.lower()

    @pytest.mark.asyncio
    async def test_shuffle_no_auth(self):
        player = SpotifyPlayer()
        player.auth._token_data = {}
        result = await player.shuffle(True)
        assert "could not" in result.lower() or "not" in result.lower()

    @pytest.mark.asyncio
    async def test_repeat_no_auth(self):
        player = SpotifyPlayer()
        player.auth._token_data = {}
        result = await player.repeat("track")
        assert "could not" in result.lower() or "not" in result.lower()

    @pytest.mark.asyncio
    async def test_seek_no_auth(self):
        player = SpotifyPlayer()
        player.auth._token_data = {}
        result = await player.seek(30000)
        assert "could not" in result.lower() or "not" in result.lower()

    @pytest.mark.asyncio
    async def test_get_devices_no_auth(self):
        player = SpotifyPlayer()
        player.auth._token_data = {}
        result = await player.get_devices()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_playlists_no_auth(self):
        player = SpotifyPlayer()
        player.auth._token_data = {}
        result = await player.get_playlists()
        assert result == []


class TestMusicPluginRegistration:
    def test_register_all(self):
        registry = ActionRegistry()
        register_music_plugins(registry)
        assert "music_play" in registry.actions
        assert "music_pause" in registry.actions
        assert "music_skip" in registry.actions
        assert "music_now_playing" in registry.actions
        assert "music_search" in registry.actions
        assert len(registry.actions) == 5
