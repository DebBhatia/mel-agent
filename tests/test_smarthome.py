"""Tests for the smart home integration module."""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from smarthome import (
    HomeAssistantClient,
    SmartHomeManager,
    DeviceState,
    register_smarthome_plugins,
)
from orchestrator import ActionRegistry


class TestHomeAssistantClient:
    def test_not_configured_by_default(self):
        ha = HomeAssistantClient()
        # Unless env vars are set, should not be configured
        if not os.getenv("HOME_ASSISTANT_URL"):
            assert not ha.is_configured()

    def test_configured_with_env(self):
        ha = HomeAssistantClient()
        ha.base_url = "http://192.168.1.100:8123"
        ha.token = "test-token"
        assert ha.is_configured()


class TestSmartHomeManager:
    def test_init_has_default_devices(self):
        manager = SmartHomeManager()
        assert len(manager._simulated_devices) > 0

    @pytest.mark.asyncio
    async def test_get_all_devices_simulated(self):
        manager = SmartHomeManager()
        manager.ha.base_url = ""  # Force simulated mode
        devices = await manager.get_all_devices()
        assert len(devices) > 0
        assert any(d["type"] == "light" for d in devices)
        assert any(d["type"] == "lock" for d in devices)
        assert any(d["type"] == "climate" for d in devices)

    @pytest.mark.asyncio
    async def test_turn_on_simulated(self):
        manager = SmartHomeManager()
        manager.ha.base_url = ""
        result = await manager.turn_on("light.living_room")
        assert "turned on" in result.lower()
        # Verify state changed
        device = manager._find_simulated("light.living_room")
        assert device["state"] == "on"

    @pytest.mark.asyncio
    async def test_turn_off_simulated(self):
        manager = SmartHomeManager()
        manager.ha.base_url = ""
        await manager.turn_on("light.living_room")
        result = await manager.turn_off("light.living_room")
        assert "turned off" in result.lower()

    @pytest.mark.asyncio
    async def test_set_temperature(self):
        manager = SmartHomeManager()
        manager.ha.base_url = ""
        result = await manager.set_temperature(75)
        assert "75" in result
        thermostat = next(d for d in manager._simulated_devices if d["type"] == "climate")
        assert thermostat["temperature"] == 75

    @pytest.mark.asyncio
    async def test_lock_door(self):
        manager = SmartHomeManager()
        manager.ha.base_url = ""
        result = await manager.lock_door("front_door")
        assert "locked" in result.lower()

    @pytest.mark.asyncio
    async def test_unlock_door(self):
        manager = SmartHomeManager()
        manager.ha.base_url = ""
        result = await manager.unlock_door("front_door")
        assert "unlocked" in result.lower()

    @pytest.mark.asyncio
    async def test_set_brightness(self):
        manager = SmartHomeManager()
        manager.ha.base_url = ""
        result = await manager.set_brightness("light.living_room", 50)
        assert "50%" in result

    @pytest.mark.asyncio
    async def test_activate_scene(self):
        manager = SmartHomeManager()
        manager.ha.base_url = ""
        result = await manager.activate_scene("movie_night")
        assert "movie night" in result.lower()

    @pytest.mark.asyncio
    async def test_status_summary(self):
        manager = SmartHomeManager()
        manager.ha.base_url = ""
        summary = await manager.get_status_summary()
        assert "lights" in summary.lower()
        assert "thermostat" in summary.lower()
        assert "locks" in summary.lower()

    @pytest.mark.asyncio
    async def test_device_not_found(self):
        manager = SmartHomeManager()
        manager.ha.base_url = ""
        result = await manager.turn_on("nonexistent.device")
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_find_by_name(self):
        manager = SmartHomeManager()
        device = manager._find_by_name("living room")
        assert device is not None
        assert "living" in device["name"].lower()


class TestSmartHomePluginRegistration:
    def test_register_all(self):
        registry = ActionRegistry()
        manager = register_smarthome_plugins(registry)
        assert "home_turn_on" in registry.actions
        assert "home_turn_off" in registry.actions
        assert "home_temperature" in registry.actions
        assert "home_lock" in registry.actions
        assert "home_unlock" in registry.actions
        assert "home_brightness" in registry.actions
        assert "home_scene" in registry.actions
        assert "home_status" in registry.actions
        assert "home_devices" in registry.actions
        assert len(registry.actions) == 9
        assert manager is not None
