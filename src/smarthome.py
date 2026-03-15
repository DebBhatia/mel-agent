"""
SMART HOME INTEGRATION
=======================
Controls smart home devices via Home Assistant REST API or MQTT.
Supports lights, thermostats, locks, switches, and scenes.

Requires: HOME_ASSISTANT_URL and HOME_ASSISTANT_TOKEN in .env
for Home Assistant mode. Or MQTT_BROKER for MQTT mode.

All communication stays on local network — no cloud dependency.
"""

import os
import json
import logging
from typing import Optional
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger("smarthome")


@dataclass
class DeviceState:
    entity_id: str
    name: str
    state: str  # "on", "off", "unavailable", numeric value
    device_type: str  # "light", "switch", "climate", "lock", "scene", "cover"
    attributes: dict = field(default_factory=dict)


class HomeAssistantClient:
    """
    Controls devices via Home Assistant REST API.
    Runs entirely on your local network.
    """

    def __init__(self):
        self.base_url = os.getenv("HOME_ASSISTANT_URL", "").rstrip("/")
        self.token = os.getenv("HOME_ASSISTANT_TOKEN", "")
        self._headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def is_configured(self) -> bool:
        return bool(self.base_url and self.token)

    async def _api(self, method: str, endpoint: str, json_body: dict = None) -> Optional[dict]:
        if not self.is_configured():
            return None
        url = f"{self.base_url}/api{endpoint}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if method == "GET":
                    resp = await client.get(url, headers=self._headers)
                elif method == "POST":
                    resp = await client.post(url, headers=self._headers, json=json_body or {})
                else:
                    return None
                if resp.status_code == 200:
                    return resp.json()
                logger.warning(f"HA API {method} {endpoint}: {resp.status_code}")
                return None
        except Exception as e:
            logger.error(f"Home Assistant API error: {e}")
            return None

    async def get_states(self) -> list[DeviceState]:
        data = await self._api("GET", "/states")
        if not data:
            return []
        devices = []
        for entity in data:
            eid = entity.get("entity_id", "")
            domain = eid.split(".")[0] if "." in eid else "unknown"
            if domain in ("light", "switch", "climate", "lock", "scene", "cover", "fan", "media_player"):
                devices.append(DeviceState(
                    entity_id=eid,
                    name=entity.get("attributes", {}).get("friendly_name", eid),
                    state=entity.get("state", "unknown"),
                    device_type=domain,
                    attributes=entity.get("attributes", {}),
                ))
        return devices

    async def get_device(self, entity_id: str) -> Optional[DeviceState]:
        data = await self._api("GET", f"/states/{entity_id}")
        if not data:
            return None
        domain = entity_id.split(".")[0] if "." in entity_id else "unknown"
        return DeviceState(
            entity_id=entity_id,
            name=data.get("attributes", {}).get("friendly_name", entity_id),
            state=data.get("state", "unknown"),
            device_type=domain,
            attributes=data.get("attributes", {}),
        )

    async def call_service(self, domain: str, service: str, entity_id: str = None, data: dict = None) -> bool:
        body = data or {}
        if entity_id:
            body["entity_id"] = entity_id
        result = await self._api("POST", f"/services/{domain}/{service}", body)
        return result is not None

    async def turn_on(self, entity_id: str, **kwargs) -> str:
        domain = entity_id.split(".")[0]
        success = await self.call_service(domain, "turn_on", entity_id, kwargs if kwargs else None)
        name = entity_id.split(".")[-1].replace("_", " ").title()
        return f"Turned on {name}." if success else f"Failed to turn on {name}."

    async def turn_off(self, entity_id: str) -> str:
        domain = entity_id.split(".")[0]
        success = await self.call_service(domain, "turn_off", entity_id)
        name = entity_id.split(".")[-1].replace("_", " ").title()
        return f"Turned off {name}." if success else f"Failed to turn off {name}."

    async def set_temperature(self, entity_id: str, temperature: float) -> str:
        success = await self.call_service("climate", "set_temperature", entity_id, {"temperature": temperature})
        return f"Thermostat set to {temperature}°." if success else "Failed to set temperature."

    async def lock(self, entity_id: str) -> str:
        success = await self.call_service("lock", "lock", entity_id)
        name = entity_id.split(".")[-1].replace("_", " ").title()
        return f"Locked {name}." if success else f"Failed to lock {name}."

    async def unlock(self, entity_id: str) -> str:
        success = await self.call_service("lock", "unlock", entity_id)
        name = entity_id.split(".")[-1].replace("_", " ").title()
        return f"Unlocked {name}." if success else f"Failed to unlock {name}."

    async def set_brightness(self, entity_id: str, brightness: int) -> str:
        """Set light brightness (0-255)."""
        brightness = max(0, min(255, brightness))
        success = await self.call_service("light", "turn_on", entity_id, {"brightness": brightness})
        pct = round(brightness / 255 * 100)
        return f"Brightness set to {pct}%." if success else "Failed to set brightness."

    async def set_color(self, entity_id: str, rgb: list) -> str:
        """Set light color via RGB."""
        success = await self.call_service("light", "turn_on", entity_id, {"rgb_color": rgb})
        return "Color changed." if success else "Failed to set color."

    async def activate_scene(self, entity_id: str) -> str:
        success = await self.call_service("scene", "turn_on", entity_id)
        name = entity_id.split(".")[-1].replace("_", " ").title()
        return f"Activated scene: {name}." if success else f"Failed to activate scene: {name}."


class SmartHomeManager:
    """
    High-level smart home manager that provides a natural language interface.
    Falls back to a simulated device list when Home Assistant is not configured.
    """

    def __init__(self):
        self.ha = HomeAssistantClient()
        self._simulated_devices = self._default_devices()

    @staticmethod
    def _default_devices() -> list[dict]:
        """Default simulated devices for demo/testing when HA is not connected."""
        return [
            {"entity_id": "light.living_room", "name": "Living Room Light", "state": "off", "type": "light", "brightness": 0},
            {"entity_id": "light.bedroom", "name": "Bedroom Light", "state": "off", "type": "light", "brightness": 0},
            {"entity_id": "light.kitchen", "name": "Kitchen Light", "state": "off", "type": "light", "brightness": 0},
            {"entity_id": "switch.fan", "name": "Ceiling Fan", "state": "off", "type": "switch"},
            {"entity_id": "climate.thermostat", "name": "Thermostat", "state": "72", "type": "climate", "temperature": 72},
            {"entity_id": "lock.front_door", "name": "Front Door Lock", "state": "locked", "type": "lock"},
            {"entity_id": "lock.garage", "name": "Garage Door Lock", "state": "locked", "type": "lock"},
            {"entity_id": "scene.movie_night", "name": "Movie Night", "state": "idle", "type": "scene"},
            {"entity_id": "scene.good_morning", "name": "Good Morning", "state": "idle", "type": "scene"},
        ]

    def _find_simulated(self, entity_id: str) -> Optional[dict]:
        for d in self._simulated_devices:
            if d["entity_id"] == entity_id:
                return d
        return None

    def _find_by_name(self, name: str) -> Optional[dict]:
        name_lower = name.lower()
        for d in self._simulated_devices:
            if name_lower in d["name"].lower() or name_lower in d["entity_id"]:
                return d
        return None

    async def get_all_devices(self) -> list[dict]:
        if self.ha.is_configured():
            states = await self.ha.get_states()
            return [
                {"entity_id": s.entity_id, "name": s.name, "state": s.state,
                 "type": s.device_type, **s.attributes}
                for s in states
            ]
        return self._simulated_devices

    async def turn_on(self, identifier: str) -> str:
        if self.ha.is_configured():
            return await self.ha.turn_on(identifier)
        device = self._find_simulated(identifier) or self._find_by_name(identifier)
        if device:
            device["state"] = "on"
            return f"Turned on {device['name']}."
        return f"Device '{identifier}' not found."

    async def turn_off(self, identifier: str) -> str:
        if self.ha.is_configured():
            return await self.ha.turn_off(identifier)
        device = self._find_simulated(identifier) or self._find_by_name(identifier)
        if device:
            device["state"] = "off"
            return f"Turned off {device['name']}."
        return f"Device '{identifier}' not found."

    async def set_temperature(self, temperature: float) -> str:
        if self.ha.is_configured():
            return await self.ha.set_temperature("climate.thermostat", temperature)
        for d in self._simulated_devices:
            if d["type"] == "climate":
                d["state"] = str(int(temperature))
                d["temperature"] = temperature
                return f"Thermostat set to {temperature}°F."
        return "No thermostat found."

    async def lock_door(self, identifier: str = "front_door") -> str:
        if self.ha.is_configured():
            return await self.ha.lock(f"lock.{identifier}")
        device = self._find_simulated(f"lock.{identifier}") or self._find_by_name(identifier)
        if device:
            device["state"] = "locked"
            return f"Locked {device['name']}."
        return f"Lock '{identifier}' not found."

    async def unlock_door(self, identifier: str = "front_door") -> str:
        if self.ha.is_configured():
            return await self.ha.unlock(f"lock.{identifier}")
        device = self._find_simulated(f"lock.{identifier}") or self._find_by_name(identifier)
        if device:
            device["state"] = "unlocked"
            return f"Unlocked {device['name']}."
        return f"Lock '{identifier}' not found."

    async def set_brightness(self, identifier: str, brightness_pct: int) -> str:
        brightness = max(0, min(255, int(brightness_pct * 255 / 100)))
        if self.ha.is_configured():
            return await self.ha.set_brightness(identifier, brightness)
        device = self._find_simulated(identifier) or self._find_by_name(identifier)
        if device and device["type"] == "light":
            device["state"] = "on" if brightness_pct > 0 else "off"
            device["brightness"] = brightness_pct
            return f"Set {device['name']} brightness to {brightness_pct}%."
        return f"Light '{identifier}' not found."

    async def activate_scene(self, scene_name: str) -> str:
        if self.ha.is_configured():
            return await self.ha.activate_scene(f"scene.{scene_name}")
        device = self._find_simulated(f"scene.{scene_name}") or self._find_by_name(scene_name)
        if device:
            return f"Activated scene: {device['name']}."
        return f"Scene '{scene_name}' not found."

    async def get_status_summary(self) -> str:
        devices = await self.get_all_devices()
        if not devices:
            return "No smart home devices connected."
        lights_on = sum(1 for d in devices if d["type"] == "light" and d["state"] == "on")
        total_lights = sum(1 for d in devices if d["type"] == "light")
        thermostat = next((d for d in devices if d["type"] == "climate"), None)
        locks = [d for d in devices if d["type"] == "lock"]
        locked = sum(1 for d in locks if d["state"] == "locked")

        lines = [f"Lights: {lights_on}/{total_lights} on"]
        if thermostat:
            temp = thermostat.get("temperature", thermostat.get("state", "??"))
            lines.append(f"Thermostat: {temp}°F")
        if locks:
            lines.append(f"Locks: {locked}/{len(locks)} locked")
        return " | ".join(lines)


def register_smarthome_plugins(action_registry):
    """Register smart home actions."""
    manager = SmartHomeManager()

    async def home_turn_on(params):
        return await manager.turn_on(params.get("device", ""))

    async def home_turn_off(params):
        return await manager.turn_off(params.get("device", ""))

    async def home_temperature(params):
        return await manager.set_temperature(float(params.get("temperature", 72)))

    async def home_lock(params):
        return await manager.lock_door(params.get("door", "front_door"))

    async def home_unlock(params):
        return await manager.unlock_door(params.get("door", "front_door"))

    async def home_brightness(params):
        return await manager.set_brightness(params.get("device", ""), int(params.get("brightness", 100)))

    async def home_scene(params):
        return await manager.activate_scene(params.get("scene", ""))

    async def home_status(params):
        return await manager.get_status_summary()

    async def home_devices(params):
        devices = await manager.get_all_devices()
        return json.dumps(devices, indent=2)

    action_registry.register("home_turn_on", home_turn_on, "Turn on a device")
    action_registry.register("home_turn_off", home_turn_off, "Turn off a device")
    action_registry.register("home_temperature", home_temperature, "Set thermostat temperature")
    action_registry.register("home_lock", home_lock, "Lock a door")
    action_registry.register("home_unlock", home_unlock, "Unlock a door")
    action_registry.register("home_brightness", home_brightness, "Set light brightness")
    action_registry.register("home_scene", home_scene, "Activate a scene")
    action_registry.register("home_status", home_status, "Get smart home status summary")
    action_registry.register("home_devices", home_devices, "List all devices")

    logger.info("Smart home plugins registered")
    return manager
