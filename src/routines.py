"""
ROUTINES ENGINE
================
Automated routines that chain multiple agent actions together.
Morning briefing, goodnight routine, leaving home, custom routines.
Routines can be triggered by voice, schedule, or API call.
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("routines")

ROUTINES_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "routines.json")


@dataclass
class RoutineStep:
    action: str           # Action name (e.g., "weather_current", "calendar_list")
    params: dict = field(default_factory=dict)
    label: str = ""       # Human-readable label for this step


@dataclass
class Routine:
    id: str
    name: str
    description: str
    steps: list           # List of RoutineStep dicts
    enabled: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ─────────────────────────────────────────────
# Built-in Routine Definitions
# ─────────────────────────────────────────────
BUILTIN_ROUTINES = {
    "morning_briefing": Routine(
        id="morning_briefing",
        name="Morning Briefing",
        description="Weather, calendar, and a greeting to start your day.",
        steps=[
            {"action": "weather_current", "params": {}, "label": "Current weather"},
            {"action": "weather_forecast", "params": {"days": 1}, "label": "Today's forecast"},
            {"action": "calendar_list", "params": {"days": 1}, "label": "Today's events"},
        ],
    ),
    "goodnight": Routine(
        id="goodnight",
        name="Goodnight",
        description="Lock up, lights off, and tomorrow's preview.",
        steps=[
            {"action": "home_lock", "params": {"device": "front_door"}, "label": "Lock front door"},
            {"action": "home_turn_off", "params": {"device": "light.living_room"}, "label": "Living room lights off"},
            {"action": "home_turn_off", "params": {"device": "light.kitchen"}, "label": "Kitchen lights off"},
            {"action": "home_temperature", "params": {"value": "68"}, "label": "Set thermostat to 68°F"},
            {"action": "calendar_list", "params": {"days": 1}, "label": "Tomorrow's schedule"},
        ],
    ),
    "leaving_home": Routine(
        id="leaving_home",
        name="Leaving Home",
        description="Turn off lights, lock doors, and lower thermostat.",
        steps=[
            {"action": "home_turn_off", "params": {"device": "light.living_room"}, "label": "Living room lights off"},
            {"action": "home_turn_off", "params": {"device": "light.bedroom"}, "label": "Bedroom lights off"},
            {"action": "home_turn_off", "params": {"device": "light.kitchen"}, "label": "Kitchen lights off"},
            {"action": "home_lock", "params": {"device": "front_door"}, "label": "Lock front door"},
            {"action": "home_temperature", "params": {"value": "72"}, "label": "Thermostat to eco mode"},
            {"action": "weather_current", "params": {}, "label": "Weather check before you go"},
        ],
    ),
    "welcome_home": Routine(
        id="welcome_home",
        name="Welcome Home",
        description="Lights on, comfortable temperature, and what's happening.",
        steps=[
            {"action": "home_turn_on", "params": {"device": "light.living_room"}, "label": "Living room lights on"},
            {"action": "home_temperature", "params": {"value": "72"}, "label": "Set comfortable temperature"},
            {"action": "weather_current", "params": {}, "label": "Current weather"},
            {"action": "calendar_list", "params": {"days": 1}, "label": "Remaining events today"},
            {"action": "music_play", "params": {}, "label": "Resume music"},
        ],
    ),
}


class RoutineEngine:
    """Manages and executes multi-step routines."""

    def __init__(self):
        self.routines = dict(BUILTIN_ROUTINES)
        self.custom_routines = {}
        self._load_custom()

    def _ensure_data_dir(self):
        os.makedirs(os.path.dirname(ROUTINES_FILE), exist_ok=True)

    def _load_custom(self):
        """Load user-defined custom routines from disk."""
        try:
            if os.path.exists(ROUTINES_FILE):
                with open(ROUTINES_FILE) as f:
                    data = json.load(f)
                for r in data:
                    routine = Routine(**r)
                    self.custom_routines[routine.id] = routine
                    self.routines[routine.id] = routine
        except Exception as e:
            logger.warning(f"Could not load custom routines: {e}")

    def _save_custom(self):
        """Persist custom routines to disk."""
        self._ensure_data_dir()
        try:
            data = [asdict(r) for r in self.custom_routines.values()]
            with open(ROUTINES_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save routines: {e}")

    def list_routines(self) -> list:
        """List all available routines."""
        result = []
        for rid, routine in self.routines.items():
            result.append({
                "id": routine.id,
                "name": routine.name,
                "description": routine.description,
                "steps": len(routine.steps),
                "enabled": routine.enabled,
                "builtin": rid in BUILTIN_ROUTINES,
            })
        return result

    def get_routine(self, routine_id: str) -> Optional[Routine]:
        return self.routines.get(routine_id)

    def create_custom(self, name: str, description: str, steps: list) -> Routine:
        """Create a new custom routine."""
        import uuid
        rid = f"custom_{uuid.uuid4().hex[:8]}"
        routine = Routine(
            id=rid,
            name=name,
            description=description,
            steps=steps,
        )
        self.custom_routines[rid] = routine
        self.routines[rid] = routine
        self._save_custom()
        return routine

    def delete_custom(self, routine_id: str) -> bool:
        """Delete a custom routine."""
        if routine_id in BUILTIN_ROUTINES:
            return False  # Can't delete built-in routines
        if routine_id in self.custom_routines:
            del self.custom_routines[routine_id]
            del self.routines[routine_id]
            self._save_custom()
            return True
        return False

    async def execute(self, routine_id: str, action_registry) -> dict:
        """Execute a routine step by step."""
        routine = self.routines.get(routine_id)
        if not routine:
            return {"status": "error", "message": f"Routine '{routine_id}' not found."}

        if not routine.enabled:
            return {"status": "error", "message": f"Routine '{routine.name}' is disabled."}

        results = []
        errors = []

        for i, step in enumerate(routine.steps):
            action = step.get("action", "") if isinstance(step, dict) else step.action
            params = step.get("params", {}) if isinstance(step, dict) else step.params
            label = step.get("label", action) if isinstance(step, dict) else step.label

            try:
                result = await action_registry.execute(action, params)
                results.append({"step": i + 1, "label": label, "result": result, "status": "ok"})
            except Exception as e:
                err_msg = f"Step failed: {e}"
                results.append({"step": i + 1, "label": label, "result": err_msg, "status": "error"})
                errors.append(err_msg)
                logger.error(f"Routine '{routine_id}' step {i+1} ({action}) failed: {e}")

        return {
            "status": "completed" if not errors else "completed_with_errors",
            "routine": routine.name,
            "steps_total": len(routine.steps),
            "steps_ok": len(results) - len(errors),
            "steps_failed": len(errors),
            "results": results,
        }

    async def execute_and_summarize(self, routine_id: str, action_registry, agent_name: str = "Mel", user_name: str = "Deb") -> str:
        """Execute routine and return a natural language summary."""
        result = await self.execute(routine_id, action_registry)

        if result["status"] == "error":
            return result["message"]

        routine = self.routines[routine_id]
        lines = [f"Running {routine.name}...\n"]

        for step_result in result["results"]:
            status_icon = "OK" if step_result["status"] == "ok" else "FAIL"
            lines.append(f"[{status_icon}] {step_result['label']}: {step_result['result']}")

        if result["steps_failed"] > 0:
            lines.append(f"\n{result['steps_failed']} step(s) had issues, but everything else is handled, {user_name}.")
        else:
            lines.append(f"\nAll done, {user_name}. {routine.name} complete.")

        return "\n".join(lines)
