"""
WEATHER MODULE
===============
Real-time weather data via OpenWeatherMap API.
Supports current conditions, forecasts, and natural language summaries.
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional

import httpx

logger = logging.getLogger("weather")


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


class WeatherService:
    """Fetches weather data from OpenWeatherMap API."""

    BASE_URL = "https://api.openweathermap.org/data/2.5"

    def __init__(self):
        self.api_key = _load_secret("OPENWEATHER_API_KEY")
        self.default_city = os.getenv("WEATHER_CITY", "Dallas")
        self.units = os.getenv("WEATHER_UNITS", "imperial")  # imperial=°F, metric=°C

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def get_current(self, city: Optional[str] = None) -> dict:
        """Get current weather for a city."""
        if not self.api_key:
            return {"error": "OpenWeatherMap not configured. Add OPENWEATHER_API_KEY to .env"}

        city = city or self.default_city
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.BASE_URL}/weather",
                    params={"q": city, "appid": self.api_key, "units": self.units},
                )
                if resp.status_code != 200:
                    return {"error": f"Weather API error: {resp.status_code}"}
                data = resp.json()

            unit_symbol = "°F" if self.units == "imperial" else "°C"
            speed_unit = "mph" if self.units == "imperial" else "m/s"

            return {
                "city": data["name"],
                "country": data["sys"]["country"],
                "temperature": round(data["main"]["temp"]),
                "feels_like": round(data["main"]["feels_like"]),
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"].title(),
                "icon": data["weather"][0]["icon"],
                "wind_speed": round(data["wind"]["speed"]),
                "unit_symbol": unit_symbol,
                "speed_unit": speed_unit,
                "sunrise": datetime.fromtimestamp(data["sys"]["sunrise"]).strftime("%I:%M %p").lstrip("0"),
                "sunset": datetime.fromtimestamp(data["sys"]["sunset"]).strftime("%I:%M %p").lstrip("0"),
            }
        except Exception as e:
            logger.error(f"Weather fetch failed: {e}")
            return {"error": f"Could not fetch weather: {e}"}

    async def get_forecast(self, city: Optional[str] = None, days: int = 3) -> dict:
        """Get multi-day forecast (3-hour intervals, grouped by day)."""
        if not self.api_key:
            return {"error": "OpenWeatherMap not configured."}

        city = city or self.default_city
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.BASE_URL}/forecast",
                    params={"q": city, "appid": self.api_key, "units": self.units, "cnt": days * 8},
                )
                if resp.status_code != 200:
                    return {"error": f"Forecast API error: {resp.status_code}"}
                data = resp.json()

            unit_symbol = "°F" if self.units == "imperial" else "°C"
            daily = {}
            for item in data["list"]:
                date = item["dt_txt"].split(" ")[0]
                if date not in daily:
                    daily[date] = {"temps": [], "descriptions": [], "date": date}
                daily[date]["temps"].append(item["main"]["temp"])
                daily[date]["descriptions"].append(item["weather"][0]["description"])

            forecast = []
            for date, info in list(daily.items())[:days]:
                most_common_desc = max(set(info["descriptions"]), key=info["descriptions"].count)
                forecast.append({
                    "date": date,
                    "high": round(max(info["temps"])),
                    "low": round(min(info["temps"])),
                    "description": most_common_desc.title(),
                    "unit_symbol": unit_symbol,
                })

            return {"city": data["city"]["name"], "forecast": forecast}
        except Exception as e:
            logger.error(f"Forecast fetch failed: {e}")
            return {"error": f"Could not fetch forecast: {e}"}

    async def get_summary(self, city: Optional[str] = None) -> str:
        """Get a natural language weather summary for voice/text responses."""
        current = await self.get_current(city)
        if "error" in current:
            return current["error"]

        unit = current["unit_symbol"]
        summary = (
            f"It's currently {current['temperature']}{unit} in {current['city']} — "
            f"{current['description'].lower()}. "
            f"Feels like {current['feels_like']}{unit} with "
            f"{current['humidity']}% humidity and "
            f"winds at {current['wind_speed']} {current['speed_unit']}. "
            f"Sunrise was at {current['sunrise']}, sunset at {current['sunset']}."
        )

        # Add umbrella advice
        rain_words = ["rain", "drizzle", "shower", "storm", "thunderstorm"]
        if any(w in current["description"].lower() for w in rain_words):
            summary += " You might want to grab an umbrella!"

        return summary

    @staticmethod
    def get_weather_advice(temp: float, description: str, units: str = "imperial") -> str:
        """Return smart contextual advice based on weather conditions.

        Returns one or more natural-language tips suitable for spoken TTS.
        """
        advice = []
        desc_lower = description.lower()

        # Rain / storms
        rain_words = ["rain", "drizzle", "shower", "storm", "thunderstorm", "sleet", "hail"]
        if any(w in desc_lower for w in rain_words):
            advice.append("Grab your umbrella — it's wet out there.")

        # Snow / ice
        snow_words = ["snow", "blizzard", "flurr"]
        if any(w in desc_lower for w in snow_words):
            advice.append("Roads may be slippery — drive carefully.")

        # Temperature-based advice (imperial °F)
        if units == "imperial":
            if temp >= 95:
                advice.append("It's blazing hot. Stay hydrated and try to stay cool.")
            elif temp >= 85:
                advice.append("It's pretty hot out. Stay hydrated.")
            elif temp <= 20:
                advice.append("Bundle up — it's dangerously cold out there.")
            elif temp <= 35:
                advice.append("Wear your heavy coat and gloves, it's freezing.")
            elif temp <= 45:
                advice.append("Wear your jacket — it's cold out.")
        else:  # metric °C
            if temp >= 35:
                advice.append("It's blazing hot. Stay hydrated and try to stay cool.")
            elif temp >= 30:
                advice.append("It's pretty hot out. Stay hydrated.")
            elif temp <= -6:
                advice.append("Bundle up — it's dangerously cold out there.")
            elif temp <= 2:
                advice.append("Wear your heavy coat and gloves, it's freezing.")
            elif temp <= 7:
                advice.append("Wear your jacket — it's cold out.")

        # Fog / low visibility
        if "fog" in desc_lower or "mist" in desc_lower:
            advice.append("Low visibility out there — leave a bit early if you're driving.")

        # Wind
        if "wind" in desc_lower or "gust" in desc_lower:
            advice.append("It's quite windy today.")

        return " ".join(advice) if advice else ""

    async def needs_umbrella(self, city: Optional[str] = None) -> str:
        """Quick check if umbrella is needed."""
        current = await self.get_current(city)
        if "error" in current:
            return current["error"]

        rain_words = ["rain", "drizzle", "shower", "storm", "thunderstorm"]
        if any(w in current["description"].lower() for w in rain_words):
            return f"Yes! It's {current['description'].lower()} in {current['city']}. Definitely grab an umbrella."
        return f"No umbrella needed — it's {current['description'].lower()} in {current['city']}."


def register_weather_plugins(action_registry):
    """Register weather actions with the action registry."""
    weather = WeatherService()

    async def weather_current(params: dict) -> str:
        return await weather.get_summary(params.get("city"))

    async def weather_forecast(params: dict) -> str:
        result = await weather.get_forecast(params.get("city"), params.get("days", 3))
        if "error" in result:
            return result["error"]
        lines = [f"Forecast for {result['city']}:"]
        for day in result["forecast"]:
            lines.append(f"  {day['date']}: {day['description']} — {day['low']}{day['unit_symbol']} to {day['high']}{day['unit_symbol']}")
        return "\n".join(lines)

    async def weather_umbrella(params: dict) -> str:
        return await weather.needs_umbrella(params.get("city"))

    action_registry.register("weather_current", weather_current, "Get current weather")
    action_registry.register("weather_forecast", weather_forecast, "Get weather forecast")
    action_registry.register("weather_umbrella", weather_umbrella, "Check if umbrella needed")

    return weather
