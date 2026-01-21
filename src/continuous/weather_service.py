"""
Weather Service for Recording Sessions

Fetches current weather data for Viroflay, France location when recordings are made.
Supports multiple weather API providers with fallback options.
"""

import os
import logging
import requests
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WeatherData:
    """Weather observation data"""
    temperature_c: float
    humidity_percent: float
    pressure_hpa: float
    conditions: str
    wind_speed_kmh: float
    wind_direction: str
    timestamp: datetime
    api_source: str


class WeatherService:
    """
    Fetch weather data for recording location.

    Default location: 2 rue de la Tuilerie, 78220 Viroflay, France
    Coordinates: 48.8006° N, 2.1714° E

    Implements caching to reduce API calls (15 minute cache by default).
    """

    # Viroflay, France coordinates
    DEFAULT_LAT = 48.8006
    DEFAULT_LON = 2.1714
    DEFAULT_LOCATION_NAME = "Viroflay, France"

    # Cache duration in seconds (default: 15 minutes)
    CACHE_DURATION_SECONDS = 900

    def __init__(self, api_key: Optional[str] = None, cache_duration: int = None):
        """
        Initialize weather service.

        Args:
            api_key: OpenWeatherMap API key (or read from OPENWEATHER_API_KEY env var)
            cache_duration: Cache duration in seconds (default: 900 = 15 minutes)
        """
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.cache_duration = cache_duration or self.CACHE_DURATION_SECONDS

        # Cache storage: {(lat, lon): (timestamp, WeatherData)}
        self._cache = {}

        if not self.api_key:
            logger.warning("No OpenWeatherMap API key found. Set OPENWEATHER_API_KEY environment variable.")
            logger.warning("Will attempt to use Open-Meteo (free, no key required) as fallback.")

    def get_current_weather(self,
                           lat: Optional[float] = None,
                           lon: Optional[float] = None,
                           force_refresh: bool = False) -> Optional[WeatherData]:
        """
        Get current weather for location (with caching).

        Args:
            lat: Latitude (default: Viroflay)
            lon: Longitude (default: Viroflay)
            force_refresh: Skip cache and fetch fresh data

        Returns:
            WeatherData object or None if all providers fail
        """
        lat = lat or self.DEFAULT_LAT
        lon = lon or self.DEFAULT_LON

        cache_key = (lat, lon)

        # Check cache first (unless force refresh)
        if not force_refresh and cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            age_seconds = (datetime.now() - cached_time).total_seconds()

            if age_seconds < self.cache_duration:
                logger.info(f"Using cached weather data (age: {age_seconds:.0f}s)")
                return cached_data

        # Cache miss or expired - fetch fresh data
        weather_data = None

        # Try OpenWeatherMap first (if API key available)
        if self.api_key:
            try:
                weather_data = self._fetch_openweathermap(lat, lon)
            except Exception as e:
                logger.error(f"OpenWeatherMap failed: {e}")

        # Fallback to Open-Meteo (free, no API key)
        if not weather_data:
            try:
                weather_data = self._fetch_openmeteo(lat, lon)
            except Exception as e:
                logger.error(f"Open-Meteo failed: {e}")

        if weather_data:
            # Store in cache
            self._cache[cache_key] = (datetime.now(), weather_data)
            logger.info(f"Weather data cached for {self.cache_duration}s")
        else:
            logger.error("All weather providers failed")

        return weather_data

    def _fetch_openweathermap(self, lat: float, lon: float) -> WeatherData:
        """
        Fetch weather from OpenWeatherMap API.

        API docs: https://openweathermap.org/current
        """
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'  # Celsius, m/s
        }

        logger.info(f"Fetching weather from OpenWeatherMap for ({lat}, {lon})...")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Parse response
        return WeatherData(
            temperature_c=data['main']['temp'],
            humidity_percent=data['main']['humidity'],
            pressure_hpa=data['main']['pressure'],
            conditions=data['weather'][0]['description'].title(),
            wind_speed_kmh=data['wind']['speed'] * 3.6,  # Convert m/s to km/h
            wind_direction=self._degrees_to_direction(data['wind'].get('deg', 0)),
            timestamp=datetime.fromtimestamp(data['dt']),
            api_source='OpenWeatherMap'
        )

    def _fetch_openmeteo(self, lat: float, lon: float) -> WeatherData:
        """
        Fetch weather from Open-Meteo API (free, no key required).

        API docs: https://open-meteo.com/
        """
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'current_weather': 'true',
            'hourly': 'relativehumidity_2m,surface_pressure'
        }

        logger.info(f"Fetching weather from Open-Meteo for ({lat}, {lon})...")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        current = data['current_weather']

        # Get humidity and pressure from hourly data (current hour)
        hourly = data['hourly']
        current_time_idx = 0  # First entry is current hour
        humidity = hourly['relativehumidity_2m'][current_time_idx] if 'relativehumidity_2m' in hourly else 0
        pressure = hourly['surface_pressure'][current_time_idx] if 'surface_pressure' in hourly else 1013

        # Map WMO weather code to description
        conditions = self._wmo_code_to_description(current.get('weathercode', 0))

        return WeatherData(
            temperature_c=current['temperature'],
            humidity_percent=humidity,
            pressure_hpa=pressure,
            conditions=conditions,
            wind_speed_kmh=current['windspeed'],
            wind_direction=self._degrees_to_direction(current['winddirection']),
            timestamp=datetime.fromisoformat(current['time']),
            api_source='Open-Meteo'
        )

    @staticmethod
    def _degrees_to_direction(degrees: float) -> str:
        """Convert wind direction from degrees to cardinal direction"""
        if degrees is None:
            return "N"

        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        index = int((degrees + 22.5) / 45) % 8
        return directions[index]

    @staticmethod
    def _wmo_code_to_description(code: int) -> str:
        """
        Convert WMO weather code to human-readable description.

        WMO codes: https://www.nodc.noaa.gov/archive/arc0021/0002199/1.1/data/0-data/HTML/WMO-CODE/WMO4677.HTM
        """
        code_map = {
            0: "Clear Sky",
            1: "Mainly Clear",
            2: "Partly Cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing Rime Fog",
            51: "Light Drizzle",
            53: "Moderate Drizzle",
            55: "Dense Drizzle",
            61: "Slight Rain",
            63: "Moderate Rain",
            65: "Heavy Rain",
            71: "Slight Snow",
            73: "Moderate Snow",
            75: "Heavy Snow",
            77: "Snow Grains",
            80: "Slight Rain Showers",
            81: "Moderate Rain Showers",
            82: "Violent Rain Showers",
            85: "Slight Snow Showers",
            86: "Heavy Snow Showers",
            95: "Thunderstorm",
            96: "Thunderstorm with Slight Hail",
            99: "Thunderstorm with Heavy Hail"
        }
        return code_map.get(code, f"Unknown ({code})")

    def get_weather_dict(self,
                        lat: Optional[float] = None,
                        lon: Optional[float] = None) -> Dict:
        """
        Get weather as dictionary for database storage.

        Returns:
            Dictionary with weather fields, or empty dict if fetch fails
        """
        weather = self.get_current_weather(lat, lon)

        if not weather:
            logger.warning("Weather fetch failed, returning empty data")
            return {}

        return {
            'weather_temperature_c': weather.temperature_c,
            'weather_humidity_percent': weather.humidity_percent,
            'weather_pressure_hpa': weather.pressure_hpa,
            'weather_conditions': weather.conditions,
            'weather_wind_speed_kmh': weather.wind_speed_kmh,
            'weather_wind_direction': weather.wind_direction,
            'weather_timestamp': weather.timestamp,
            'weather_api_source': weather.api_source
        }

    def get_weather_summary(self,
                           lat: Optional[float] = None,
                           lon: Optional[float] = None) -> str:
        """
        Get human-readable weather summary.

        Returns:
            String like "12.5°C, 72% humidity, Partly Cloudy"
        """
        weather = self.get_current_weather(lat, lon)

        if not weather:
            return "Weather data unavailable"

        return (f"{weather.temperature_c:.1f}°C, "
                f"{weather.humidity_percent:.0f}% humidity, "
                f"{weather.conditions}")


# Convenience function for quick access
def get_viroflay_weather() -> Dict:
    """
    Get current weather for Viroflay, France location.

    Returns:
        Dictionary with weather fields
    """
    service = WeatherService()
    return service.get_weather_dict()


if __name__ == '__main__':
    # Test weather service
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("Testing Weather Service")
    print("=" * 60)
    print(f"Location: {WeatherService.DEFAULT_LOCATION_NAME}")
    print(f"Coordinates: {WeatherService.DEFAULT_LAT}°N, {WeatherService.DEFAULT_LON}°E")
    print()

    service = WeatherService()

    # Test fetch
    weather = service.get_current_weather()

    if weather:
        print("Current Weather:")
        print(f"  Temperature: {weather.temperature_c:.1f}°C")
        print(f"  Humidity: {weather.humidity_percent:.0f}%")
        print(f"  Pressure: {weather.pressure_hpa:.0f} hPa")
        print(f"  Conditions: {weather.conditions}")
        print(f"  Wind: {weather.wind_speed_kmh:.1f} km/h from {weather.wind_direction}")
        print(f"  Observed: {weather.timestamp}")
        print(f"  Source: {weather.api_source}")
        print()
        print("Summary:", service.get_weather_summary())
    else:
        print("✗ Weather fetch failed")

    print()
    print("Dictionary format (for database):")
    print(service.get_weather_dict())
