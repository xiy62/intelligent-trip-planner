"""Regression tests for weather geocoding selection."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from app.models.schemas import WeatherInfo
from app.services.weather_service import OpenMeteoWeatherProvider, WeatherService


class OpenMeteoWeatherProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = OpenMeteoWeatherProvider()

    def test_build_geocoding_queries_prefers_english_alias_for_cn_city(self):
        self.assertEqual(self.provider._build_geocoding_queries("北京"), ["Beijing", "北京"])
        self.assertEqual(self.provider._build_geocoding_queries("杭州"), ["Hangzhou", "杭州"])

    def test_select_best_result_prefers_true_beijing_over_same_name_town(self):
        results = [
            {
                "name": "北京",
                "latitude": 30.72608,
                "longitude": 108.67483,
                "feature_code": "PPL",
                "country_code": "CN",
                "admin1": "重庆市",
                "admin2": "重庆市",
            },
            {
                "name": "北京市",
                "latitude": 39.9075,
                "longitude": 116.39723,
                "feature_code": "PPLC",
                "country_code": "CN",
                "admin1": "北京",
                "admin2": "北京市",
            },
        ]

        best = self.provider._select_best_result("北京", results)
        self.assertIsNotNone(best)
        self.assertEqual(best["latitude"], 39.9075)
        self.assertEqual(best["longitude"], 116.39723)

    def test_select_best_result_prefers_true_hangzhou_over_same_name_county(self):
        results = [
            {
                "name": "杭州",
                "latitude": 30.06517,
                "longitude": 102.19527,
                "feature_code": "PPL",
                "country_code": "CN",
                "admin1": "四川",
                "admin2": "甘孜",
            },
            {
                "name": "杭州市",
                "latitude": 30.29365,
                "longitude": 120.16142,
                "feature_code": "PPLA",
                "country_code": "CN",
                "admin1": "浙江",
                "admin2": "杭州",
            },
        ]

        best = self.provider._select_best_result("杭州", results)
        self.assertIsNotNone(best)
        self.assertEqual(best["latitude"], 30.29365)
        self.assertEqual(best["longitude"], 120.16142)

    def test_future_dates_outside_forecast_horizon_return_unknown_weather(self):
        start_date = (datetime.now().date() + timedelta(days=40)).strftime("%Y-%m-%d")
        forecast = self.provider.get_forecast("北京", start_date, 2)

        self.assertEqual(len(forecast), 2)
        self.assertTrue(all(item.day_weather == "Unknown" for item in forecast))
        self.assertTrue(all(item.night_weather == "Unknown" for item in forecast))

    def test_null_weather_code_returns_unknown_weather(self):
        self.provider._within_forecast_horizon = lambda _start_date: True
        self.provider._geocode_city = lambda _city: (40.7128, -74.0060)
        self.provider._fetch_forecast = lambda *_args: {
            "daily": {
                "time": ["2026-07-01"],
                "weather_code": [None],
                "temperature_2m_max": [None],
                "temperature_2m_min": [None],
                "wind_speed_10m_max": [None],
                "wind_direction_10m_dominant": [None],
            }
        }

        forecast = self.provider.get_forecast("New York", "2026-07-01", 1)

        self.assertEqual(len(forecast), 1)
        self.assertEqual(forecast[0].day_weather, "Unknown")
        self.assertEqual(forecast[0].night_weather, "Unknown")
        self.assertEqual(forecast[0].day_temp, 0)

    def test_formatter_hides_zero_temperature_when_forecast_unavailable(self):
        service = WeatherService()
        summary = service.format_weather_for_planner(
            "New York",
            [
                WeatherInfo(
                    date="2026-07-01",
                    day_weather="Unknown",
                    night_weather="Unknown",
                    day_temp=0,
                    night_temp=0,
                )
            ],
        )

        self.assertIn("forecast unavailable", summary)
        self.assertNotIn("0°C", summary)


if __name__ == "__main__":
    unittest.main()
