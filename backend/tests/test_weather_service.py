"""Regression tests for weather geocoding selection."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from app.services.weather_service import OpenMeteoWeatherProvider


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
        self.assertTrue(all(item.day_weather == "未知" for item in forecast))
        self.assertTrue(all(item.night_weather == "未知" for item in forecast))


if __name__ == "__main__":
    unittest.main()
