"""Tests for precise, fail-closed weather location labels."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from services.weather import (  # noqa: E402
    _format_reverse_geocode_label,
    get_saved_weather_location,
    reverse_geocode_label,
)


class TestWeatherLocationLabel(unittest.IsolatedAsyncioTestCase):
    def test_formats_city_district_and_short_country_name(self):
        label = _format_reverse_geocode_label(
            {
                "city": "上海市",
                "locality": "闵行区",
                "principalSubdivision": "上海市",
                "countryName": "中华人民共和国",
                "countryCode": "CN",
            }
        )
        self.assertEqual(label, "上海市 · 闵行区 · 中国")

    async def test_reverse_geocoding_failure_is_not_hidden(self):
        client = AsyncMock()
        client.__aenter__.return_value.get.side_effect = httpx.ConnectError("down")
        with patch("services.weather.httpx.AsyncClient", return_value=client):
            with self.assertRaises(httpx.ConnectError):
                await reverse_geocode_label(31.0, 121.0)

    async def test_legacy_placeholder_is_resolved_and_persisted(self):
        raw = json.dumps(
            {
                "latitude": 31.0,
                "longitude": 121.0,
                "label": "当前位置",
                "source": "browser",
            }
        )
        with (
            patch("services.weather.get_setting", new=AsyncMock(return_value=raw)),
            patch(
                "services.weather.reverse_geocode_label",
                new=AsyncMock(return_value="上海市 · 闵行区 · 中国"),
            ),
            patch("services.weather.set_setting", new_callable=AsyncMock) as save,
        ):
            location = await get_saved_weather_location("user-a")

        self.assertEqual(location["label"], "上海市 · 闵行区 · 中国")
        persisted = json.loads(save.await_args.args[1])
        self.assertEqual(persisted["label"], "上海市 · 闵行区 · 中国")


if __name__ == "__main__":
    unittest.main()
