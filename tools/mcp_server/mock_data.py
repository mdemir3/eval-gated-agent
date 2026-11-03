"""Deterministic in-memory fake API behind the `mock_api_get` tool.

All records are invented demo data (placeholder names, fake orders, fake
weather). Two endpoints exist only to exercise failure handling:

* ``/tracking/{order_id}`` is flaky: the first request for each order in a
  server session fails with a 503, and the next request succeeds. It proves
  the Researcher's retry-once logic.
* ``/inventory/{sku}`` always fails with a 500. It proves the Researcher's
  fallback path and the agent's refusal to guess.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

USERS: dict[int, dict[str, Any]] = {
    1: {"id": 1, "name": "Alex Demo", "city": "Paris", "member_since": 2019},
    2: {"id": 2, "name": "Sam Demo", "city": "Tokyo", "member_since": 2021},
    3: {"id": 3, "name": "Riley Demo", "city": "Ottawa", "member_since": 2020},
    4: {"id": 4, "name": "Jordan Demo", "city": "Nairobi", "member_since": 2022},
}

ORDERS: dict[int, dict[str, Any]] = {
    1001: {"id": 1001, "user_id": 2, "item": "telescope", "status": "shipped"},
    1002: {"id": 1002, "user_id": 1, "item": "telephone", "status": "processing"},
    1003: {"id": 1003, "user_id": 3, "item": "light bulb", "status": "delivered"},
    1004: {"id": 1004, "user_id": 4, "item": "printing press model", "status": "cancelled"},
}

WEATHER: dict[str, dict[str, Any]] = {
    "paris": {"city": "Paris", "condition": "cloudy", "temperature_c": 14},
    "tokyo": {"city": "Tokyo", "condition": "sunny", "temperature_c": 22},
    "ottawa": {"city": "Ottawa", "condition": "windy", "temperature_c": 9},
    "cairo": {"city": "Cairo", "condition": "clear", "temperature_c": 31},
    "oslo": {"city": "Oslo", "condition": "snow", "temperature_c": -2},
    "nairobi": {"city": "Nairobi", "condition": "partly cloudy", "temperature_c": 24},
}

TRACKING: dict[int, dict[str, Any]] = {
    1001: {"order_id": 1001, "carrier": "Demo Freight", "current_city": "Paris", "status": "in transit"},
    1002: {"order_id": 1002, "carrier": "Demo Freight", "current_city": "Oslo", "status": "awaiting pickup"},
}

# Human-readable endpoint catalogue; also used in the MCP tool description.
ENDPOINTS = {
    "/users/{id}": "user profile: name, home city, member_since",
    "/orders/{id}": "order: item, status, user_id",
    "/weather/{city}": "current weather for a city: condition, temperature_c",
    "/tracking/{order_id}": "shipment tracking for an order: carrier, current_city, status",
    "/inventory/{sku}": "stock level for a product SKU",
}


class MockAPIError(Exception):
    """An HTTP-style failure from the fake API."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"{status} {message}")
        self.status = status
        self.message = message


def _slug(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[\s_-]+", " ", value).strip().lower()


def _int_id(raw: str, kind: str) -> int:
    if not raw.isdigit():
        raise MockAPIError(400, f"{kind} id must be an integer, got {raw!r}")
    return int(raw)


def resolve_path(endpoint: str, params: dict[str, Any] | None = None) -> str:
    """Fill ``{name}`` placeholders from params, e.g. ('/users/{id}', {'id': 3}) -> '/users/3'."""
    params = params or {}

    def fill(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in params:
            raise MockAPIError(400, f"missing path parameter {key!r}")
        return str(params[key])

    path = re.sub(r"\{(\w+)\}", fill, endpoint.strip())
    return path if path.startswith("/") else "/" + path


_ROUTES = {template.split("/")[1]: template for template in ENDPOINTS}


def route_of(path: str) -> str | None:
    """Return the endpoint template a concrete path matches, or None."""
    segments = path.strip("/").split("/")
    if len(segments) == 2 and segments[1]:
        return _ROUTES.get(segments[0])
    return None


class MockAPI:
    """Stateful only for the flaky endpoint; everything else is a pure lookup."""

    def __init__(self) -> None:
        self._tracking_attempts: dict[int, int] = {}
        self._handlers: dict[str, Callable[[str], dict[str, Any]]] = {
            "users": self._user,
            "orders": self._order,
            "weather": self._weather,
            "tracking": self._tracking,
            "inventory": self._inventory,
        }

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        path = resolve_path(endpoint, params)
        if route_of(path) is None:
            raise MockAPIError(404, f"unknown endpoint {path!r}; known: {', '.join(ENDPOINTS)}")
        resource, arg = path.strip("/").split("/")
        return self._handlers[resource](arg)

    def _user(self, raw_id: str) -> dict[str, Any]:
        user = USERS.get(_int_id(raw_id, "user"))
        if user is None:
            raise MockAPIError(404, f"user {raw_id} not found")
        return dict(user)

    def _order(self, raw_id: str) -> dict[str, Any]:
        order = ORDERS.get(_int_id(raw_id, "order"))
        if order is None:
            raise MockAPIError(404, f"order {raw_id} not found")
        return dict(order)

    def _weather(self, city: str) -> dict[str, Any]:
        record = WEATHER.get(_slug(city))
        if record is None:
            raise MockAPIError(404, f"no weather data for city {city!r}")
        return dict(record)

    def _tracking(self, raw_id: str) -> dict[str, Any]:
        order_id = _int_id(raw_id, "order")
        if order_id not in TRACKING:
            raise MockAPIError(404, f"no tracking for order {raw_id}")
        attempts = self._tracking_attempts.get(order_id, 0) + 1
        self._tracking_attempts[order_id] = attempts
        if attempts == 1:
            raise MockAPIError(503, "tracking service timed out (transient, retry)")
        return dict(TRACKING[order_id])

    def _inventory(self, sku: str) -> dict[str, Any]:
        raise MockAPIError(500, "inventory service unavailable")
