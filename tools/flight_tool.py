"""Flight lookup tool backed by the Aviationstack API.

Reads AVIATIONSTACK_API_KEY from the environment (.env) and returns up to
5 flights, each split into airline, departure, arrival and status.
"""

import os
from datetime import datetime
from zoneinfo import ZoneInfo  # needs the "tzdata" package on Windows

import requests
from dotenv import load_dotenv

load_dotenv()

# Note: Aviationstack's free plan only supports HTTP, not HTTPS.
AVIATIONSTACK_URL = "http://api.aviationstack.com/v1/flights"
MAX_RESULTS = 5


def split_datetime(iso: str | None) -> tuple[str | None, str | None]:
    """'2026-11-11T16:25:00+00:00' -> ('Wed, 11 Nov 2026', '16:25').

    Aviationstack returns each airport's LOCAL time but labels it +00:00,
    so we read the wall-clock date/time as-is and never convert time zones.
    """
    if not iso:
        return None, None
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return iso, None
    return dt.strftime("%a, %d %b %Y"), dt.strftime("%H:%M")


def local_to_aware(iso: str | None, tz: str | None) -> datetime | None:
    """Treat Aviationstack's wall-clock time as local to the airport's time zone."""
    if not iso or not tz:
        return None
    try:
        return datetime.fromisoformat(iso).replace(tzinfo=ZoneInfo(tz))
    except Exception:
        return None


def trip_timing(dep: dict, arr: dict) -> tuple[int | None, int]:
    """(duration in minutes, days between departure and arrival dates)."""
    start = local_to_aware(dep.get("scheduled"), dep.get("timezone"))
    end = local_to_aware(arr.get("scheduled"), arr.get("timezone"))
    duration = int((end - start).total_seconds() // 60) if start and end else None
    if duration is not None and duration <= 0:
        duration = None  # bad data; better to show nothing than a wrong number
    day_offset = 0
    try:
        day_offset = (
            datetime.fromisoformat(arr["scheduled"]).date()
            - datetime.fromisoformat(dep["scheduled"]).date()
        ).days
    except Exception:
        pass
    return duration, day_offset


def build_leg(leg: dict) -> dict:
    """Departure/arrival details with the date and time split out."""
    date, time = split_datetime(leg.get("scheduled"))
    _, est_time = split_datetime(leg.get("estimated"))
    _, actual_time = split_datetime(leg.get("actual"))
    return {
        "airport": leg.get("airport"),
        "iata": leg.get("iata"),
        "terminal": leg.get("terminal"),
        "gate": leg.get("gate"),
        "date": date,                # e.g. "Wed, 11 Nov 2026" (local)
        "time": time,                # e.g. "16:25" (local)
        "timezone": leg.get("timezone"),
        "estimated_time": est_time,
        "actual_time": actual_time,
        "scheduled": leg.get("scheduled"),
        "estimated": leg.get("estimated"),
        "delay_min": leg.get("delay"),
    }


def get_flights(dep_iata: str, arr_iata: str, limit: int = MAX_RESULTS) -> list[dict]:
    """Get flights between two airports.

    Args:
        dep_iata: Departure airport IATA code, e.g. "JFK".
        arr_iata: Arrival airport IATA code, e.g. "LAX".
        limit: Max number of flights to return (capped at 5).

    Returns:
        A list of up to 5 dicts with keys: airline, departure, arrival, status.
        On failure, a single-item list with an "error" key.
    """
    api_key = os.getenv("AVIATIONSTACK_API_KEY")
    if not api_key:
        return [{"error": "AVIATIONSTACK_API_KEY is not set in the environment."}]

    limit = max(1, min(limit, MAX_RESULTS))
    params = {
        "access_key": api_key,
        "dep_iata": dep_iata.upper(),
        "arr_iata": arr_iata.upper(),
        "limit": limit,
    }

    try:
        response = requests.get(AVIATIONSTACK_URL, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return [{"error": f"Request to Aviationstack failed: {exc}"}]
    except ValueError:
        return [{"error": "Aviationstack returned a non-JSON response."}]

    # Aviationstack reports API errors (bad key, quota, etc.) in the body.
    if "error" in payload:
        err = payload["error"]
        return [{"error": f"{err.get('code')}: {err.get('message')}"}]

    results = []
    for flight in payload.get("data") or []:
        airline = flight.get("airline") or {}
        flight_info = flight.get("flight") or {}
        dep = flight.get("departure") or {}
        arr = flight.get("arrival") or {}
        codeshared = flight_info.get("codeshared") or {}
        duration_min, day_offset = trip_timing(dep, arr)

        results.append(
            {
                "airline": {
                    "name": airline.get("name"),
                    "iata": airline.get("iata"),
                    "flight_number": flight_info.get("iata") or flight_info.get("number"),
                    # Codeshare: this ticket is sold by `name` but flown by another airline
                    "operated_by": codeshared.get("airline_name"),
                    "operating_flight": (codeshared.get("flight_iata") or "").upper() or None,
                },
                "flight_date": flight.get("flight_date"),
                "departure": build_leg(dep),
                "arrival": build_leg(arr),
                "duration_min": duration_min,   # e.g. 220 -> "3h 40m"
                "day_offset": day_offset,       # 1 -> lands the next day ("+1")
                "stops": 0,                     # each Aviationstack record is one direct flight
                "status": flight.get("flight_status"),
            }
        )

        if len(results) >= limit:
            break

    return results


if __name__ == "__main__":
    import json

    print(json.dumps(get_flights("JFK", "LAX"), indent=2))
