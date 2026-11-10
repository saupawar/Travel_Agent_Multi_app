"""Flight lookup tool backed by the Aviationstack API.

Reads AVIATIONSTACK_API_KEY from the environment (.env) and returns up to
5 flights, each split into airline, departure, arrival and status.
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

# Note: Aviationstack's free plan only supports HTTP, not HTTPS.
AVIATIONSTACK_URL = "http://api.aviationstack.com/v1/flights"
MAX_RESULTS = 5


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

        results.append(
            {
                "airline": {
                    "name": airline.get("name"),
                    "iata": airline.get("iata"),
                    "flight_number": flight_info.get("iata") or flight_info.get("number"),
                },
                "departure": {
                    "airport": dep.get("airport"),
                    "iata": dep.get("iata"),
                    "terminal": dep.get("terminal"),
                    "gate": dep.get("gate"),
                    "scheduled": dep.get("scheduled"),
                    "estimated": dep.get("estimated"),
                    "delay_min": dep.get("delay"),
                },
                "arrival": {
                    "airport": arr.get("airport"),
                    "iata": arr.get("iata"),
                    "terminal": arr.get("terminal"),
                    "gate": arr.get("gate"),
                    "scheduled": arr.get("scheduled"),
                    "estimated": arr.get("estimated"),
                    "delay_min": arr.get("delay"),
                },
                "status": flight.get("flight_status"),
            }
        )

        if len(results) >= limit:
            break

    return results


if __name__ == "__main__":
    import json

    print(json.dumps(get_flights("JFK", "LAX"), indent=2))
