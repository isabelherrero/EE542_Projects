import getpass
import math
import os
import time

import requests

TB_URL = "http://localhost:8080"

PHONE_PREFIX = "Phone_"

# Libraries are stored under Assets.
LIBRARY_ENTITY_TYPE = "ASSET"

session = requests.Session()


def api(method, path, **kwargs):
    response = session.request(
        method,
        TB_URL + path,
        timeout=20,
        **kwargs,
    )
    response.raise_for_status()
    return response


def authenticate():
    api_key = os.getenv("FAIRPLAY_API_KEY", "").strip()

    if api_key:
        session.headers["X-Authorization"] = f"ApiKey {api_key}"
    else:
        # Use the same tenant account you use in the ThingsBoard website.
        email = input(
            "ThingsBoard email [tenant@thingsboard.org]: "
        ).strip() or "tenant@thingsboard.org"

        password = getpass.getpass("ThingsBoard password: ")

        result = api(
            "POST",
            "/api/auth/login",
            json={"username": email, "password": password},
        ).json()

        session.headers["X-Authorization"] = (
            f"Bearer {result['token']}"
        )


def coordinates(lat, lon):
    lat, lon = float(lat), float(lon)

    if not (
        math.isfinite(lat)
        and math.isfinite(lon)
        and -90 <= lat <= 90
        and -180 <= lon <= 180
    ):
        raise ValueError("Invalid latitude or longitude.")

    return lat, lon


def get_phone_devices():
    devices = []
    page = 0

    while True:
        result = api(
            "GET",
            "/api/tenant/devices",
            params={"pageSize": 100, "page": page},
        ).json()

        for device in result.get("data", []):
            name = device.get("name", "")

            if name.startswith(PHONE_PREFIX):
                devices.append(device)

        if not result.get("hasNext", False):
            return devices

        page += 1


def get_people():
    people = {}
    phones = get_phone_devices()

    if not phones:
        raise ValueError(
            f"No phone devices found. Phone device names "
            f"should start with '{PHONE_PREFIX}'."
        )

    for device in phones:
        device_name = device["name"]
        device_id = device["id"]["id"]

        name = device_name[len(PHONE_PREFIX):].strip()

        if not name:
            name = device_name

        data = api(
            "GET",
            f"/api/plugins/telemetry/DEVICE/{device_id}"
            "/values/timeseries",
            params={"keys": "lat,lon"},
        ).json()

        if not data.get("lat") or not data.get("lon"):
            raise ValueError(
                f"{name}: no lat/lon. Send location from OwnTracks."
            )

        people[name] = coordinates(
            data["lat"][0]["value"],
            data["lon"][0]["value"],
        )

        # Show how old the older of the two coordinate readings is.
        timestamp = min(
            data["lat"][0]["ts"],
            data["lon"][0]["ts"],
        )
        age_minutes = max(
            0, (time.time() - timestamp / 1000) / 60
        )
        print(f"{name}: coordinate age = {age_minutes:.1f} minutes")

        if age_minutes > 15:
            print(
                f"  Warning: {name}'s location is old. "
                "Send a new location before recording."
            )

    return people


def get_group_center(people):
    center_lat = (
        sum(location[0] for location in people.values())
        / len(people)
    )

    center_lon = (
        sum(location[1] for location in people.values())
        / len(people)
    )

    return center_lat, center_lon


def get_all_library_entities():
    collection = (
        "assets" if LIBRARY_ENTITY_TYPE == "ASSET" else "devices"
    )
    entities = []
    page = 0

    while True:
        result = api(
            "GET",
            f"/api/tenant/{collection}",
            params={"pageSize": 100, "page": page},
        ).json()

        entities.extend(result.get("data", []))

        if not result.get("hasNext", False):
            return entities

        page += 1


def get_libraries():
    libraries = []

    for entity in get_all_library_entities():
        description = " ".join(
            str(entity.get(key) or "")
            for key in ("name", "label", "type")
        ).lower()

        if "library" not in description:
            continue

        entity_id = entity["id"]["id"]

        rows = api(
            "GET",
            f"/api/plugins/telemetry/{LIBRARY_ENTITY_TYPE}/"
            f"{entity_id}/values/attributes/SERVER_SCOPE",
            params={"keys": "lat,lon"},
        ).json()

        attributes = {
            row["key"]: row["value"] for row in rows
        }

        if "lat" not in attributes or "lon" not in attributes:
            raise ValueError(
                f"{entity['name']}: add lat and lon "
                "under Server attributes."
            )

        lat, lon = coordinates(
            attributes["lat"], attributes["lon"]
        )

        libraries.append({
            "id": entity_id,
            "name": entity["name"],
            "lat": lat,
            "lon": lon,
        })

    if not libraries:
        raise ValueError(
            "No libraries found. Include 'Library' in each "
            "library asset's Name or Label."
        )

    return libraries


def distance_km(point1, point2):
    lat1, lon1 = map(math.radians, point1)
    lat2, lon2 = map(math.radians, point2)

    value = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin((lon2 - lon1) / 2) ** 2
    )

    value = max(0.0, min(1.0, value))
    return 2 * 6371.0 * math.asin(math.sqrt(value))


def rank_libraries(people, libraries, center):
    for library in libraries:
        location = (library["lat"], library["lon"])

        distances = {
            name: distance_km(person_location, location)
            for name, person_location in people.items()
        }

        library["distances"] = distances
        library["maximum"] = max(distances.values())
        library["average"] = (
            sum(distances.values()) / len(distances)
        )
        library["center_distance"] = distance_km(
            center,
            location,
        )

    return sorted(
        libraries,
        key=lambda library: (
            library["center_distance"],
            library["average"],
            library["name"],
        ),
    )


def save_results(ranked):
    calculated_at = int(time.time() * 1000)

    # Clear old recommendation flags first.
    for library in ranked:
        api(
            "POST",
            f"/api/plugins/telemetry/{LIBRARY_ENTITY_TYPE}/"
            f"{library['id']}/SERVER_SCOPE",
            json={"recommended": False},
        )

    # Save the winner last, so incomplete writes do not mark a new winner.
    for index in reversed(range(len(ranked))):
        library = ranked[index]

        attributes = {
            "recommended": index == 0,
            "recommendation_rank": index + 1,
            "center_distance_km": round(
                library["center_distance"], 3
            ),
            "max_distance_km": round(library["maximum"], 3),
            "average_distance_km": round(library["average"], 3),
            "recommendation_updated_at": calculated_at,
        }

        for name, distance in library["distances"].items():
            attributes[f"distance_{name.lower()}_km"] = round(
                distance, 3
            )

        api(
            "POST",
            f"/api/plugins/telemetry/{LIBRARY_ENTITY_TYPE}/"
            f"{library['id']}/SERVER_SCOPE",
            json=attributes,
        )


def main():
    authenticate()
    people = get_people()
    center = get_group_center(people)
    libraries = get_libraries()
    ranked = rank_libraries(people, libraries, center)

    print(
        f"\nGroup center: "
        f"{center[0]:.6f}, {center[1]:.6f}"
    )

    print("\nLibrary ranking (straight-line distance):")

    for number, library in enumerate(ranked, 1):
        print(
            f"{number}. {library['name']} | "
            f"Center: {library['center_distance']:.2f} km | "
            f"Farthest: {library['maximum']:.2f} km | "
            f"Average: {library['average']:.2f} km"
        )

    winner = ranked[0]
    print(f"\nRECOMMENDED: {winner['name']}")

    for name, distance in winner["distances"].items():
        print(
            f"  {name}: {distance:.2f} km "
            f"({distance * 0.621371:.2f} miles)"
        )

    save_results(ranked)

    print("\nResults saved to library Server attributes.")
    print("Winner: recommended = true")
    print("Others: recommended = false")
    print("Run this program again to recalculate.")


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as error:
        status = error.response.status_code
        print(f"\nThingsBoard API error: HTTP {status}")
        print(
            "Check your tenant login, phone devices, and "
            "whether libraries are Assets."
        )
        print("If saving failed, fix the issue and rerun.")
    except requests.RequestException:
        print(
            "\nCannot reach ThingsBoard. "
            "Check that its service is running."
        )
    except (ValueError, KeyError, IndexError, TypeError) as error:
        print(f"\nData error: {error}")