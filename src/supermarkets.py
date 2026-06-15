"""Localizador de supermercados: geocoding (Nominatim) + pesquisa via Overpass API (OpenStreetMap)."""

from __future__ import annotations

import math
import unicodedata

import requests
from geopy.geocoders import Nominatim

from src import config

RECOMMENDED_CHAINS = [
    "continente",
    "pingo doce",
    "lidl",
    "mercadona",
    "auchan",
    "intermarche",
    "minipreco",
    "aldi",
    "leclerc",
]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


def geocode_address(address: str) -> tuple[float, float] | None:
    """Converte um endereço/código postal em (latitude, longitude) via Nominatim."""
    geolocator = Nominatim(user_agent=config.NOMINATIM_USER_AGENT)
    location = geolocator.geocode(address, timeout=10)
    if location is None:
        return None
    return location.latitude, location.longitude


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * earth_radius_km * math.asin(math.sqrt(a))


def _is_recommended(tags: dict) -> bool:
    nome = _normalize((tags.get("name") or "") + " " + (tags.get("brand") or ""))
    return any(chain in nome for chain in RECOMMENDED_CHAINS)


def _format_address(tags: dict) -> str:
    partes = []
    if tags.get("addr:street"):
        rua = tags["addr:street"]
        if tags.get("addr:housenumber"):
            rua += f", {tags['addr:housenumber']}"
        partes.append(rua)
    if tags.get("addr:city"):
        partes.append(tags["addr:city"])
    return ", ".join(partes)


def _query_overpass(query: str) -> list[dict]:
    """Envia a query aos mirrors da Overpass API definidos em config.OVERPASS_URLS,
    tentando o próximo em caso de erro/timeout (ex: 504 do servidor principal)."""
    erro = None
    for url in config.OVERPASS_URLS:
        try:
            response = requests.post(
                url,
                data={"data": query},
                headers={"User-Agent": config.NOMINATIM_USER_AGENT},
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("elements", [])
        except requests.exceptions.RequestException as exc:
            erro = exc
    raise RuntimeError(f"Todos os servidores Overpass falharam: {erro}") from erro


def find_supermarkets(lat: float, lon: float, radius_km: float = config.MAX_RADIUS_KM) -> list[dict]:
    """Procura supermercados num raio (km) à volta de (lat, lon) usando a Overpass API.

    Devolve uma lista ordenada por distância, cada item com:
    nome, distancia_km, lat, lon, endereco, recomendado.
    """
    radius_m = int(radius_km * 1000)
    query = f"""
    [out:json][timeout:25];
    node["shop"="supermarket"](around:{radius_m},{lat},{lon});
    out;
    """
    elements = _query_overpass(query)

    supermercados = []
    for el in elements:
        tags = el.get("tags", {})
        el_lat, el_lon = el.get("lat"), el.get("lon")
        if el_lat is None or el_lon is None:
            continue

        supermercados.append(
            {
                "nome": tags.get("name") or tags.get("brand") or "Supermercado",
                "distancia_km": round(_haversine_km(lat, lon, el_lat, el_lon), 2),
                "lat": el_lat,
                "lon": el_lon,
                "endereco": _format_address(tags),
                "recomendado": _is_recommended(tags),
            }
        )

    supermercados.sort(key=lambda s: s["distancia_km"])
    return supermercados


def filter_by_distance(supermercados: list[dict], max_km: float) -> list[dict]:
    return [s for s in supermercados if s["distancia_km"] <= max_km]
