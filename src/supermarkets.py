"""Localizador de supermercados: geocoding (Photon) + pesquisa via Overpass API (OpenStreetMap)."""

from __future__ import annotations

import math
import time
import unicodedata

import requests
from geopy.exc import GeocoderServiceError
from geopy.geocoders import Photon
from geopy.point import Point

from src import config

GEOCODE_TENTATIVAS = 3
GEOCODE_ESPERA_SEGUNDOS = 2
# Caixa delimitadora aproximada de Portugal continental. Sem isto, pesquisas ambíguas
# (ex: só um código postal, sem localidade) podem ser resolvidas para fora do país -
# o OpenStreetMap tem fraca cobertura de códigos postais portugueses como entidades
# pesquisáveis, e tanto o Photon como o Nominatim por vezes "adivinham" mal sem este
# limite (ex: "1300-552" sem mais contexto era resolvido para Salt Lake City, EUA).
# Não cobre Açores/Madeira - pesquisas nessas regiões podem ficar imprecisas.
PORTUGAL_BBOX = [Point(42.2, -9.6), Point(36.8, -6.1)]

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

# Nomes (normalizados) de locais mal classificados no OpenStreetMap como shop=supermarket
# (ex: antigos supermercados convertidos noutro tipo de negócio, mas com a tag desatualizada).
EXCLUDED_NAMES = [
    "unilabs",
]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


def geocode_address(address: str) -> tuple[float, float] | None:
    """Converte um endereço/código postal em (latitude, longitude) via Photon (komoot).

    Usa-se o Photon em vez do Nominatim porque a instância pública do Nominatim
    aplica um limite de taxa partilhado por todos os utilizadores de um mesmo IP
    (ex: o IP do Streamlit Cloud), o que causava erros frequentes de rate limit.
    Mesmo assim, tenta-se algumas vezes com um pequeno intervalo antes de desistir.
    """
    geolocator = Photon(user_agent=config.APP_USER_AGENT)
    ultimo_erro: Exception | None = None
    for tentativa in range(GEOCODE_TENTATIVAS):
        try:
            location = geolocator.geocode(address, bbox=PORTUGAL_BBOX, timeout=10)
            return (location.latitude, location.longitude) if location else None
        except GeocoderServiceError as exc:
            ultimo_erro = exc
            if tentativa < GEOCODE_TENTATIVAS - 1:
                time.sleep(GEOCODE_ESPERA_SEGUNDOS * (tentativa + 1))

    raise RuntimeError(
        "Serviço de localização indisponível de momento. Tenta novamente dentro de alguns segundos."
    ) from ultimo_erro


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
                headers={"User-Agent": config.APP_USER_AGENT},
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

        nome = tags.get("name") or tags.get("brand") or "Supermercado"
        if _normalize(nome) in EXCLUDED_NAMES:
            continue

        supermercados.append(
            {
                "nome": nome,
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
