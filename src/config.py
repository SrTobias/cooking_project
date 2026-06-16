"""Configurações centrais da aplicação, carregadas a partir de variáveis de ambiente (.env)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# No Streamlit Community Cloud não existe ficheiro .env: as chaves são configuradas em
# "Settings -> Secrets" (formato TOML) e expostas via st.secrets. Copiamos para os.environ
# para que o resto do código (e scripts fora do Streamlit) continue a usar apenas os.environ.
try:
    import streamlit as st

    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass

# Caminhos
BASE_DIR = Path(__file__).resolve().parent.parent
RECIPES_DIR = BASE_DIR / "data" / "recipes"
PRICES_FILE = BASE_DIR / "data" / "precos_ingredientes.json"
CHROMA_DIR = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "cooking_project"

# Chroma Cloud (opcional) - se CHROMA_API_KEY e CHROMA_TENANT estiverem definidos,
# usa-se a base de dados vetorial alojada na cloud em vez do diretório local CHROMA_DIR.
CHROMA_API_KEY = os.environ.get("CHROMA_API_KEY")
CHROMA_TENANT = os.environ.get("CHROMA_TENANT")
CHROMA_DATABASE = os.environ.get("CHROMA_DATABASE", "cooking_project")

# OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

# LangSmith
LANGCHAIN_TRACING_V2 = os.environ.get("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "cooking_project")

# Retrieval
# Score mínimo de relevância (0-1) para um documento recuperado ser usado como contexto.
RELEVANCE_THRESHOLD = float(os.environ.get("RELEVANCE_THRESHOLD", "0.25"))

# Supermercados
MAX_RADIUS_KM = 1
# Mirrors da Overpass API, tentados por ordem (o principal devolve por vezes 504 sob carga).
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]
NOMINATIM_USER_AGENT = "cooking-rag-app"

# Preços
DEFAULT_PRICE_EUR = 1.50
