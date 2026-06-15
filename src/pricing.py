"""Estimativa do custo da lista de compras com base numa tabela de preços médios (EUR)."""

from __future__ import annotations

import difflib
import json
import re
import unicodedata

from src import config
from src.rag import Ingrediente

with open(config.PRICES_FILE, encoding="utf-8") as f:
    _PRECOS = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

_PRECO_KEYS_BY_LENGTH = sorted(_PRECOS.keys(), key=len, reverse=True)

# Conversão de unidades reconhecidas para a unidade-base usada na tabela de preços
# (kg, l ou unidade). O fator converte a quantidade dada para essa unidade-base.
_UNIT_CONVERSIONS: dict[str, tuple[str, float]] = {
    "g": ("kg", 0.001),
    "gr": ("kg", 0.001),
    "grama": ("kg", 0.001),
    "gramas": ("kg", 0.001),
    "kg": ("kg", 1.0),
    "kilo": ("kg", 1.0),
    "ml": ("l", 0.001),
    "l": ("l", 1.0),
    "litro": ("l", 1.0),
    "litros": ("l", 1.0),
    "unidade": ("unidade", 1.0),
    "unidades": ("unidade", 1.0),
    "un": ("unidade", 1.0),
    "dente": ("kg", 0.005),
    "dentes": ("kg", 0.005),
    "fatia": ("unidade", 0.1),
    "fatias": ("unidade", 0.1),
    "folha": ("unidade", 1.0),
    "folhas": ("unidade", 1.0),
    "molho": ("unidade", 1.0),
    "molhos": ("unidade", 1.0),
    "pacote": ("unidade", 1.0),
    "pacotes": ("unidade", 1.0),
    "colher": ("kg", 0.015),
    "colheres": ("kg", 0.015),
    "duzia": ("unidade", 12.0),
    "duzias": ("unidade", 12.0),
}

# Para quantidades dadas como "unidade" (ex: "1 cebola", "2 cenouras") mas cujo preço
# está tabelado por kg/l, assume-se um peso/volume médio por peça.
PESO_UNIDADE_KG = 0.15
VOLUME_UNIDADE_L = 0.2


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower().strip()


def parse_quantidade(quantidade: str) -> tuple[float, str]:
    """Converte uma string de quantidade (ex: "400 g", "2 dentes") em (valor, unidade),
    em que unidade está em {"kg", "l", "unidade"}.

    "q.b." ou texto sem número devolve (0.0, "unidade") - não é orçamentado.
    """
    texto = _normalize(quantidade)
    if "q.b" in texto or "a gosto" in texto or "quanto baste" in texto:
        return 0.0, "unidade"

    match = re.search(r"([\d.,]+)\s*([a-z]*)", texto)
    if not match:
        return 1.0, "unidade"

    valor_str, unidade_str = match.groups()
    try:
        valor = float(valor_str.replace(",", "."))
    except ValueError:
        valor = 1.0

    for prefixo, (unidade_destino, fator) in _UNIT_CONVERSIONS.items():
        if unidade_str.startswith(prefixo):
            return valor * fator, unidade_destino

    return valor, "unidade"


def match_preco(nome_ingrediente: str) -> tuple[dict, str | None]:
    """Encontra a entrada de preço mais próxima para um ingrediente.

    Devolve (entrada_de_preco, chave_correspondida). Se nada corresponder, devolve um
    preço genérico e chave None.
    """
    nome = _normalize(nome_ingrediente)

    for chave in _PRECO_KEYS_BY_LENGTH:
        if chave in nome:
            return _PRECOS[chave], chave

    for palavra in nome.split():
        candidatos = difflib.get_close_matches(palavra, _PRECOS.keys(), n=1, cutoff=0.8)
        if candidatos:
            return _PRECOS[candidatos[0]], candidatos[0]

    return {"preco": config.DEFAULT_PRICE_EUR, "unidade": "unidade"}, None


def _calcular_custo(valor: float, unidade_pedida: str, entrada: dict) -> float:
    if valor == 0.0:
        return 0.0
    if unidade_pedida == entrada["unidade"]:
        return valor * entrada["preco"]
    if entrada["unidade"] == "kg" and unidade_pedida == "unidade":
        return valor * entrada["preco"] * PESO_UNIDADE_KG
    if entrada["unidade"] == "l" and unidade_pedida == "unidade":
        return valor * entrada["preco"] * VOLUME_UNIDADE_L
    # Unidades incompatíveis: assume-se o custo de uma embalagem/unidade do produto
    return entrada["preco"]


def estimar_custo(ingredientes: list[Ingrediente]) -> tuple[float, list[dict]]:
    """Estima o custo total (EUR) de uma lista de ingredientes.

    Devolve (total, detalhe) onde detalhe é uma lista com o custo estimado por ingrediente.
    """
    detalhe = []
    total = 0.0

    for ingrediente in ingredientes:
        valor, unidade_pedida = parse_quantidade(ingrediente.quantidade)
        entrada, chave = match_preco(ingrediente.nome)
        custo = round(_calcular_custo(valor, unidade_pedida, entrada), 2)

        total += custo
        detalhe.append(
            {
                "nome": ingrediente.nome,
                "quantidade": ingrediente.quantidade,
                "preco_estimado": custo,
                "correspondencia": chave or "(estimativa genérica)",
            }
        )

    return round(total, 2), detalhe
