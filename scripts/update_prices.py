#!/usr/bin/env python
"""Atualiza precos_ingredientes.json com estimativas recentes via LLM.

Uso:
    python scripts/update_prices.py
    python scripts/update_prices.py --dry-run   # mostra o resultado sem gravar
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openai

from src import config


def main(dry_run: bool = False) -> None:
    with open(config.PRICES_FILE, encoding="utf-8") as f:
        dados = json.load(f)

    ingredientes = {k: v for k, v in dados.items() if not k.startswith("_")}
    nomes = list(ingredientes.keys())

    print(f"A atualizar {len(nomes)} ingredientes via {config.LLM_MODEL}...")

    prompt = (
        "Para cada ingrediente listado abaixo, indica o preço médio atual em supermercados "
        "portugueses (Continente, Pingo Doce, Lidl, Mercadona).\n"
        "Responde APENAS com um objeto JSON com este formato exato:\n"
        '{"<nome do ingrediente>": {"preco": <número em EUR>, "unidade": "<kg|l|unidade>"}, ...}\n\n'
        "Ingredientes:\n" + "\n".join(f"- {n}" for n in nomes)
    )

    client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    resultado: dict = json.loads(response.choices[0].message.content)

    atualizados = 0
    sem_resposta = []
    for nome in nomes:
        novo = resultado.get(nome)
        if (
            novo
            and isinstance(novo.get("preco"), (int, float))
            and novo.get("unidade") in ("kg", "l", "unidade")
        ):
            ingredientes[nome] = novo
            atualizados += 1
        else:
            sem_resposta.append(nome)

    if sem_resposta:
        print(f"  Sem resposta válida para: {', '.join(sem_resposta)} — mantidos valores atuais")

    saida = {
        "_comment": dados.get("_comment", "Preços médios aproximados (EUR) em supermercados portugueses. Usado apenas para estimativas. unidade: kg | l | unidade"),
        "_updated_at": str(date.today()),
        **ingredientes,
    }

    if dry_run:
        print("\n--- DRY RUN (nada foi gravado) ---")
        print(json.dumps(saida, ensure_ascii=False, indent=2))
    else:
        with open(config.PRICES_FILE, "w", encoding="utf-8") as f:
            json.dump(saida, f, ensure_ascii=False, indent=2)
        print(f"Concluído: {atualizados}/{len(nomes)} ingredientes atualizados.")
        print(f"Ficheiro gravado: {config.PRICES_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Mostra o resultado sem gravar o ficheiro")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
