"""Constrói (ou reconstrói) o índice ChromaDB a partir das receitas em data/recipes/.

Uso:
    python scripts/ingest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.ingestion import _cloud_enabled, build_vectorstore, load_documents


def main() -> None:
    documents = load_documents()
    print(f"A indexar {len(documents)} receita(s) a partir de {config.RECIPES_DIR} ...")

    vectorstore = build_vectorstore()
    count = vectorstore._collection.count()

    if _cloud_enabled():
        destino = f"Chroma Cloud (tenant={config.CHROMA_TENANT}, database={config.CHROMA_DATABASE})"
    else:
        destino = f"'{config.CHROMA_DIR}'"

    print(f"Índice ChromaDB '{config.COLLECTION_NAME}' criado em {destino} com {count} chunk(s).")


if __name__ == "__main__":
    main()
