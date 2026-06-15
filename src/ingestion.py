"""Pipeline de indexação: carrega receitas (Documentos), divide em chunks (Text Splitter),
gera Embeddings e persiste no ChromaDB (Vector store)."""

import re
import shutil
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config


def _parse_metadata(text: str, filename: str) -> dict:
    nome_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    categoria_match = re.search(r"##\s*Categoria\s*\n+(.+)", text)
    return {
        "source": filename,
        "nome_prato": nome_match.group(1).strip() if nome_match else filename,
        "categoria": categoria_match.group(1).strip() if categoria_match else "Desconhecida",
    }


def load_documents() -> list[Document]:
    """Carrega os documentos de receitas (.txt/.md e .pdf, se existirem) de data/recipes/."""
    documents = []

    for path in sorted(config.RECIPES_DIR.glob("*.txt")) + sorted(config.RECIPES_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        metadata = _parse_metadata(text, path.name)
        documents.append(Document(page_content=text, metadata=metadata))

    # Suporte opcional a PDFs, mantendo a pipeline alinhada com "Documentos: PDFs, TXTs, Web..."
    for path in sorted(config.RECIPES_DIR.glob("*.pdf")):
        from langchain_community.document_loaders import PyPDFLoader

        for doc in PyPDFLoader(str(path)).load():
            doc.metadata.setdefault("nome_prato", path.stem)
            doc.metadata.setdefault("categoria", "Desconhecida")
            documents.append(doc)

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)


def _cloud_enabled() -> bool:
    return bool(config.CHROMA_API_KEY and config.CHROMA_TENANT)


def _get_chroma_client() -> chromadb.ClientAPI:
    """Devolve um cliente Chroma Cloud (se configurado) ou um cliente local persistido."""
    if _cloud_enabled():
        return chromadb.CloudClient(
            api_key=config.CHROMA_API_KEY,
            tenant=config.CHROMA_TENANT,
            database=config.CHROMA_DATABASE,
        )
    return chromadb.PersistentClient(path=config.CHROMA_DIR)


def build_vectorstore() -> Chroma:
    """Reconstrói o índice Chroma a partir do zero a partir de data/recipes/."""
    documents = load_documents()
    if not documents:
        raise RuntimeError(f"Nenhuma receita encontrada em {config.RECIPES_DIR}")

    chunks = split_documents(documents)

    if _cloud_enabled():
        client = _get_chroma_client()
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass
    else:
        chroma_path = Path(config.CHROMA_DIR)
        if chroma_path.exists():
            shutil.rmtree(chroma_path)
        client = _get_chroma_client()

    return Chroma.from_documents(
        documents=chunks,
        embedding=_get_embeddings(),
        client=client,
        collection_name=config.COLLECTION_NAME,
    )


def load_vectorstore() -> Chroma:
    """Abre o índice Chroma já persistido (local ou Chroma Cloud, usado pela aplicação)."""
    return Chroma(
        client=_get_chroma_client(),
        collection_name=config.COLLECTION_NAME,
        embedding_function=_get_embeddings(),
    )


def index_exists() -> bool:
    """Verifica se o índice Chroma já foi criado (local ou Chroma Cloud)."""
    if _cloud_enabled():
        try:
            client = _get_chroma_client()
            return config.COLLECTION_NAME in [c.name for c in client.list_collections()]
        except Exception:
            return False
    return Path(config.CHROMA_DIR).exists()
