"""Configuração do LangSmith: ativa o Tracing automático das chains e permite enviar Feedback."""

from __future__ import annotations

import os

from src import config


def setup_langsmith() -> bool:
    """Ativa o tracing do LangSmith (Tracing/Datasets/Dashboard) se houver API key configurada.

    Devolve True se o tracing foi ativado.
    """
    if not config.LANGCHAIN_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = config.LANGCHAIN_PROJECT
    return True


def send_feedback(run_id: str | None, score: int, comment: str | None = None) -> None:
    """Envia feedback (ex: 👍/👎) para o LangSmith, associado ao run_id da geração."""
    if run_id is None or not config.LANGCHAIN_API_KEY:
        return

    from langsmith import Client

    Client().create_feedback(run_id, key="user_rating", score=score, comment=comment)
