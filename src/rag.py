"""Pipeline de query: Pergunta -> Retriever (Chroma) -> LLM (OpenAI) -> Resposta (FichaTecnica).

Implementa as 3 opções de interação descritas no enunciado:
  1. gerar_opcao1: ingredientes disponíveis em casa -> sugestão de prato
  2. gerar_opcao2: nome do prato desejado -> ficha técnica desse prato
  3. gerar_opcao3: a IA sugere um prato (opcionalmente filtrado por tipo de refeição)
"""

from __future__ import annotations

import random

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.context import collect_runs
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src import config
from src.ingestion import add_recipe_document, load_vectorstore, recipe_exists

TIPOS_REFEICAO = ["Qualquer", "Sopa", "Prato principal", "Sobremesa"]

CATEGORIA_FILTROS = {
    "Sopa": ["Sopa"],
    "Prato principal": [
        "Prato principal - Peixe",
        "Prato principal - Carne",
        "Prato principal - Vegetariano",
    ],
    "Sobremesa": ["Sobremesa"],
}


class Ingrediente(BaseModel):
    nome: str = Field(description="Nome do ingrediente, sem quantidade (ex: 'Cebola')")
    quantidade: str = Field(
        description="Quantidade necessária, incluindo a unidade (ex: '400 g', '2 unidades', '1 l', 'q.b.')"
    )


class FichaTecnica(BaseModel):
    nome_prato: str = Field(description="Nome do prato sugerido")
    categoria: str = Field(description="Categoria do prato (ex: Sopa, Prato principal - Peixe, Sobremesa)")
    porcoes: int = Field(description="Número de porções/pessoas para as quais a receita foi ajustada")
    tempo_preparacao: str = Field(description="Tempo estimado de preparação")
    ingredientes: list[Ingrediente] = Field(
        description="Lista completa de ingredientes, com quantidades já ajustadas ao número de porções pedido"
    )
    modo_preparacao: list[str] = Field(description="Passos do modo de preparação, um por item, sem numeração")
    ingredientes_em_falta: list[Ingrediente] = Field(
        description="Lista de ingredientes que é necessário comprar, com quantidades ajustadas"
    )


SYSTEM_PROMPT = """Tu és um chef de cozinha português especialista em adaptar receitas tradicionais.
Usa o CONTEXTO abaixo (receitas da base de dados) como inspiração e base de conhecimento principal.

Regras importantes:
- A ficha técnica final deve ter as quantidades dos ingredientes ajustadas/escaladas
  proporcionalmente para {pessoas} pessoas (o contexto indica normalmente "Porções base: 4 pessoas" -
  usa essa proporção como referência para escalar).
- Os nomes dos ingredientes ("nome") devem estar limpos, sem quantidade (ex: "Cebola", nunca "1 cebola").
- As quantidades ("quantidade") devem incluir a unidade (ex: "400 g", "2 unidades", "1 l", "q.b.").
- O campo "modo_preparacao" é uma lista de passos claros, cada um sem numeração.
- O campo "porcoes" deve ser igual a {pessoas}.
"""

TASK_OPCAO1 = """O utilizador tem os seguintes ingredientes disponíveis em casa:
{ingredientes_disponiveis}

CONTEXTO (receitas da base de dados):
{context}

Escolhe, de entre o contexto, o prato que melhor aproveita estes ingredientes (podes adaptar
ligeiramente a receita). Gera a ficha técnica completa para {pessoas} pessoas.

No campo "ingredientes_em_falta", lista APENAS os ingredientes da receita que o utilizador
NÃO tem em casa (compara com a lista de ingredientes disponíveis acima) e que precisa de
comprar, com as quantidades já ajustadas a {pessoas} pessoas."""

TASK_OPCAO2 = """O utilizador quer preparar o seguinte prato: "{nome_prato}"

CONTEXTO (receitas da base de dados):
{context}

Se o prato pedido corresponder (ou for muito semelhante) a uma receita do contexto, usa-a como
base. Caso contrário, gera a ficha técnica com o teu conhecimento culinário, mantendo o mesmo
formato. Gera a ficha técnica completa para {pessoas} pessoas.

No campo "ingredientes_em_falta", lista TODOS os ingredientes da receita (lista de compras
completa), com as quantidades já ajustadas a {pessoas} pessoas."""

TASK_OPCAO3 = """O utilizador pede uma sugestão de prato{tipo_refeicao_txt}.

CONTEXTO (receitas da base de dados):
{context}

Escolhe UM prato do contexto para sugerir ao utilizador. Gera a ficha técnica completa para
{pessoas} pessoas.

No campo "ingredientes_em_falta", lista TODOS os ingredientes da receita (lista de compras
completa), com as quantidades já ajustadas a {pessoas} pessoas."""


def retrieve_with_likes(query: str, k: int, fetch_k: int | None = None) -> list[Document]:
    """Recupera os fetch_k documentos mais semelhantes, descarta os pouco relevantes para a
    query (score < RELEVANCE_THRESHOLD) e reordena os restantes pelos likes líquidos
    (likes - dislikes), para que receitas mais bem avaliadas apareçam primeiro no contexto."""
    vectorstore = load_vectorstore()
    pares = vectorstore.similarity_search_with_relevance_scores(query, k=fetch_k or k * 2)

    candidatos = [doc for doc, score in pares if score >= config.RELEVANCE_THRESHOLD]
    if not candidatos and pares:
        candidatos = [pares[0][0]]

    candidatos.sort(key=lambda d: d.metadata.get("likes", 0) - d.metadata.get("dislikes", 0), reverse=True)
    return candidatos[:k]


def _format_docs(docs) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=config.LLM_MODEL, temperature=0.4)


def _build_chain(task_template: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", task_template),
        ]
    )
    return prompt | _get_llm().with_structured_output(FichaTecnica)


def _invoke_with_tracing(chain, inputs: dict) -> tuple[FichaTecnica, str | None]:
    with collect_runs() as cb:
        result = chain.invoke(inputs)
    run_id = str(cb.traced_runs[0].id) if cb.traced_runs else None
    return result, run_id


def _ficha_to_document(ficha: FichaTecnica) -> Document:
    linhas = [
        f"# {ficha.nome_prato}",
        "",
        "## Categoria",
        ficha.categoria,
        "",
        "## Porções base",
        f"{ficha.porcoes} pessoas",
        "",
        "## Tempo de preparação",
        ficha.tempo_preparacao,
        "",
        "## Ingredientes",
        *(f"- {ing.quantidade} de {ing.nome}" for ing in ficha.ingredientes),
        "",
        "## Modo de preparação",
        *(f"{idx}. {passo}" for idx, passo in enumerate(ficha.modo_preparacao, start=1)),
    ]
    return Document(
        page_content="\n".join(linhas) + "\n",
        metadata={"nome_prato": ficha.nome_prato, "categoria": ficha.categoria},
    )


def _guardar_se_nova(ficha: FichaTecnica) -> None:
    """Se o prato sugerido/pedido ainda não constar da coleção, adiciona-o ao índice Chroma."""
    try:
        if not recipe_exists(ficha.nome_prato):
            add_recipe_document(_ficha_to_document(ficha))
    except Exception:
        pass


def gerar_opcao1(ingredientes_disponiveis: list[str], pessoas: int) -> tuple[FichaTecnica, str | None]:
    query = "Receita que utilize estes ingredientes: " + ", ".join(ingredientes_disponiveis)
    docs = retrieve_with_likes(query, k=4)

    chain = _build_chain(TASK_OPCAO1)
    inputs = {
        "ingredientes_disponiveis": "\n".join(f"- {item}" for item in ingredientes_disponiveis),
        "context": _format_docs(docs),
        "pessoas": pessoas,
    }
    ficha, run_id = _invoke_with_tracing(chain, inputs)
    _guardar_se_nova(ficha)
    return ficha, run_id


def gerar_opcao2(nome_prato: str, pessoas: int) -> tuple[FichaTecnica, str | None]:
    docs = retrieve_with_likes(nome_prato, k=3)

    chain = _build_chain(TASK_OPCAO2)
    inputs = {
        "nome_prato": nome_prato,
        "context": _format_docs(docs),
        "pessoas": pessoas,
    }
    ficha, run_id = _invoke_with_tracing(chain, inputs)
    _guardar_se_nova(ficha)
    return ficha, run_id


def _sample_context(tipo_refeicao: str | None, amostra: int = 5) -> str:
    vectorstore = load_vectorstore()
    data = vectorstore.get(include=["metadatas", "documents"])
    pares = list(zip(data["documents"], data["metadatas"]))

    if tipo_refeicao and tipo_refeicao != "Qualquer":
        categorias_validas = set(CATEGORIA_FILTROS.get(tipo_refeicao, []))
        filtrados = [p for p in pares if p[1].get("categoria") in categorias_validas]
        if filtrados:
            pares = filtrados

    escolhidos = random.sample(pares, k=min(amostra, len(pares)))
    return "\n\n---\n\n".join(doc for doc, _ in escolhidos)


def gerar_opcao3(pessoas: int, tipo_refeicao: str | None = None) -> tuple[FichaTecnica, str | None]:
    context = _sample_context(tipo_refeicao)
    tipo_txt = f" do tipo '{tipo_refeicao}'" if tipo_refeicao and tipo_refeicao != "Qualquer" else ""

    chain = _build_chain(TASK_OPCAO3)
    inputs = {
        "context": context,
        "pessoas": pessoas,
        "tipo_refeicao_txt": tipo_txt,
    }
    ficha, run_id = _invoke_with_tracing(chain, inputs)
    _guardar_se_nova(ficha)
    return ficha, run_id
