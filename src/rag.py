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
from src.ingestion import add_recipe_document, load_vectorstore, recipe_exists, tempo_para_minutos

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
{restricao_dieta}{restricao_tempo}
CONTEXTO (receitas da base de dados):
{context}

Escolhe, de entre o contexto, o prato que melhor aproveita estes ingredientes (podes adaptar
ligeiramente a receita). Gera a ficha técnica completa para {pessoas} pessoas.

No campo "ingredientes_em_falta", lista APENAS os ingredientes da receita que o utilizador
NÃO tem em casa e que precisa de comprar, com as quantidades já ajustadas a {pessoas} pessoas.
IMPORTANTE: o utilizador tem SOMENTE os ingredientes listados acima e NADA MAIS — não assumas
que tem ingredientes básicos como cebola, alho, azeite, sal ou pimenta a não ser que estejam
na lista. Ingredientes com quantidade "q.b." não devem aparecer nesta lista."""

TASK_OPCAO2 = """O utilizador quer preparar o seguinte prato: "{nome_prato}"
{restricao_tempo}
CONTEXTO (receitas da base de dados):
{context}

Usa uma receita do contexto como base SOMENTE se for o MESMO prato que "{nome_prato}" (mesmo
nome, sinónimo direto ou variação ortográfica/regional do mesmo nome). Pratos apenas parecidos
ou da mesma categoria (ex: outro prato de carne ao forno) NÃO contam como o mesmo prato.

Se nenhuma receita do contexto corresponder ao mesmo prato, ignora os ingredientes e o modo de
preparação do contexto e gera a ficha técnica de "{nome_prato}" a partir do teu conhecimento
culinário geral sobre este prato, mantendo apenas o formato/estilo das fichas técnicas do
contexto. Gera a ficha técnica completa para {pessoas} pessoas.

No campo "ingredientes_em_falta", lista TODOS os ingredientes da receita (lista de compras
completa), com as quantidades já ajustadas a {pessoas} pessoas."""

TASK_OPCAO3 = """O utilizador pede uma sugestão de prato{tipo_refeicao_txt}.
{restricao_dieta}{restricao_tempo}
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
        metadata={
            "nome_prato": ficha.nome_prato,
            "categoria": ficha.categoria,
            "tempo_preparacao": tempo_para_minutos(ficha.tempo_preparacao),
        },
    )


def _guardar_se_nova(ficha: FichaTecnica) -> None:
    """Se o prato sugerido/pedido ainda não constar da coleção, adiciona-o ao índice Chroma."""
    try:
        if not recipe_exists(ficha.nome_prato):
            add_recipe_document(_ficha_to_document(ficha))
    except Exception:
        pass


def _restricao_tempo_txt(tempo_max: int | None) -> str:
    if tempo_max:
        return f"Restrição de tempo: o prato deve ter um tempo de confeção máximo de {tempo_max} minutos.\n"
    return ""


def _restricao_dieta_txt(dieta: str | None) -> str:
    if dieta == "🥗 Vegetariano":
        return "Restrição alimentar: o utilizador é vegetariano — sugere apenas pratos sem carne nem peixe.\n"
    if dieta == "🌱 Vegan":
        return "Restrição alimentar: o utilizador é vegan — sugere apenas pratos sem qualquer produto de origem animal (carne, peixe, ovos, lacticínios, mel).\n"
    return ""


def gerar_opcao1(ingredientes_disponiveis: list[str], pessoas: int, dieta: str | None = None, tempo_max: int | None = None) -> tuple[FichaTecnica, str | None]:
    _validar_ingredientes_culinarios(ingredientes_disponiveis)
    query = "Receita que utilize estes ingredientes: " + ", ".join(ingredientes_disponiveis)
    docs = retrieve_with_likes(query, k=4)

    chain = _build_chain(TASK_OPCAO1)
    inputs = {
        "ingredientes_disponiveis": "\n".join(f"- {item}" for item in ingredientes_disponiveis),
        "context": _format_docs(docs),
        "pessoas": pessoas,
        "restricao_dieta": _restricao_dieta_txt(dieta),
        "restricao_tempo": _restricao_tempo_txt(tempo_max),
    }
    ficha, run_id = _invoke_with_tracing(chain, inputs)
    _guardar_se_nova(ficha)
    return ficha, run_id


def _validar_ingredientes_culinarios(ingredientes: list[str]) -> None:
    """Lança ValueError se algum dos ingredientes não for um alimento ou ingrediente de cozinha."""
    llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0, max_tokens=5)
    lista = ", ".join(f'"{i}"' for i in ingredientes)
    resp = llm.invoke([
        {"role": "user", "content": (
            f"Os seguintes itens são TODOS alimentos ou ingredientes de culinária? {lista} "
            f"Responde APENAS com \"sim\" ou \"não\"."
        )}
    ])
    if "sim" not in resp.content.strip().lower():
        raise ValueError(
            "A lista de ingredientes contém itens que não são alimentos ou ingredientes de culinária. "
            "Indica apenas ingredientes de cozinha (ex: arroz, frango, cebola)."
        )


def _validar_prato_culinario(nome: str) -> None:
    """Lança ValueError se o nome não for um prato ou receita de culinária."""
    llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0, max_tokens=3)
    resp = llm.invoke([
        {"role": "user", "content": (
            f'"{nome}" é um prato de culinária ou receita de cozinha? '
            f'Responde APENAS com "sim" ou "não".'
        )}
    ])
    if "sim" not in resp.content.strip().lower():
        raise ValueError(f'"{nome}" não parece ser um prato de culinária. Indica o nome de uma receita ou prato.')


def gerar_opcao2(nome_prato: str, pessoas: int, tempo_max: int | None = None) -> tuple[FichaTecnica, str | None]:
    _validar_prato_culinario(nome_prato)
    docs = retrieve_with_likes(nome_prato, k=3)

    chain = _build_chain(TASK_OPCAO2)
    inputs = {
        "nome_prato": nome_prato,
        "context": _format_docs(docs),
        "pessoas": pessoas,
        "restricao_tempo": _restricao_tempo_txt(tempo_max),
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


def gerar_opcao3(pessoas: int, tipo_refeicao: str | None = None, dieta: str | None = None, tempo_max: int | None = None) -> tuple[FichaTecnica, str | None]:
    context = _sample_context(tipo_refeicao)
    tipo_txt = f" do tipo '{tipo_refeicao}'" if tipo_refeicao and tipo_refeicao != "Qualquer" else ""

    chain = _build_chain(TASK_OPCAO3)
    inputs = {
        "context": context,
        "pessoas": pessoas,
        "tipo_refeicao_txt": tipo_txt,
        "restricao_dieta": _restricao_dieta_txt(dieta),
        "restricao_tempo": _restricao_tempo_txt(tempo_max),
    }
    ficha, run_id = _invoke_with_tracing(chain, inputs)
    _guardar_se_nova(ficha)
    return ficha, run_id
