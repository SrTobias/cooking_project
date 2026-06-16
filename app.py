"""Assistente de Cozinha RAG - Streamlit UI.

Pipeline de query: Pergunta -> Retriever (Chroma) -> LLM (OpenAI) -> Resposta, com
tracing/feedback via LangSmith. Inclui também a lista de supermercados próximos e
estimativa de custo da lista de compras.
"""

from __future__ import annotations

import streamlit as st
from streamlit_geolocation import streamlit_geolocation

from src import config
from src.ingestion import index_exists, update_recipe_feedback
from src.observability import send_feedback, setup_langsmith
from src.pricing import estimar_custo
from src.rag import TIPOS_REFEICAO, FichaTecnica, gerar_opcao1, gerar_opcao2, gerar_opcao3
from src.supermarkets import filter_by_distance, find_supermarkets, geocode_address

st.set_page_config(page_title="Assistente de Cozinha RAG", page_icon="🍳", layout="wide")

langsmith_ativo = setup_langsmith()
PESSOAS_OPCOES = list(range(1, 11))


@st.cache_data(show_spinner=False, ttl=3600)
def _geocode(endereco: str):
    return geocode_address(endereco)


_MAX_FETCH_RADIUS_KM = 20

@st.cache_data(show_spinner=False, ttl=3600)
def _supermercados(lat: float, lon: float):
    return find_supermarkets(lat, lon, radius_km=_MAX_FETCH_RADIUS_KM)


@st.cache_data(show_spinner=False, ttl=300)
def _index_exists() -> bool:
    return index_exists()


def _render_ficha(ficha: FichaTecnica) -> None:
    st.subheader(f"🍽️ {ficha.nome_prato}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Categoria", ficha.categoria)
    col2.metric("Porções", ficha.porcoes)
    col3.metric("Tempo de preparação", ficha.tempo_preparacao)

    st.markdown("#### 🧂 Ingredientes")
    st.table([{"Ingrediente": i.nome, "Quantidade": i.quantidade} for i in ficha.ingredientes])

    st.markdown("#### 👩‍🍳 Modo de preparação")
    for idx, passo in enumerate(ficha.modo_preparacao, start=1):
        st.markdown(f"{idx}. {passo}")


def _render_feedback(ficha: FichaTecnica, run_id: str | None) -> None:
    feedback_key = f"feedback_{run_id or ficha.nome_prato}"
    st.markdown("#### O que achaste desta sugestão?")

    if st.session_state.get(feedback_key):
        st.caption("Obrigado pelo feedback! 🙏")
        return

    col1, col2, _ = st.columns([1, 1, 8])
    if col1.button("👍", key=f"like_{feedback_key}"):
        if langsmith_ativo and run_id:
            send_feedback(run_id, score=1)
        update_recipe_feedback(ficha.nome_prato, gostou=True)
        st.session_state[feedback_key] = True
        st.rerun()
    if col2.button("👎", key=f"dislike_{feedback_key}"):
        if langsmith_ativo and run_id:
            send_feedback(run_id, score=0)
        update_recipe_feedback(ficha.nome_prato, gostou=False)
        st.session_state[feedback_key] = True
        st.rerun()


def _render_lista_compras(ficha: FichaTecnica, ficha_key: str) -> None:
    col1, col2 = st.columns([4, 1])
    col1.markdown("#### Custo estimado da lista")

    if not ficha.ingredientes_em_falta:
        st.info("Não é necessário comprar nada — já tens todos os ingredientes!")
        return

    show_key = f"show_precos_{ficha_key}"
    result_key = f"precos_result_{ficha_key}"

    if col2.button("💳 Estimar preços", key=f"btn_precos_{ficha_key}", use_container_width=True):
        st.session_state[show_key] = True

    if st.session_state.get(show_key):
        if result_key not in st.session_state:
            with st.spinner("A calcular preços..."):
                st.session_state[result_key] = estimar_custo(ficha.ingredientes_em_falta)
        total, detalhe = st.session_state[result_key]
        st.table(
            [
                {
                    "Ingrediente": d["nome"],
                    "Quantidade": d["quantidade"],
                    "Preço estimado (€)": f"{d['preco_estimado']:.2f}",
                }
                for d in detalhe
            ]
        )
        st.metric("Custo total estimado", f"{total:.2f} €")
        st.caption(
            "Os preços são estimativas médias indicativas e podem não refletir os valores "
            "reais em loja."
        )


def _render_supermercados(ficha_key: str) -> None:
    col1, col2 = st.columns([4, 1])
    col1.markdown("#### Onde comprar perto de ti")

    show_key = f"show_supermercados_{ficha_key}"

    if col2.button("📍 Encontrar supermercados perto", key=f"btn_supermercados_{ficha_key}", use_container_width=True):
        st.session_state[show_key] = True

    if not st.session_state.get(show_key):
        return

    col_loc, col_raio = st.columns([3, 2])
    with col_loc:
        modo = st.segmented_control(
            "Localização",
            options=["📍 Usar localização", "✏️ Indicar morada"],
            default="📍 Usar localização",
            key=f"loc_modo_{ficha_key}",
        )
    with col_raio:
        raio_km = st.segmented_control(
            "Raio de pesquisa",
            options=[1, 5, 10, 20],
            format_func=lambda x: f"{x} km",
            default=config.MAX_RADIUS_KM,
            key=f"loc_raio_{ficha_key}",
        )
        if raio_km is None:
            raio_km = config.MAX_RADIUS_KM

    coords = None
    if modo != "✏️ Indicar morada":
        localizacao = streamlit_geolocation()
        if localizacao and localizacao.get("latitude") is not None:
            coords = (localizacao["latitude"], localizacao["longitude"])
        elif localizacao and localizacao.get("latitude") is None:
            st.caption("Permite o acesso à localização no browser para continuar.")
    else:
        endereco = st.text_input(
            "Morada", placeholder="ex: Avenida da Liberdade, Lisboa", key=f"loc_endereco_{ficha_key}"
        )
        if endereco:
            coords = _geocode(endereco)
            if coords is None:
                st.warning("Não foi possível localizar esse endereço. Verifica e tenta novamente.")

    if coords is None:
        return

    lat, lon = coords
    try:
        supermercados = _supermercados(lat, lon)
    except Exception as exc:
        st.warning(f"Não foi possível obter a lista de supermercados: {exc}")
        return

    supermercados = filter_by_distance(supermercados, raio_km)
    if not supermercados:
        st.info("Não foram encontrados supermercados nesse raio.")
        return

    st.caption("⭐ = cadeia recomendada")

    for s in supermercados:
        with st.container(border=True):
            col_info, col_link = st.columns([5, 1])
            with col_info:
                nome = ("⭐ " if s["recomendado"] else "") + s["nome"]
                st.markdown(f"**{nome}**")
                if s["endereco"]:
                    st.caption(s["endereco"])
                st.caption(f"📍 {s['distancia_km']} km")
            with col_link:
                maps_url = f"https://www.google.com/maps/search/?api=1&query={s['lat']},{s['lon']}"
                st.link_button("Ver ↗", maps_url, use_container_width=True)

    mapa = [{"lat": lat, "lon": lon}] + [{"lat": s["lat"], "lon": s["lon"]} for s in supermercados]
    st.map(mapa, latitude="lat", longitude="lon")


# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.caption(f"Modelo LLM: `{config.LLM_MODEL}`")
    st.caption(f"Modelo de embeddings: `{config.EMBEDDING_MODEL}`")
    st.caption("LangSmith: " + ("✅ ativo" if langsmith_ativo else "desativado (define LANGCHAIN_API_KEY)"))


# --- Corpo principal -----------------------------------------------------------------
st.title("🍳 Assistente de Cozinha RAG")
st.caption("Receitas portuguesas com Chroma + LangSmith + OpenAI")

if not config.OPENAI_API_KEY:
    st.error("Define a variável OPENAI_API_KEY no ficheiro .env antes de usar a aplicação.")

if not _index_exists():
    st.warning("Índice ChromaDB não encontrado. Corre `python scripts/ingest.py` para o criar.")

tab1, tab2, tab3 = st.tabs(
    ["💡 Tenho estes ingredientes", "🍽️ Quero fazer este prato", "🎲 Surpreende-me"]
)

with tab1:
    st.markdown("Indica os ingredientes que tens disponíveis em casa (um por linha).")
    ingredientes_texto = st.text_area(
        "Ingredientes disponíveis",
        placeholder="arroz\nfrango\ncebola\nalho\nazeite",
        height=150,
        key="opcao1_ingredientes",
    )
    pessoas1 = st.selectbox("Para quantas pessoas?", options=PESSOAS_OPCOES, index=3, key="opcao1_pessoas")

    if st.button("Sugerir prato", key="opcao1_btn"):
        ingredientes = [linha.strip() for linha in ingredientes_texto.splitlines() if linha.strip()]
        if not ingredientes:
            st.warning("Indica pelo menos um ingrediente.")
        else:
            try:
                with st.spinner("A pensar num prato..."):
                    ficha, run_id = gerar_opcao1(ingredientes, pessoas1)
                st.session_state["resultado"] = (ficha, run_id)
            except Exception as exc:
                st.error(f"Ocorreu um erro ao gerar a sugestão: {exc}")

with tab2:
    st.markdown("Indica o nome do prato que queres preparar.")
    nome_prato = st.text_input("Prato", placeholder="ex: Caldo Verde", key="opcao2_nome")
    pessoas2 = st.selectbox("Para quantas pessoas?", options=PESSOAS_OPCOES, index=3, key="opcao2_pessoas")

    if st.button("Gerar receita", key="opcao2_btn"):
        if not nome_prato.strip():
            st.warning("Indica o nome do prato.")
        else:
            try:
                with st.spinner("A preparar a ficha técnica..."):
                    ficha, run_id = gerar_opcao2(nome_prato.strip(), pessoas2)
                st.session_state["resultado"] = (ficha, run_id)
            except Exception as exc:
                st.error(f"Ocorreu um erro ao gerar a receita: {exc}")

with tab3:
    st.markdown("Deixa a IA sugerir um prato para ti.")
    tipo_refeicao = st.selectbox("Tipo de refeição", options=TIPOS_REFEICAO, key="opcao3_tipo")
    pessoas3 = st.selectbox("Para quantas pessoas?", options=PESSOAS_OPCOES, index=3, key="opcao3_pessoas")

    if st.button("Sugerir", key="opcao3_btn"):
        try:
            with st.spinner("A escolher um prato..."):
                ficha, run_id = gerar_opcao3(pessoas3, tipo_refeicao)
            st.session_state["resultado"] = (ficha, run_id)
        except Exception as exc:
            st.error(f"Ocorreu um erro ao gerar a sugestão: {exc}")


if "resultado" in st.session_state:
    ficha, run_id = st.session_state["resultado"]
    ficha_key = run_id or ficha.nome_prato.lower().replace(" ", "_")
    st.divider()
    _render_ficha(ficha)
    _render_feedback(ficha, run_id)
    st.divider()
    with st.container(border=True):
        _render_lista_compras(ficha, ficha_key)
        st.write("")
        _render_supermercados(ficha_key)
