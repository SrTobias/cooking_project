"""Cooking for Dummies - Streamlit UI.

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
from src.rag import FichaTecnica, gerar_opcao1, gerar_opcao2, gerar_opcao3
from src.supermarkets import filter_by_distance, find_supermarkets, geocode_address

st.set_page_config(page_title="Cooking for Dummies", page_icon="🍳", layout="wide")

langsmith_ativo = setup_langsmith()
PESSOAS_OPCOES = list(range(1, 11))
_TEMPO_MAP = {"30 min": 30, "60 min": 60, "90 min": 90}
MODO_OPCOES = ["💡 Tenho estes ingredientes", "🍽️ Quero fazer este prato", "🎲 Surpreende-me"]


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
    with col1:
        st.caption("Categoria")
        st.markdown(f"**{ficha.categoria}**")
    with col2:
        st.caption("Porções")
        st.markdown(f"**{ficha.porcoes}**")
    with col3:
        st.caption("Tempo de preparação")
        st.markdown(f"**{ficha.tempo_preparacao}**")

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

    _QB = {"q.b.", "q.b", "quanto baste", "a gosto"}
    ingredientes_a_comprar = [
        i for i in ficha.ingredientes_em_falta
        if i.quantidade.strip().lower() not in _QB
    ]

    if not ingredientes_a_comprar:
        st.info("Não é necessário comprar nada — já tens todos os ingredientes!")
        return

    show_key = f"show_precos_{ficha_key}"
    result_key = f"precos_result_{ficha_key}"

    if col2.button("💳 Estimar preços", key=f"btn_precos_{ficha_key}", use_container_width=True):
        st.session_state[show_key] = True

    if st.session_state.get(show_key):
        if result_key not in st.session_state:
            with st.spinner("A calcular preços..."):
                st.session_state[result_key] = estimar_custo(ingredientes_a_comprar)
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
            "reais em loja. Os ingredientes q.b não são incluidos na lista de compras."
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
            "Morada",
            placeholder="ex: Avenida da Liberdade, Lisboa  ou  1300-552 Lisboa",
            key=f"loc_endereco_{ficha_key}",
        )
        if endereco:
            try:
                coords = _geocode(endereco)
            except Exception as exc:
                st.warning(f"Não foi possível obter a localização: {exc}")
            else:
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

    _PASSO = 5
    count_key = f"count_supermercados_{ficha_key}"
    if count_key not in st.session_state:
        st.session_state[count_key] = _PASSO
    visiveis = supermercados[:st.session_state[count_key]]

    st.caption("⭐ = cadeia recomendada")

    for s in visiveis:
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

    restantes = len(supermercados) - st.session_state[count_key]
    if restantes > 0:
        proximos = min(_PASSO, restantes)
        if st.button(f"Mostrar mais {proximos} supermercado{'s' if proximos != 1 else ''}", key=f"btn_mais_{ficha_key}"):
            st.session_state[count_key] += _PASSO
            st.rerun()

    mapa = [{"lat": lat, "lon": lon}] + [{"lat": s["lat"], "lon": s["lon"]} for s in supermercados]
    st.map(mapa, latitude="lat", longitude="lon")


# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.caption(f"Modelo LLM: `{config.LLM_MODEL}`")
    st.caption(f"Modelo de embeddings: `{config.EMBEDDING_MODEL}`")
    st.caption("LangSmith: " + ("✅ ativo" if langsmith_ativo else "desativado (define LANGCHAIN_API_KEY)"))


# --- Corpo principal -----------------------------------------------------------------
st.title("🍳 Cooking for Dummies")
st.caption("Receitas para a sua refeição gerada pela IA")

if not config.OPENAI_API_KEY:
    st.error("Define a variável OPENAI_API_KEY no ficheiro .env antes de usar a aplicação.")

if not _index_exists():
    st.warning("Índice ChromaDB não encontrado. Corre `python scripts/ingest.py` para o criar.")

modo = st.segmented_control(
    "Modo", options=MODO_OPCOES, default=MODO_OPCOES[0], label_visibility="collapsed"
)
if modo is None:
    modo = st.session_state.get("_ultimo_modo", MODO_OPCOES[0])
if modo != st.session_state.get("_ultimo_modo"):
    st.session_state.pop("resultado", None)
st.session_state["_ultimo_modo"] = modo

st.divider()

if modo == MODO_OPCOES[0]:
    st.markdown("Indica os ingredientes que tens disponíveis em casa (um por linha).")
    ingredientes_texto = st.text_area(
        "Ingredientes disponíveis",
        placeholder="arroz\nfrango\ncebola\nalho\nazeite",
        height=150,
        key="opcao1_ingredientes",
    )
    col_p1, col_d1, col_t1, _ = st.columns([1, 3, 3, 2])
    with col_p1:
        pessoas1 = st.selectbox("Para quantas pessoas?", options=PESSOAS_OPCOES, index=3, key="opcao1_pessoas")
    with col_d1:
        dieta1 = st.segmented_control("Dieta", options=["🍖 Como de tudo", "🥗 Vegetariano", "🌱 Vegan"], default="🍖 Como de tudo", key="opcao1_dieta")
    with col_t1:
        tempo1 = st.segmented_control("Tempo máximo", options=["Sem limite", "30 min", "60 min", "90 min"], default="Sem limite", key="opcao1_tempo")

    if st.button("Sugerir prato", key="opcao1_btn"):
        ingredientes = [linha.strip() for linha in ingredientes_texto.splitlines() if linha.strip()]
        if not ingredientes:
            st.warning("Indica pelo menos um ingrediente.")
        else:
            try:
                with st.spinner("A pensar num prato..."):
                    ficha, run_id = gerar_opcao1(ingredientes, pessoas1, dieta1, _TEMPO_MAP.get(tempo1))
                st.session_state["resultado"] = (ficha, run_id)
            except Exception as exc:
                st.error(f"Ocorreu um erro ao gerar a sugestão: {exc}")

elif modo == MODO_OPCOES[1]:
    st.markdown("Indica o nome do prato que queres preparar.")
    nome_prato = st.text_input("Prato", placeholder="ex: Caldo Verde", key="opcao2_nome")
    col_p2, col_t2, _ = st.columns([1, 3, 5])
    with col_p2:
        pessoas2 = st.selectbox("Para quantas pessoas?", options=PESSOAS_OPCOES, index=3, key="opcao2_pessoas")
    with col_t2:
        tempo2 = st.segmented_control("Tempo máximo", options=["Sem limite", "30 min", "60 min", "90 min"], default="Sem limite", key="opcao2_tempo")

    if st.button("Gerar receita", key="opcao2_btn"):
        if not nome_prato.strip():
            st.warning("Indica o nome do prato.")
        else:
            try:
                with st.spinner("A preparar a ficha técnica..."):
                    ficha, run_id = gerar_opcao2(nome_prato.strip(), pessoas2, _TEMPO_MAP.get(tempo2))
                st.session_state["resultado"] = (ficha, run_id)
            except Exception as exc:
                st.error(f"Ocorreu um erro ao gerar a receita: {exc}")

else:
    st.markdown("Deixa a IA sugerir um prato para ti.")
    col_p3, col_d3, col_t3, _ = st.columns([1, 3, 3, 2])
    with col_p3:
        pessoas3 = st.selectbox("Para quantas pessoas?", options=PESSOAS_OPCOES, index=3, key="opcao3_pessoas")
    with col_d3:
        dieta3 = st.segmented_control("Dieta", options=["🍖 Como de tudo", "🥗 Vegetariano", "🌱 Vegan"], default="🍖 Como de tudo", key="opcao3_dieta")
    with col_t3:
        tempo3 = st.segmented_control("Tempo máximo", options=["Sem limite", "30 min", "60 min", "90 min"], default="Sem limite", key="opcao3_tempo")

    if st.button("Sugerir", key="opcao3_btn"):
        try:
            with st.spinner("A escolher um prato..."):
                ficha, run_id = gerar_opcao3(pessoas3, dieta=dieta3, tempo_max=_TEMPO_MAP.get(tempo3))
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
