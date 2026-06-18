# 🍳 Cooking for Dummies

Aplicação RAG (Retrieval-Augmented Generation) sobre receitas portuguesas, construída com
**LangChain**, **ChromaDB**, **OpenAI** e **LangSmith**, com interface em **Streamlit**.

## Arquitetura

```
Pipeline de indexação:
  Documentos (data/recipes/*.txt) -> Text Splitter -> Embeddings (OpenAI) -> ChromaDB

Pipeline de query:
  Pergunta -> Retriever (Chroma, top-k) -> LLM (OpenAI) -> Resposta (Ficha Técnica)
  (com Tracing/Feedback via LangSmith)
```

- `src/ingestion.py` — carrega as receitas, divide em chunks e gera/persiste o índice Chroma
- `src/rag.py` — schemas (`FichaTecnica`) e as 3 chains (opções 1/2/3)
- `src/observability.py` — ativa o tracing do LangSmith e envia feedback
- `src/supermarkets.py` — geocoding (Nominatim) + procura de supermercados (Overpass/OSM,
  com fallback automático entre vários mirrors em `config.OVERPASS_URLS` em caso de
  erro/timeout, ex: 504 do servidor principal)
- `src/pricing.py` — estimativa do custo da lista de compras
- `app.py` — interface Streamlit

## Configuração

1. Cria um ambiente virtual e instala as dependências:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copia `.env.example` para `.env` e configura:
   - `OPENAI_API_KEY` (obrigatório) — usado para o LLM e para os embeddings
   - `LLM_MODEL` (opcional, default `gpt-4o-mini`) — modelo OpenAI a usar para geração
   - `EMBEDDING_MODEL` (opcional, default `text-embedding-3-small`) — modelo de embeddings
   - `RELEVANCE_THRESHOLD` (opcional, default `0.25`) — score mínimo para usar um documento
     como contexto; ver secção [Recuperação de receitas](#recuperação-de-receitas-likes--relevância)
   - `LANGCHAIN_API_KEY` (opcional) — se definido, ativa o tracing/dashboard/feedback no
     [LangSmith](https://smith.langchain.com)
   - `LANGCHAIN_TRACING_V2=true` e `LANGCHAIN_PROJECT` (opcional) — necessários para o
     tracing funcionar corretamente no LangSmith
   - `CHROMA_API_KEY` / `CHROMA_TENANT` / `CHROMA_DATABASE` (opcional) — se definidos, o
     índice é guardado no [Chroma Cloud](https://www.trychroma.com/) em vez de um diretório
     local `chroma_db/`. Deixa por preencher para usar ChromaDB local (sem conta nem chave).

3. Constrói o índice ChromaDB (local ou Chroma Cloud, conforme a configuração acima) a
   partir das receitas de exemplo:

   ```bash
   python scripts/ingest.py
   ```

4. Arranca a aplicação:

   ```bash
   streamlit run app.py
   ```

   **Streamlit Cloud:** a app suporta deploy direto no [Streamlit Cloud](https://streamlit.io/cloud).
   Nesse caso, define as variáveis acima em **Settings → Secrets** (formato TOML) em vez de
   usar um ficheiro `.env` — a app lê automaticamente `st.secrets` como fallback.

## Funcionalidades

A aplicação tem 3 modos de interação, todos com seleção do número de pessoas (as
quantidades dos ingredientes são ajustadas automaticamente):

1. **💡 Tenho estes ingredientes** — descreve o que tens em casa e recebe uma sugestão de
   prato com a ficha técnica completa. Inclui filtros de **preferência alimentar** (🍖 Como
   de tudo / 🥗 Vegetariano / 🌱 Vegan) e **tempo máximo de confeção** (Sem limite / 30 /
   60 / 90 min).
2. **🍽️ Quero fazer este prato** — indica o prato que queres preparar e recebe a ficha
   técnica (ingredientes + modo de preparação). Inclui filtro de **tempo máximo de confeção**.
   Entradas que não correspondam a pratos de culinária são rejeitadas antes de chamar o LLM.
3. **🎲 Surpreende-me** — a IA sugere um prato livremente. Inclui filtros de **preferência
   alimentar** e **tempo máximo de confeção**.

Em todos os casos, é apresentada uma secção **🛒 Lista de compras e supermercados próximos**
com:
- estimativa do custo dos ingredientes que faltam comprar (preços médios indicativos);
- cards individuais por supermercado (nome, endereço, distância) com filtro de raio (1/5/10/20 km),
  indicação ⭐ para cadeias recomendadas (Continente, Pingo Doce, Lidl, Mercadona, etc.) e
  botão **"Ver ↗"** que abre diretamente a localização no Google Maps.

## Receitas (corpus)

`data/recipes/` contém 50 receitas portuguesas em ficheiros `.txt` (peixe, carne,
vegetariano, sopas e sobremesas), com o nome do ficheiro a corresponder ao nome do prato.
Para adicionar as tuas próprias receitas, segue o mesmo
formato (`# Nome do prato`, `## Categoria`, `## Porções base`, `## Tempo de preparação`,
`## Ingredientes`, `## Modo de preparação`) e volta a correr `python scripts/ingest.py`.
Também são suportados ficheiros `.pdf` na mesma pasta.

### Metadata indexada

Cada chunk no Chroma guarda os seguintes campos de metadata: `nome_prato`, `categoria`,
`tempo_preparacao` (número inteiro, em minutos, convertido a partir da secção "## Tempo de
preparação"), `likes` e `dislikes`.

### Auto-enriquecimento da base de conhecimento

Sempre que uma das 3 opções gera uma ficha técnica para um prato que ainda não existe na
coleção `cooking_project` (por nome, ignorando acentos/maiúsculas), essa receita é
automaticamente guardada em `data/recipes/<nome-do-prato>.txt` e adicionada ao índice
Chroma — fica disponível de imediato para futuras pesquisas/recuperações.

### Recuperação de receitas (likes + relevância)

`retrieve_with_likes` (em `src/rag.py`) procura `fetch_k` candidatos por semelhança,
descarta os que tiverem um score de relevância abaixo de `RELEVANCE_THRESHOLD` (0-1,
configurável via variável de ambiente, default `0.25`) e reordena os restantes pelos likes
líquidos (likes - dislikes), para que as receitas mais bem avaliadas apareçam primeiro no
contexto dado ao LLM. Se nenhum candidato atingir o threshold, mantém-se o mais relevante
para evitar contexto vazio.

### Performance

A ligação ao Chroma (cliente Chroma Cloud/local + embeddings) é cacheada em memória
(`load_vectorstore`, com `lru_cache`), evitando recriar a ligação a cada pedido e reduzindo
a latência das gerações.

## LangSmith

Com `LANGCHAIN_API_KEY` configurado, todas as chamadas ao LLM ficam visíveis no dashboard
do [LangSmith](https://smith.langchain.com) (projeto `LANGCHAIN_PROJECT`), incluindo
Tracing, Datasets e Feedback (os botões 👍/👎 na app enviam feedback associado ao traço da
geração).

## Preços dos ingredientes

As estimativas de custo usam três fontes, por esta ordem:

1. **`data/precos_ingredientes.json`** — tabela de preços médios para ingredientes comuns em
   supermercados portugueses. O campo `_updated_at` indica a data da última atualização.
2. **LLM como fallback** — se um ingrediente não for encontrado na tabela, é feita uma chamada
   ao LLM (`LLM_MODEL`) para estimar o preço em supermercados portugueses. O resultado é
   cacheado em memória durante a sessão. A coluna "correspondência" na lista de compras mostra
   `(via LLM)` para estes casos.
3. **Estimativa genérica** — se a chamada ao LLM falhar (ex: erro de rede), usa-se um preço
   genérico (`config.DEFAULT_PRICE_EUR`), assinalado como `(estimativa genérica)`.

### Atualizar a tabela de preços

A tabela é atualizada automaticamente todas as segundas-feiras através do workflow
[`.github/workflows/update-prices.yml`](.github/workflows/update-prices.yml) (GitHub Actions),
que corre `scripts/update_prices.py` e faz commit do ficheiro se houver alterações. Requer o
secret `OPENAI_API_KEY` configurado no repositório.

Para refrescar manualmente os preços de todos os ingredientes da tabela com estimativas
recentes do LLM:

```bash
python scripts/update_prices.py
```

Usa `--dry-run` para ver o resultado sem gravar o ficheiro.

Os valores são sempre **referências aproximadas** — não refletem os preços reais em loja.
