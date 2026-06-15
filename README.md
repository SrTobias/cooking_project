# 🍳 Assistente de Cozinha RAG

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
- `src/supermarkets.py` — geocoding (Nominatim) + procura de supermercados (Overpass/OSM)
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
   - `OPENAI_API_KEY` (obrigatório) — usado para o LLM (`gpt-4o-mini` por defeito) e para os
     embeddings (`text-embedding-3-small`)
   - `LANGCHAIN_API_KEY` (opcional) — se definido, ativa o tracing/dashboard/feedback no
     [LangSmith](https://smith.langchain.com)
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

## Funcionalidades

A aplicação tem 3 modos de interação, todos com seleção do número de pessoas (as
quantidades dos ingredientes são escaladas automaticamente):

1. **💡 Tenho estes ingredientes** — descreve o que tens em casa e recebe uma sugestão de
   prato com a ficha técnica completa.
2. **🍽️ Quero fazer este prato** — indica o prato que queres preparar e recebe a ficha
   técnica (ingredientes + modo de preparação).
3. **🎲 Surpreende-me** — a IA sugere um prato (com filtro opcional por tipo de refeição:
   sopa, prato principal ou sobremesa).

Em todos os casos, é apresentada uma secção **🛒 Lista de compras e supermercados próximos**
com:
- estimativa do custo dos ingredientes que faltam comprar (preços médios indicativos);
- lista de supermercados num raio até 20 km do endereço indicado na barra lateral, com
  filtro por distância e indicação ⭐ para cadeias recomendadas (Continente, Pingo Doce,
  Lidl, Mercadona, etc.).

## Receitas (corpus)

`data/recipes/` contém ~20 receitas portuguesas em ficheiros `.txt` (peixe, carne,
vegetariano, sopas e sobremesas). Para adicionar as tuas próprias receitas, segue o mesmo
formato (`# Nome do prato`, `## Categoria`, `## Porções base`, `## Tempo de preparação`,
`## Ingredientes`, `## Modo de preparação`) e volta a correr `python scripts/ingest.py`.
Também são suportados ficheiros `.pdf` na mesma pasta.

### Metadata indexada

Cada chunk no Chroma guarda os seguintes campos de metadata: `nome_prato`, `categoria`,
`tempo_preparacao` (extraído da secção "## Tempo de preparação"), `likes` e `dislikes`.

### Auto-enriquecimento da base de conhecimento

Sempre que uma das 3 opções gera uma ficha técnica para um prato que ainda não existe na
coleção `cooking_project` (por nome, ignorando acentos/maiúsculas), essa receita é
automaticamente guardada em `data/recipes/gerado_<nome-do-prato>.txt` e adicionada ao
índice Chroma — fica disponível de imediato para futuras pesquisas/recuperações.

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

## Aviso sobre preços

As estimativas de custo baseiam-se numa tabela de preços médios (`data/precos_ingredientes.json`)
e servem apenas como referência aproximada — não refletem os preços reais em loja.
