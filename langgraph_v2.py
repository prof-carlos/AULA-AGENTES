"""
App: Planejador de Viagens com LangGraph (com fallback headless)

Correção aplicada: o erro `ModuleNotFoundError: No module named 'streamlit'` ocorre quando o
Streamlit não está instalado/indisponível. Este arquivo agora:
- Usa Streamlit **se disponível** (UI completa).
- Cai para um **modo CLI/headless** quando Streamlit não está presente.
- Inclui um **LLM de fallback** (DummyLLM) se `langchain_groq`/`GROQ_API_KEY` não estiverem disponíveis.
- Adiciona **testes embutidos** executáveis com `RUN_TESTS=1 python app.py`.

Como executar (UI):
  pip install streamlit langgraph langchain-core langchain-groq
  export GROQ_API_KEY=...  # ou informe no campo da UI
  streamlit run app.py

Como executar (headless/CLI):
  # sem streamlit instalado, roda em modo texto
  python app.py
  # ou com variáveis de ambiente para não interagir:
  DESTINO="Lisboa, Portugal" DATA_INICIO=2025-01-01 DATA_FIM=2025-01-05 ORCAMENTO="R$ 5.000" PREFERENCIAS="museus" python app.py

Como rodar os testes internos:
  RUN_TESTS=1 python app.py
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from typing import Any, TypedDict

# ==========================
# Imports opcionais (com fallback)
# ==========================
try:  # Streamlit pode não estar instalado no ambiente
    import streamlit as _st
    st = _st
    STREAMLIT_AVAILABLE = True
except ModuleNotFoundError:
    st = None
    STREAMLIT_AVAILABLE = False

try:  # LangChain - mensagens
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:  # fallback mínimo para rodar sem langchain_core
    class _Msg:  # type: ignore
        def __init__(self, content: str):
            self.content = content
    SystemMessage = HumanMessage = _Msg  # type: ignore

try:  # Groq LLM (opcional)
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None  # type: ignore

# LangGraph é obrigatório para o grafo
try:
    from langgraph.graph import StateGraph, END
except Exception as e:  # falha dura, mas com mensagem clara
    raise RuntimeError(
        "LangGraph é necessário. Instale com: pip install langgraph"
    ) from e

# ==========================
# Estado do grafo
# ==========================
class TripState(TypedDict, total=False):
    destino: str
    data_inicio: str
    data_fim: str
    orcamento: str
    preferencias: str

    plan: str
    hotels: str
    leisure: str
    food: str
    final: str

# ==========================
# Utilidades
# ==========================

def _parse_date(d: Any) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    # tenta ISO
    return datetime.fromisoformat(str(d)).date()


def validate_inputs(destino: str, data_inicio: Any, data_fim: Any) -> tuple[bool, str | None]:
    if not destino or not str(destino).strip():
        return False, "Por favor, informe o destino."
    try:
        d0 = _parse_date(data_inicio)
        d1 = _parse_date(data_fim)
    except Exception:
        return False, "Datas inválidas: use formato ISO AAAA-MM-DD."
    if d1 < d0:
        return False, "A data de término deve ser posterior à data de início."
    return True, None

# ==========================
# Fallback LLM (quando Groq/LLM real indisponível)
# ==========================
class _DummyResp:
    def __init__(self, content: str):
        self.content = content


class DummyLLM:
    """LLM simples para ambientes offline/teste.
    Gera conteúdo sintético, mas com estrutura esperada pelas tarefas.
    """

    def __init__(self, temperature: float = 0.0):
        self.temperature = temperature

    def invoke(self, messages: list[Any]) -> _DummyResp:
        prompt = "\n\n".join(getattr(m, "content", str(m)) for m in messages)
        if "HOSPEDAGEM" in prompt.upper() and "TABELA" in prompt.upper():
            return _DummyResp(
                """
| Nome | Endereço | Site | Telefone |
| --- | --- | --- | --- |
| Hotel Central | Rua A, 123 | https://hotelcentral.example | +351 21 000 000 |
| Boutique Vista | Av. B, 456 | https://boutiquevista.example | +351 21 111 111 |
| Porto Inn | Rua C, 789 | https://portoinn.example | +351 21 222 222 |

**Fontes**: [Turismo Local](https://turismo.local), [Guias Exemplo](https://guia.exemplo)
                """.strip()
            )
        if "LAZER" in prompt.upper():
            return _DummyResp(
                """
- Castelo Histórico — Panorama da cidade. <https://castelo.example>
- Museu de Arte — Coleções modernas. <https://museu.example>
- Mercado Central — Gastronomia local. <https://mercado.example>

**Eventos (período)**
- Festival de Verão — Música ao ar livre. <https://festival.example>
- Feira de Livros — Autores locais. <https://feira.example>

**Fontes**: [Agenda Cultural](https://agenda.example)
                """.strip()
            )
        if "ALIMENTAÇÃO" in prompt.upper():
            return _DummyResp(
                """
| Nome | Bairro | Faixa de Preço | Cozinha | Site |
| --- | --- | --- | --- | --- |
| Tasca do Bairro | Centro | $$ | Portuguesa | https://tasca.example |
| Mar & Brasa | Ribeira | $$$ | Peixes e grelhados | https://marebrasa.example |

**Comidas típicas**: Bacalhau à Brás, Pastel de Nata, Caldo Verde, Francesinha.

**Fontes**: [Guia Gastronômico](https://gastronomia.example)
                """.strip()
            )
        if "RELATÓRIO FINAL" in prompt.upper():
            return _DummyResp(
                ("Introdução: Este roteiro sintetiza opções de hospedagem, lazer e alimentação "
                 "para uma experiência equilibrada.\n\n"
                 "Hospedagem: ver tabela acima.\n\nLazer: destaques culturais e eventos no período.\n\n"
                 "Alimentação: restaurantes recomendados e comidas típicas.\n\n"
                 "Mini-roteiro: Dia 1 — centro histórico; Dia 2 — museus; Dia 3 — orla.\n\n"
                 "Dicas rápidas: compre bilhetes antecipados; use transporte público; atenção a pertences.\n\n"
                 "Fontes: consolidadas ao final das seções.")
            )
        # Planejamento geral
        return _DummyResp(
            """
1) HOSPEDAGEM — Selecionar hotéis bem localizados e com bom custo-benefício.
2) LAZER — Mapear atrações essenciais e eventos no período.
3) ALIMENTAÇÃO — Levantar restaurantes e comidas típicas.

**Critérios**
- Proximidade a transporte/centro
- Avaliações consistentes
- Adequação ao orçamento/preferências
- Variedade de experiências

Justificativa: o recorte em 3 pilares organiza a pesquisa e facilita decisões.
            """.strip()
        )

# ==========================
# LLM Factory (com cache quando Streamlit existir)
# ==========================

def _get_llm_impl(api_key: str | None, temperature: float = 0.2):
    key = api_key or os.environ.get("GROQ_API_KEY")
    if ChatGroq and key:
        return ChatGroq(api_key=key, model_name="llama-3.3-70b-versatile", temperature=temperature)
    # fallback
    return DummyLLM(temperature=temperature)


if STREAMLIT_AVAILABLE:
    @st.cache_resource(show_spinner=False)
    def get_llm(api_key: str | None, temperature: float = 0.2):
        return _get_llm_impl(api_key, temperature)
else:
    def get_llm(api_key: str | None, temperature: float = 0.2):  # type: ignore
        return _get_llm_impl(api_key, temperature)

# ==========================
# Nós do grafo (LangGraph)
# ==========================

def make_planner_node(llm: Any):
    def node(state: TripState) -> dict:
        system = "Você estrutura planos objetivos e práticos em 3 passos fixos."
        human = f"""
PLANEJAMENTO GERAL\n
Destino: {state['destino']}\n
Datas: {state['data_inicio']} a {state['data_fim']}\n
Orçamento: {state['orcamento']}\n
Preferências: {state['preferencias']}\n
\n
Sua função: **Roteirista de Viagens**.\n
Objetivo: Gerar um plano de pesquisa dividido em **EXATAMENTE 3 subtarefas numeradas**:\n
1) HOSPEDAGEM; 2) LAZER; 3) ALIMENTAÇÃO.\n
\n
Regras de saída (em Markdown):\n
- Liste as 3 subtarefas numeradas (cada uma com 1 frase).\n
- Em seguida, traga 3–5 critérios de seleção (bullets) considerando orçamento e preferências.\n
- Feche com 1–2 linhas de justificativa.\n
Retorne **somente** esse conteúdo.
"""
        resp = get_llm(None).invoke([SystemMessage(content=system), HumanMessage(content=human)])
        return {"plan": resp.content}

    return node


def make_hotels_node(llm: Any):
    def node(state: TripState) -> dict:
        system = "Você verifica informações de hotéis e organiza dados de contato."
        human = f"""
HOSPEDAGEM\n
Destino: {state['destino']}\n
Período: {state['data_inicio']} – {state['data_fim']}\n
Orçamento: {state['orcamento']}\n
Preferências: {state['preferencias']}\n
Plano do roteirista:\n
{state.get('plan','')}\n
\n
Entregue uma **tabela Markdown** com as colunas: **Nome | Endereço | Site | Telefone**.\n
Inclua **5–8 opções** e **2–4 fontes** (título + URL) ao final.
"""
        resp = get_llm(None).invoke([SystemMessage(content=system), HumanMessage(content=human)])
        return {"hotels": resp.content}

    return node


def make_leisure_node(llm: Any):
    def node(state: TripState) -> dict:
        system = "Você encontra atrações e eventos relevantes às datas."
        human = f"""
LAZER\n
Destino: {state['destino']}\n
Período: {state['data_inicio']} – {state['data_fim']}\n
Plano do roteirista:\n
{state.get('plan','')}\n
\n
Liste **8–12 pontos turísticos essenciais** (com breve descrição e link).\n
Depois, **3–5 eventos** que ocorram no período informado (com breve descrição e link).\n
Formate em listas e finalize com **2–4 fontes** (título + URL).
"""
        resp = get_llm(None).invoke([SystemMessage(content=system), HumanMessage(content=human)])
        return {"leisure": resp.content}

    return node


def make_food_node(llm: Any):
    def node(state: TripState) -> dict:
        system = "Você conhece a cena gastronômica e as especialidades locais."
        human = f"""
ALIMENTAÇÃO\n
Destino: {state['destino']}\n
Preferências: {state['preferencias']}\n
Plano do roteirista:\n
{state.get('plan','')}\n
\n
1) Recomende **8–12 restaurantes** (\n
   entregue em **tabela Markdown** com **Nome | Bairro | Faixa de Preço | Cozinha | Site**).\n
2) Liste **5–8 comidas típicas** com breve explicação.\n
Finalize com **2–4 fontes** (título + URL).
"""
        resp = get_llm(None).invoke([SystemMessage(content=system), HumanMessage(content=human)])
        return {"food": resp.content}

    return node


def make_writer_node(llm: Any):
    def node(state: TripState) -> dict:
        system = "Você escreve de forma clara, didática e organizada."
        human = f"""
RELATÓRIO FINAL\n
Use o plano (PLANEJAMENTO) e as entregas de HOSPEDAGEM, LAZER e ALIMENTAÇÃO para compor o texto final **(500–700 palavras)**.\n
Inclua:\n
- Introdução breve;\n
- Seções: **Hospedagem**, **Lazer**, **Alimentação** (incorpore tabelas/listas quando aplicável);\n
- Mini-roteiro sugerido **por dia** (alto nível);\n
- **Dicas rápidas** (transporte/segurança);\n
- Seção **Fontes** consolidada.\n
\n
Contexto:\n
- Destino: {state['destino']}\n
- Datas: {state['data_inicio']} a {state['data_fim']}\n
- Orçamento: {state['orcamento']}\n
- Preferências: {state['preferencias']}\n
\n
=== PLANEJAMENTO ===\n
{state.get('plan','')}\n
\n
=== HOSPEDAGEM ===\n
{state.get('hotels','')}\n
\n
=== LAZER ===\n
{state.get('leisure','')}\n
\n
=== ALIMENTAÇÃO ===\n
{state.get('food','')}
"""
        resp = get_llm(None).invoke([SystemMessage(content=system), HumanMessage(content=human)])
        return {"final": resp.content}

    return node

# ==========================
# Builder do grafo
# ==========================

def build_graph(llm: Any) -> Any:
    builder = StateGraph(TripState)
    builder.add_node("planner", make_planner_node(llm))
    builder.add_node("hotels", make_hotels_node(llm))
    builder.add_node("leisure", make_leisure_node(llm))
    builder.add_node("food", make_food_node(llm))
    builder.add_node("writer", make_writer_node(llm))

    builder.set_entry_point("planner")
    builder.add_edge("planner", "hotels")
    builder.add_edge("hotels", "leisure")
    builder.add_edge("leisure", "food")
    builder.add_edge("food", "writer")
    builder.add_edge("writer", END)

    return builder.compile()

# ==========================
# UI Streamlit (executada somente se Streamlit existir)
# ==========================
if STREAMLIT_AVAILABLE:
    st.set_page_config(page_title="Agentes de Viagem IA (LangGraph)", page_icon="🧭", layout="wide")
    st.title("🧭 Planejador de Viagens com LangGraph")
    st.markdown(
        """
Forneça os detalhes da sua viagem e deixe nossa **graph** de nós especializados criar um roteiro completo para você.
Os nós pesquisam hospedagem, lazer, gastronomia e consolidam tudo em um relatório final.
"""
    )
    st.divider()

    with st.form("travel_planner_form"):
        st.subheader("Preencha os dados da sua viagem:")
        col1, col2 = st.columns(2)
        with col1:
            destino = st.text_input("Destino (cidade, país)", placeholder="Ex.: Lisboa, Portugal")
            data_inicio = st.date_input("Data de início", value=date.today())
        with col2:
            orcamento = st.text_input("Orçamento aproximado (opcional)", placeholder="Ex.: R$ 5.000 no total")
            data_fim = st.date_input("Data de término", value=date.today() + timedelta(days=7))

        preferencias = st.text_area(
            "Preferências e observações (opcional)",
            placeholder=(
                "Ex: Gosto de museus e bairros históricos. Prefiro hotéis boutique. "
                "Tenho restrição a glúten."
            ),
        )

        st.markdown("---")
        col_api_1, col_api_2 = st.columns([2, 1])
        with col_api_1:
            api_key_input = st.text_input(
                "GROQ API Key (opcional — se ausente, usa variável de ambiente GROQ_API_KEY)",
                type="password",
            )
        with col_api_2:
            temperatura = st.slider("Temperatura", 0.0, 1.0, 0.2, 0.05)

        executar = st.form_submit_button("Gerar Roteiro de Viagem", use_container_width=True)

    if executar:
        ok, msg = validate_inputs(destino, data_inicio, data_fim)
        if not ok:
            st.error(msg)
            st.stop()

        llm = get_llm(api_key_input, temperature=temperatura)
        graph = build_graph(llm)

        with st.spinner(
            "Planejando sua viagem com LangGraph... Montando plano, hospedagem, lazer, alimentação e relatório final."
        ):
            final_state: TripState = graph.invoke(
                {
                    "destino": destino,
                    "data_inicio": str(data_inicio),
                    "data_fim": str(data_fim),
                    "orcamento": orcamento or "não informado",
                    "preferencias": preferencias or "não informado",
                }
            )

        st.success("Seu roteiro de viagem está pronto! ✅")
        aba_plano, aba_hosp, aba_alim, aba_lazer, aba_final = st.tabs(
            ["📋 Planejamento", "🏨 Hospedagem", "🍽️ Alimentação", "🎭 Lazer", "✨ Relatório Final"]
        )
        with aba_plano:
            st.subheader("Plano de Ação dos Nós")
            st.markdown(final_state.get("plan", ""))
        with aba_hosp:
            st.subheader("Pesquisa de Hospedagem")
            st.markdown(final_state.get("hotels", ""))
        with aba_alim:
            st.subheader("Recomendações Gastronômicas")
            st.markdown(final_state.get("food", ""))
        with aba_lazer:
            st.subheader("Sugestões de Lazer e Eventos")
            st.markdown(final_state.get("leisure", ""))
        with aba_final:
            st.subheader("Seu Roteiro de Viagem Personalizado")
            st.markdown(final_state.get("final", ""))

# ==========================
# CLI / Headless fallback (sem Streamlit)
# ==========================

def run_cli() -> None:
    print("[Modo CLI] Streamlit não detectado — rodando em modo texto.\n")
    destino = os.environ.get("DESTINO") or input("Destino (cidade, país): ").strip()
    data_inicio = os.environ.get("DATA_INICIO") or input("Data de início (AAAA-MM-DD): ").strip()
    data_fim = os.environ.get("DATA_FIM") or input("Data de término (AAAA-MM-DD): ").strip()
    orcamento = os.environ.get("ORCAMENTO", "não informado")
    preferencias = os.environ.get("PREFERENCIAS", "não informado")

    ok, msg = validate_inputs(destino, data_inicio, data_fim)
    if not ok:
        print(f"Erro: {msg}")
        sys.exit(2)

    temperatura = float(os.environ.get("TEMPERATURA", "0.2"))
    api_key = os.environ.get("GROQ_API_KEY")  # opcional
    llm = get_llm(api_key, temperature=temperatura)
    graph = build_graph(llm)

    final_state: TripState = graph.invoke(
        {
            "destino": destino,
            "data_inicio": str(data_inicio),
            "data_fim": str(data_fim),
            "orcamento": orcamento,
            "preferencias": preferencias,
        }
    )

    print("\n=== 📋 PLANEJAMENTO ===\n")
    print(final_state.get("plan", ""))
    print("\n=== 🏨 HOSPEDAGEM ===\n")
    print(final_state.get("hotels", ""))
    print("\n=== 🎭 LAZER ===\n")
    print(final_state.get("leisure", ""))
    print("\n=== 🍽️ ALIMENTAÇÃO ===\n")
    print(final_state.get("food", ""))
    print("\n=== ✨ RELATÓRIO FINAL ===\n")
    print(final_state.get("final", ""))

# ==========================
# Testes embutidos (sempre que RUN_TESTS=1)
# ==========================

def run_tests() -> None:
    print("Executando testes internos...")

    # 1) Validação de entradas
    ok, msg = validate_inputs("", "2025-01-01", "2025-01-02")
    assert not ok and msg, "Deveria falhar quando destino está vazio"

    ok, msg = validate_inputs("Porto", "2025-01-03", "2025-01-01")
    assert not ok and "término" in (msg or "").lower(), "Deveria detectar data_fim < data_inicio"

    ok, msg = validate_inputs("Lisboa", "2025-01-01", "2025-01-03")
    assert ok, f"Validação deveria passar, msg: {msg}"

    # 2) Execução do grafo com DummyLLM
    dummy = DummyLLM()
    graph = build_graph(dummy)
    state_in: TripState = {
        "destino": "Lisboa, Portugal",
        "data_inicio": "2025-01-01",
        "data_fim": "2025-01-05",
        "orcamento": "R$ 5.000",
        "preferencias": "Museus e bairros históricos",
    }
    state_out = graph.invoke(state_in)
    for k in ("plan", "hotels", "leisure", "food", "final"):
        assert state_out.get(k), f"Saída '{k}' não foi preenchida"

    # 3) Smoke test com get_llm() sem GROQ_API_KEY (deve cair no DummyLLM)
    llm = get_llm(api_key=None)
    assert hasattr(llm, "invoke"), "LLM retornado deve ter método invoke()"

    print("✅ Todos os testes passaram!")


if __name__ == "__main__":
    if os.environ.get("RUN_TESTS") == "1":
        run_tests()
    elif not STREAMLIT_AVAILABLE:
        run_cli()
    else:
        print(
            "Streamlit está disponível. Execute a UI com: \n  streamlit run app.py\n"
        )
