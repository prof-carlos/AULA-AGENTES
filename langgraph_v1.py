# pip install streamlit langgraph langchain-core langchain-groq

import os
from datetime import date, timedelta
from typing import TypedDict

import streamlit as st

# LangChain / LangGraph
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# ==========================
# UI CONFIG
# ==========================
st.set_page_config(
    page_title="Agentes de Viagem IA (LangGraph)",
    page_icon="🧭",
    layout="wide",
)

st.title("🧭 Planejador de Viagens com LangGraph")
st.markdown(
    """
Forneça os detalhes da sua viagem e deixe nossa **graph** de nós especializados criar um roteiro completo para você.
Os nós pesquisam hospedagem, lazer, gastronomia e consolidam tudo em um relatório final.
"""
)
st.divider()

# ==========================
# LLM FACTORY
# ==========================
@st.cache_resource(show_spinner=False)
def get_llm(api_key: str | None, temperature: float = 0.2):
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "Defina a GROQ_API_KEY no ambiente, em st.secrets, ou informe no campo de API Key."
        )
    return ChatGroq(
        api_key=key,
        model_name="llama-3.3-70b-versatile",
        temperature=temperature,
    )

# ==========================
# STATE (LangGraph)
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
# NODE BUILDERS
# ==========================

def make_planner_node(llm: ChatGroq):
    def node(state: TripState) -> dict:
        system = (
            "Você estrutura planos objetivos e práticos em 3 passos fixos."
        )
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
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        return {"plan": resp.content}

    return node


def make_hotels_node(llm: ChatGroq):
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
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        return {"hotels": resp.content}

    return node


def make_leisure_node(llm: ChatGroq):
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
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        return {"leisure": resp.content}

    return node


def make_food_node(llm: ChatGroq):
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
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        return {"food": resp.content}

    return node


def make_writer_node(llm: ChatGroq):
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
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        return {"final": resp.content}

    return node

# ==========================
# STREAMLIT FORM
# ==========================
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

# ==========================
# EXECUTION (LangGraph)
# ==========================
if executar:
    # Validações
    if not destino or not data_inicio or not data_fim:
        st.error("Por favor, informe o destino e as datas da viagem para continuar.")
        st.stop()
    if data_fim < data_inicio:
        st.error("A data de término deve ser posterior à data de início.")
        st.stop()

    try:
        llm = get_llm(api_key_input, temperature=temperatura)
    except Exception as e:
        st.error(f"Erro ao inicializar o LLM: {e}")
        st.stop()

    # Build graph (sequential)
    builder = StateGraph(TripState)

    planner_node = make_planner_node(llm)
    hotels_node = make_hotels_node(llm)
    leisure_node = make_leisure_node(llm)
    food_node = make_food_node(llm)
    writer_node = make_writer_node(llm)

    builder.add_node("planner", planner_node)
    builder.add_node("hotels", hotels_node)
    builder.add_node("leisure", leisure_node)
    builder.add_node("food", food_node)
    builder.add_node("writer", writer_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "hotels")
    builder.add_edge("hotels", "leisure")
    builder.add_edge("leisure", "food")
    builder.add_edge("food", "writer")
    builder.add_edge("writer", END)

    graph = builder.compile()

    initial_state: TripState = {
        "destino": destino,
        "data_inicio": str(data_inicio),
        "data_fim": str(data_fim),
        "orcamento": orcamento or "não informado",
        "preferencias": preferencias or "não informado",
    }

    with st.spinner(
        "Planejando sua viagem com LangGraph... Montando plano, hospedagem, lazer, alimentação e relatório final."
    ):
        final_state: TripState = graph.invoke(initial_state)

    # ==========================
    # UI OUTPUT
    # ==========================
    st.success("Seu roteiro de viagem está pronto! ✅")

    plano = final_state.get("plan", "")
    hospedagem_out = final_state.get("hotels", "")
    lazer_out = final_state.get("leisure", "")
    alimentacao_out = final_state.get("food", "")
    final_out = final_state.get("final", "")

    aba_plano, aba_hosp, aba_alim, aba_lazer, aba_final = st.tabs(
        ["📋 Planejamento", "🏨 Hospedagem", "🍽️ Alimentação", "🎭 Lazer", "✨ Relatório Final"]
    )

    with aba_plano:
        st.subheader("Plano de Ação dos Nós")
        with st.container(border=True):
            st.markdown(plano)
    with aba_hosp:
        st.subheader("Pesquisa de Hospedagem")
        with st.container(border=True):
            st.markdown(hospedagem_out)
    with aba_alim:
        st.subheader("Recomendações Gastronômicas")
        with st.container(border=True):
            st.markdown(alimentacao_out)
    with aba_lazer:
        st.subheader("Sugestões de Lazer e Eventos")
        with st.container(border=True):
            st.markdown(lazer_out)
    with aba_final:
        st.subheader("Seu Roteiro de Viagem Personalizado")
        with st.container(border=True):
            st.markdown(final_out)

# ==========================
# Notas:
# - Instale dependências: pip install streamlit langgraph langchain-core langchain-groq
# - Defina GROQ_API_KEY como variável de ambiente ou informe no campo do app.
# - A graph está sequencial (planner -> hotels -> leisure -> food -> writer). Você pode paralelizar
#   alguns nós (ex.: hotels/leisure/food) criando junções e condicionais conforme necessário.
