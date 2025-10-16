import os
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from datetime import date, timedelta # 1. IMPORTAÇÃO CORRIGIDA

# ---------------------------
# CONFIGURAÇÃO DA PÁGINA (UI)
# ---------------------------
st.set_page_config(
    page_title="Agentes de Viagem IA",
    page_icon="🧭",
    layout="wide"
)

# --- CABEÇALHO E DESCRIÇÃO ---
st.title("🧭 Planejador de Viagens com Agentes IA")
st.markdown("""
Forneça os detalhes da sua viagem e deixe nossa equipe de agentes especializados criar um roteiro completo para você.
Eles pesquisarão hospedagem, lazer, gastronomia e compilarão tudo em um relatório final.
""")
st.divider()

# ---------------------------
# FORMULÁRIO DE ENTRADA (UI)
# ---------------------------
with st.form("travel_planner_form"):
    st.subheader("Preencha os dados da sua viagem:")

    # Organiza os campos principais em colunas
    col1, col2 = st.columns(2)
    with col1:
        destino = st.text_input("Destino (cidade, país)", placeholder="Ex.: Lisboa, Portugal")
        data_inicio = st.date_input("Data de início", value=date.today())
    with col2:
        orcamento = st.text_input("Orçamento aproximado (opcional)", placeholder="Ex.: R$ 5.000 no total")
        # 2. LINHA CORRIGIDA PARA USAR timedelta
        data_fim = st.date_input("Data de término", value=date.today() + timedelta(days=7))

    preferencias = st.text_area(
        "Preferências e observações (opcional)",
        placeholder="Ex: Gosto de museus e bairros históricos. Prefiro hotéis boutique. Tenho restrição a glúten."
    )
    
    # 3. BOTÃO DE SUBMISSÃO ADICIONADO DENTRO DO FORMULÁRIO
    executar = st.form_submit_button("Gerar Roteiro de Viagem", use_container_width=True)


# ---------------------------
# LÓGICA DE EXECUÇÃO
# ---------------------------
if executar:
    # Validação dos campos obrigatórios
    if not destino or not data_inicio or not data_fim:
        st.error("Por favor, informe o destino e as datas da viagem para continuar.")
        st.stop()
    if data_fim < data_inicio:
        st.error("A data de término deve ser posterior à data de início.")
        st.stop()

    # Feedback visual para o usuário enquanto os agentes trabalham
    with st.spinner("Planejando sua viagem... Os agentes estão pesquisando as melhores opções. Isso pode levar um momento."):
        
        # O restante do seu código (LLM, Agentes, Tarefas, Crew) permanece aqui...
        # ...
        # [CÓDIGO DOS AGENTES E TAREFAS SEM ALTERAÇÃO]
        # ...
        
        # ---------------------------
        # LLM (Groq / Llama 3.3 70B)
        # ---------------------------
        llm = LLM(
            model="groq/llama-3.3-70b-versatile",
            api_key='SUA_CHAVE_API', # Lembre-se de substituir pela sua chave
            temperature=0.2
        )
    
        # ---------------------------
        # Agentes especializados
        # ---------------------------
        planejador = Agent(
            role="Roteirista de Viagens",
            goal=(
                "Gerar um plano de pesquisa para {destino} entre {data_inicio} e {data_fim}, "
                "dividindo em EXATAMENTE 3 subtarefas numeradas: "
                "1) HOSPEDAGEM; 2) LAZER; 3) ALIMENTAÇÃO. "
                "Considerar orçamento ({orcamento}) e preferências ({preferencias})."
            ),
            backstory="Você estrutura planos objetivos e práticos em 3 passos fixos.",
            llm=llm, verbose=False
        )
    
        agente_hospedagem = Agent(
            role="Especialista em Hospedagem",
            goal=(
                "Listar hotéis para {destino} com foco em localização e custo-benefício "
                "nas datas {data_inicio}–{data_fim}. "
                "ENTREGAR uma tabela Markdown com as colunas: Nome | Endereço | Site | Telefone. "
                "Incluir 5–8 opções e 2–4 fontes (título + URL)."
            ),
            backstory="Você verifica informações de hotéis e organiza dados de contato.",
            llm=llm, verbose=False
        )
    
        agente_lazer = Agent(
            role="Especialista em Lazer",
            goal=(
                "Sugerir 8–12 pontos turísticos ESSENCIAIS em {destino} "
                "e 3–5 eventos que ocorram entre {data_inicio} e {data_fim}. "
                "Para cada item, incluir breve descrição e link. "
                "Formatar em listas; ao final, 2–4 fontes (título + URL)."
            ),
            backstory="Você encontra atrações e eventos relevantes às datas.",
            llm=llm, verbose=False
        )
    
        agente_alimentacao = Agent(
            role="Especialista em Gastronomia",
            goal=(
                "Recomendar 8–12 restaurantes em {destino} (com bairro e site) "
                "e listar 5–8 comidas típicas locais com breve explicação. "
                "Entregar restaurantes em tabela Markdown: Nome | Bairro | Faixa de Preço | Cozinha | Site. "
                "Finalizar com 2–4 fontes (título + URL)."
            ),
            backstory="Você conhece bem a cena gastronômica e as especialidades locais.",
            llm=llm, verbose=False
        )
    
        redator = Agent(
            role="Redator de Roteiro",
            goal=(
                "Usar o plano do Roteirista e as entregas de Hospedagem, Lazer e Alimentação "
                "para compor o RELATÓRIO FINAL (500–700 palavras) com: "
                "introdução breve, 3 seções (Hospedagem, Lazer, Alimentação) incorporando tabelas/listas, "
                "mini-roteiro sugerido por dia (alto nível), dicas rápidas (transporte/segurança), "
                "e uma seção 'Fontes' consolidada."
            ),
            backstory="Você escreve de forma clara, didática e organizada.",
            llm=llm, verbose=False
        )
    
        # ---------------------------
        # Tarefas
        # ---------------------------
        t1 = Task(
            description=(
                "PLANEJAMENTO GERAL\n"
                "Destino: {destino}\nDatas: {data_inicio} a {data_fim}\n"
                "Orçamento: {orcamento}\nPreferências: {preferencias}\n\n"
                "1) Defina EXATAMENTE 3 subtarefas numeradas: HOSPEDAGEM, LAZER, ALIMENTAÇÃO (cada uma com 1 frase).\n"
                "2) Liste critérios de seleção (3–5 bullets) considerando orçamento e preferências.\n"
                "3) Escreva uma justificativa de 1–2 linhas."
            ),
            agent=planejador,
            expected_output="3 subtarefas numeradas + critérios (bullets) + justificativa."
        )
    
        t2 = Task(
            description=(
                "HOSPEDAGEM\n"
                "Usando o plano do Roteirista, pesquise hotéis para {destino} nas datas {data_inicio}–{data_fim}.\n"
                "Entregue TABELA Markdown com: Nome | Endereço | Site | Telefone (DDI se disponível). "
                "Inclua 5–8 opções e 2–4 fontes (título + URL)."
            ),
            agent=agente_hospedagem,
            expected_output="Tabela Markdown de hotéis + 2–4 fontes."
        )
    
        t3 = Task(
            description=(
                "LAZER\n"
                "Listar 8–12 pontos turísticos IMPERDÍVEIS em {destino} e 3–5 eventos nas datas {data_inicio}–{data_fim}. "
                "Incluir breve descrição e link por item. "
                "Feche com 2–4 fontes (título + URL)."
            ),
            agent=agente_lazer,
            expected_output="Listas de atrações e eventos + 2–4 fontes."
        )
    
        t4 = Task(
            description=(
                "ALIMENTAÇÃO\n"
                "Recomendar 8–12 restaurantes (Nome | Bairro | Faixa de Preço | Cozinha | Site em TABELA Markdown) "
                "e 5–8 comidas típicas com breve explicação. "
                "Feche com 2–4 fontes (título + URL)."
            ),
            agent=agente_alimentacao,
            expected_output="Tabela de restaurantes + lista de comidas típicas + 2–4 fontes."
        )
    
        t5 = Task(
            description=(
                "RELATÓRIO FINAL\n"
                "Usando o plano (t1) e as entregas de hospedagem (t2), lazer (t3) e alimentação (t4), "
                "entregue um texto final (500–700 palavras) com: introdução, seções de Hospedagem/Lazer/Alimentação "
                "(incorporando tabelas/listas quando aplicável), mini-roteiro por dia, dicas rápidas e seção 'Fontes'."
            ),
            agent=redator,
            expected_output="Relatório final organizado e pronto para o usuário."
        )
    
        # ---------------------------
        # Orquestração
        # ---------------------------
        crew = Crew(
            agents=[planejador, agente_hospedagem, agente_lazer, agente_alimentacao, redator],
            tasks=[t1, t2, t3, t4, t5],
            process=Process.sequential,
        )
    
        crew.kickoff(inputs={
            "destino": destino,
            "data_inicio": str(data_inicio),
            "data_fim": str(data_fim),
            "orcamento": orcamento or "não informado",
            "preferencias": preferencias or "não informado",
        })
    
        # ---------------------------
        # Exibição dos Resultados (UI)
        # ---------------------------
        st.success("Seu roteiro de viagem está pronto!")
    
        plano = getattr(t1, "output", None) or getattr(t1, "result", "") or ""
        hospedagem_out = getattr(t2, "output", None) or getattr(t2, "result", "") or ""
        lazer_out = getattr(t3, "output", None) or getattr(t3, "result", "") or ""
        alimentacao_out = getattr(t4, "output", None) or getattr(t4, "result", "") or ""
        final_out = getattr(t5, "output", None) or getattr(t5, "result", "") or ""
    
        aba_plano, aba_hosp, aba_alim, aba_lazer, aba_final = st.tabs(
            ["📋 Planejamento", "🏨 Hospedagem",  "🍽️ Alimentação", "🎭 Lazer", "✨ Relatório Final"]
        )
        
        with aba_plano:
            st.subheader("Plano de Ação dos Agentes")
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
