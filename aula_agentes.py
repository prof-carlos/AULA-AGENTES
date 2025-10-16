import os
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM

# ---------------------------
# UI
# ---------------------------
st.header("📚 Agentes de Estudo")
st.write("Informe o tema e gere material didático automaticamente:")

tema = st.text_input("Tema de estudo", placeholder="Ex.: Algoritmos de Busca, Fotossíntese, Juros Compostos")
nivel = st.text_input("Público/nível (opcional)", placeholder="Ex.: iniciante, ensino médio, graduação, profissional")
objetivo = st.text_area("Objetivo (opcional)", placeholder="Ex.: entender conceitos básicos e aplicar em exercícios simples")

# NOVO: toggle para gabarito
mostrar_gabarito = st.toggle("Gerar e mostrar gabarito (respostas + justificativas)", value=True)

executar = st.button("Gerar material")
api_key = 'SUA_CHAVE_API'

if executar:
    if not api_key or not tema:
        st.error("Por favor, informe a API key e o tema de estudo.")
        st.stop()

    # ---------------------------
    # LLM (Groq / Llama 3.3 70B)
    # ---------------------------
    llm = LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.3
    )

    # ---------------------------
    # Agentes
    # ---------------------------
    agente_resumo = Agent(
        role="Redator(a) de Resumo Didático",
        goal=(
            "Escrever um RESUMO claro e didático sobre {tema} para o público {nivel}, "
            "alinhado ao objetivo {objetivo}. "
            "A linguagem deve ser direta, com contexto prático e sem jargões desnecessários."
        ),
        backstory="Você transforma temas técnicos/Acadêmicos em explicações curtas e precisas.",
        llm=llm, verbose=False
    )

    agente_exemplos = Agent(
        role="Criador(a) de Exemplos Contextualizados",
        goal=(
            "Gerar 4 EXEMPLOS CURTOS sobre {tema}, cada um com contexto realista. "
            "Cada exemplo com título (em negrito), cenário, dados (se houver), aplicação e resultado."
        ),
        backstory="Você mostra o conceito em ação com exemplos breves e concretos.",
        llm=llm, verbose=False
    )

    agente_exercicios = Agent(
        role="Autor(a) de Exercícios Práticos",
        goal=(
            "Criar 3 EXERCÍCIOS SIMPLES sobre {tema}. "
            "Variar formato (múltipla escolha, V/F, completar, resolução curta). "
            "Enunciados claros. NÃO incluir respostas."
        ),
        backstory="Você cria atividades rápidas que fixam os conceitos essenciais.",
        llm=llm, verbose=False
    )

    # Opcional: agente de gabarito (só se toggle estiver ligado)
    if mostrar_gabarito:
        agente_gabarito = Agent(
            role="Revisor(a) e Gabaritador(a)",
            goal=(
                "Ler os EXERCÍCIOS sobre {tema} e produzir o GABARITO oficial, "
                "com respostas corretas e justificativa breve (1–2 frases) por item."
            ),
            backstory="Você confere consistência e explica rapidamente o porquê da resposta.",
            llm=llm, verbose=False
        )

    # ---------------------------
    # Tarefas
    # ---------------------------
    t_resumo = Task(
        description=(
            "RESUMO\n"
            "Escreva em PT-BR um resumo didático sobre {tema} para o nível {nivel} e objetivo {objetivo}. "
            "Inclua: definição (2–3 frases), por que importa (1–2), onde se aplica (1–2) e 3–5 ideias-chave em bullets. "
            "150–220 palavras. Formate em Markdown com título."
        ),
        agent=agente_resumo,
        expected_output="Resumo em Markdown com título, parágrafos curtos e 3–5 bullets."
    )

    t_exemplos = Task(
        description=(
            "EXEMPLOS\n"
            "Produza 4 exemplos curtos e contextualizados sobre {tema}. "
            "Padrão (até 5 linhas cada): **Título**; cenário; dados/entrada; como aplicar (1–2 frases); resultado."
        ),
        agent=agente_exemplos,
        expected_output="Lista numerada (1–4) em Markdown com exemplos curtos e completos."
    )

    t_exercicios = Task(
        description=(
            "EXERCÍCIOS\n"
            "Crie 3 exercícios simples sobre {tema} em PT-BR. "
            "Varie formatos e não inclua respostas. "
            "Entregue lista numerada (1 a 3) em Markdown."
        ),
        agent=agente_exercicios,
        expected_output="Lista numerada (1–3) com exercícios simples, sem respostas."
    )

    # Tarefa de gabarito condicionada
    if mostrar_gabarito:
        t_gabarito = Task(
            description=(
                "GABARITO\n"
                "Com base nos EXERCÍCIOS fornecidos no contexto, produza as respostas corretas dos itens 1–3. "
                "Para cada item, dê:\n"
                "- **Resposta:** (letra/valor/solução) \n"
                "- **Comentário:** justificativa breve e direta (1–2 frases), citando o conceito-chave.\n"
                "Formato: lista numerada (1 a 3) em Markdown."
            ),
            agent=agente_gabarito,
            expected_output="Lista numerada (1–3) com resposta e comentário por exercício.",
            context=[t_exercicios]
        )

    # ---------------------------
    # Orquestração
    # ---------------------------
    agents = [agente_resumo, agente_exemplos, agente_exercicios]
    tasks = [t_resumo, t_exemplos, t_exercicios]
    if mostrar_gabarito:
        agents.append(agente_gabarito)
        tasks.append(t_gabarito)

    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
    )

    crew.kickoff(inputs={
        "tema": tema,
        "nivel": nivel or "não informado",
        "objetivo": objetivo or "não informado",
    })

    # ---------------------------
    # Exibição
    # ---------------------------
    resumo_out = getattr(t_resumo, "output", None) or getattr(t_resumo, "result", "") or ""
    exemplos_out = getattr(t_exemplos, "output", None) or getattr(t_exemplos, "result", "") or ""
    exercicios_out = getattr(t_exercicios, "output", None) or getattr(t_exercicios, "result", "") or ""
    gabarito_out = ""
    if mostrar_gabarito:
        gabarito_out = getattr(t_gabarito, "output", None) or getattr(t_gabarito, "result", "") or ""

    # Abas condicionais
    if mostrar_gabarito:
        aba_resumo, aba_exemplos, aba_exercicios, aba_gabarito = st.tabs(
            ["Resumo", "Exemplos", "Exercícios", "Gabarito"]
        )
    else:
        aba_resumo, aba_exemplos, aba_exercicios = st.tabs(
            ["Resumo", "Exemplos", "Exercícios"]
        )

    with aba_resumo:
        st.markdown(resumo_out)
    with aba_exemplos:
        st.markdown(exemplos_out)
    with aba_exercicios:
        st.markdown(exercicios_out)
    if mostrar_gabarito:
        with aba_gabarito:
            st.markdown(gabarito_out)
