import streamlit as st
import time
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import requests

# --- Inicialização de Estado ---
def init_state():
    st.session_state.setdefault("game_state", "menu")
    for key in [
        "phase1_passed", "phase2_passed", "phase3_passed", "phase4_passed",
        "show_phase1_feedback", "show_phase2_feedback", "show_phase3_feedback", "show_phase4_feedback"
    ]:
        st.session_state.setdefault(key, False)
    for key in ["p1_attempts", "p2_attempts", "p3_attempts", "p4_attempts"]:
        st.session_state.setdefault(key, 0)

init_state()

# --- Logs de Feedback ---
LOG_FILE = "game_feedback.log"

def log_feedback(feedback_text):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"[{timestamp}] {feedback_text}\n")
        st.success("Obrigado pelo seu feedback! Ele foi registrado.")
    except Exception as e:
        st.error(f"Erro ao salvar feedback: {e}")

# --- Função lateral de bug/sugestão ---
def report_bug_section():
    st.sidebar.subheader("🐞 Reportar Erro / Sugestão")
    with st.sidebar.form("bug_report_form", clear_on_submit=True):
        bug_text = st.text_area("Descreva o erro que encontrou ou sua sugestão de melhoria:")
        submitted = st.form_submit_button("Enviar Feedback ✉️")
        if submitted:
            if bug_text.strip():
                log_feedback(bug_text)
            else:
                st.sidebar.warning("Por favor, escreva algo antes de enviar.")

# --- Fase 1: Mini-game de Montagem do Transformer ---
def phase1_architecture():
    st.header("Fase 1: A Arquitetura Fundacional (Encoder-Decoder) 🏗️")
    st.write("Arraste os blocos abaixo para a ordem correta da arquitetura Transformer: de entrada até a saída.")

    componentes = [
        "Mecanismo de Atenção",
        "Camada de Saída",
        "Decoder",
        "Encoder",
        "Embedding"
    ]  # Ordem embaralhada para evitar sugestão direta

    ordem_correta = [
        "Embedding", "Encoder", "Mecanismo de Atenção", "Decoder", "Camada de Saída"
    ]

    dicas = [
        "Posição 1 - Transforma cada palavra em um vetor numérico compreensível pela IA.",
        "Posição 2 - Processa a frase de entrada e gera uma representação contextualizada.",
        "Posição 3 - Decide quais palavras são mais importantes umas para as outras.",
        "Posição 4 - Gera a frase de saída, com base na atenção e no encoder.",
        "Posição 5 - Traduz a saída do decoder para palavras compreensíveis."
    ]

    escolhas = []
    for i in range(len(ordem_correta)):
        st.markdown(dicas[i])
        escolha = st.selectbox(f"Escolha para a posição {i + 1}", ["⬇️ Escolha"] + componentes, key=f"fase1_{i}")
        escolhas.append(escolha)

    if st.button("Verificar Ordem"):
        if escolhas == ordem_correta:
            st.session_state.phase1_passed = True
            st.session_state.show_phase1_feedback = True
            st.rerun()
        else:
            st.error("❌ Ainda não está certo! Tente organizar os blocos na sequência lógica.")

    if st.session_state.get("show_phase1_feedback", False):
        st.success("✅ Correto! Essa é a ordem de processamento do Transformer.")
        st.image("img/transformer.png", width=300, caption="Arquitetura do Transformer: Encoder-Decoder com Atenção")
        if st.button("Avançar para Fase 2 ➡️", key="p1_advance_button"):
            st.session_state.game_state = "phase2"
            st.session_state.show_phase1_feedback = False
            st.rerun()

    report_bug_section()

# --- Fase 2 ---
def phase2_scaled_dot_product_attention():
    st.header("Fase 2: Corrida de Vetores e Escalonamento 🎯")
    st.write("Vamos entender como a atenção funciona usando vetores. Aqui, você controla os vetores **Q (Query)**, **K (Key)** e o parâmetro **dₖ** (dimensão da chave).")

    with st.expander("🤔 O que são Q, K e dₖ?"):
        st.markdown("""
- **Q (Query - Consulta)**: representa o vetor da palavra que está pedindo informação. Ele pergunta: “quais palavras são relevantes para mim?”
- **K (Key - Chave)**: representa cada uma das outras palavras que podem ser relevantes.
- **dₖ (dimensão da chave)**: controla o tamanho dos vetores Q e K. Serve para normalizar o cálculo de similaridade.

O cálculo da atenção é feito assim:
```Attention(Q, K, V) = softmax(Q·Kᵗ / √dₖ)·V```
Se Q·K for muito grande, a softmax se satura e os gradientes viram quase zero. A divisão por √dₖ evita isso.
        """)

    q_val = st.slider("Valor do vetor Q (intensidade da consulta)", 1, 100, 60, step=1)
    k_val = st.slider("Valor do vetor K (intensidade da chave)", 1, 100, 80, step=1)
    d_k = st.slider("Dimensão dₖ (escalonador, tamanho dos vetores)", 1, 100, 64, step=1)

    produto = q_val * k_val
    com_escalonamento = produto / (d_k ** 0.5)

    st.markdown(f"**Produto Escalar (Q·K):** `{produto}`")
    st.markdown(f"**Escalonado (÷ √dₖ):** `{com_escalonamento:.2f}`")

    if com_escalonamento < 30:
        st.success("✅ Perfeito! O escalonamento protege a função softmax de saturar, garantindo gradientes estáveis.")
        if st.button("Avançar para Fase 3 ➡️", key="p2_advance_button"):
            st.session_state.game_state = "phase3"
            st.rerun()
    else:
        st.warning("⚠️ O valor escalonado ainda está alto. Isso pode saturar a softmax e impedir o modelo de aprender corretamente. Reduza Q, K ou aumente dₖ.")

    report_bug_section()

# --- Fase 3 ---
def phase3_multi_head_attention():
    st.header("Fase 3: Cobrinha Multi-Cabeça 🐍")
    st.write("Neste mini-jogo, você vai entender como diferentes 'cabeças' de atenção podem focar em diferentes partes da entrada.")
    st.write("Cada cabeça deve capturar um tipo de informação em uma frase simplificada.")

    palavras = ["João", "correu", "até", "a", "loja"]
    opcoes = ["sintaxe", "semântica", "posição"]

    colunas = st.columns(len(palavras))
    atribuicoes = []
    for i in range(len(palavras)):
        with colunas[i]:
            escolha = st.selectbox(f"'{palavras[i]}'", ["--"] + opcoes, key=f"fase3_{i}")
            atribuicoes.append(escolha)

    if st.button("Verificar Cabeças"):
        tipos_usados = set(atribuicoes)
        if all(e in tipos_usados for e in opcoes):
            st.success("✅ Excelente! Cada cabeça está capturando uma dimensão diferente da frase.")
            if st.button("Avançar para Fase 4 ➡️", key="p3_advance_button"):
                st.session_state.game_state = "phase4"
                st.rerun()
        else:
            st.warning("⚠️ Tente distribuir as atenções entre sintaxe, semântica e posição para as palavras.")

    report_bug_section()

# --- Fase 4 ---
def phase4_positional_encoding():
    st.header("Fase 4: Corrida com Codificação Posicional 📍")
    st.write("Aqui, seu token precisa seguir um caminho com saltos senoidais para escapar dos obstáculos.")
    st.write("O objetivo é simular o comportamento da codificação posicional seno/cosseno usada no Transformer.")

    num_pontos = st.slider("Tamanho da sequência (tokens)", 5, 50, 20)
    uso_seno = st.radio("Tipo de codificação posicional:", ["Constante", "Linear", "Senoidal"], index=2)

    x = np.arange(num_pontos)
    if uso_seno == "Constante":
        y = np.ones_like(x)
    elif uso_seno == "Linear":
        y = x
    else:
        y = np.sin(x / 5)

    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o')
    ax.set_title("Caminho da posição na sequência")
    st.pyplot(fig)

    if uso_seno == "Senoidal":
        st.success("✅ Correto! Funções seno e cosseno codificam posições relativas no Transformer.")
        if st.button("Avançar para Fase 5 ➡️", key="p4_advance_button"):
            st.session_state.game_state = "phase5"
            st.rerun()
    else:
        st.warning("⚠️ A codificação correta é senoidal. Tente novamente.")

    report_bug_section()

# --- Fase 5 ---
def phase5_training_results():
    st.header("Fase 5: Gerencie seu Treinamento ⚙️")
    st.write("Nesta fase final, você é responsável por treinar seu Transformer com os melhores parâmetros.")
    st.write("Faça escolhas que equilibrem desempenho (BLEU score) e custo computacional.")

    modelo = st.selectbox("Tamanho do modelo", ["Pequeno", "Base", "Grande"], index=1)
    num_cabecas = st.slider("Número de cabeças de atenção", 2, 16, 8)
    batch_size = st.slider("Tamanho do batch (lote)", 8, 128, 32, step=8)

    if modelo == "Pequeno":
        base_bleu = 25.0
        custo = 1.0
    elif modelo == "Base":
        base_bleu = 27.3
        custo = 2.5
    else:
        base_bleu = 28.4
        custo = 5.0

    ajuste = (num_cabecas / 8) * (batch_size / 32)
    bleu = base_bleu + np.log2(ajuste + 1)
    custo_total = custo * ajuste

    st.markdown(f"**BLEU estimado:** `{bleu:.2f}`")
    st.markdown(f"**Custo estimado (FLOPs):** `{custo_total:.2f}` unidades")

    if bleu >= 28.0:
        st.success("✅ Parabéns! Você otimizou bem seu Transformer.")
        if st.button("Ver Resumo das Descobertas 🏆"):
            st.session_state.game_state = "summary"
            st.rerun()
    else:
        st.warning("⚠️ Seu BLEU score ainda pode melhorar. Tente ajustar os parâmetros!")

    report_bug_section()

# --- Resumo Final + LLM ---
def game_summary():
    st.header("Resumo: Attention Is All You Need 🎉")
    st.write("Você concluiu todas as fases e compreendeu os principais pilares da arquitetura Transformer. Veja abaixo um resumo aprofundado, totalmente alinhado ao artigo original de Vaswani et al. (2017):")

    st.markdown("""
**1. Arquitetura baseada exclusivamente em atenção**  
O Transformer elimina redes recorrentes (RNNs) e convolucionais (CNNs), usando atenção como base para capturar dependências entre tokens, permitindo paralelismo.

**2. Treinamento altamente paralelizável**  
Sem processar um token por vez, o modelo acelera o treinamento usando processamento simultâneo em GPUs/TPUs.

**3. Mecanismo de auto-atenção (self-attention)**  
Cada palavra pondera sua relação com todas as outras da sequência, produzindo uma representação contextual rica.

**4. Produto escalar escalonado**  
Para evitar saturação da softmax e gradientes instáveis, o produto escalar é dividido por √dₖ.

**5. Atenção multi-cabeça**  
Múltiplas cabeças processam diferentes subespaços de atenção em paralelo (sintaxe, semântica etc.).

**6. Codificação posicional senoidal**  
Como não há ordem natural, a posição é codificada usando seno e cosseno de frequências diferentes.

**7. Resultados em BLEU**  
No WMT 2014, o Transformer superou todos os modelos anteriores com BLEU score 27.3 (base) e 28.4 (big).

**8. Generalização para outras tarefas**  
Sua estrutura inspirou modelos como BERT, GPT, T5 — aplicados em muitas tarefas de NLP.
""")

    st.markdown("---")
    st.subheader("❓ Pergunte sobre Transformers")
    st.write("Use o assistente abaixo para tirar dúvidas sobre o conteúdo do jogo!")

    pergunta = st.text_area("Digite sua pergunta para o modelo Falcon-7B-Instruct:", key="qa_final")
    if st.button("Responder", key="qa_submit"):
        if pergunta.strip():
            with st.spinner("Gerando resposta..."):
                resposta = requests.post(
                    "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct",
                    json={"inputs": f"Pergunta: {pergunta}\\nResposta:"}
                )
                if resposta.status_code == 200:
                    saida = resposta.json()[0].get("generated_text", "")
                    st.markdown(saida)
                else:
                    st.error("Não foi possível gerar uma resposta agora.")
        else:
            st.warning("Digite sua pergunta antes de enviar.")

    if st.button("Jogar Novamente 🔁"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    report_bug_section()

# --- Menu Inicial ---
def main_menu():
    st.title("🚀 A Jornada do Transformer: Atenção Desvendada! 🚀")
    st.write("Bem-vindo, engenheiro de inteligência artificial! Sua missão é guiar um modelo Transformer por cinco fases...")
    try:
        st.image("img/transformer.png", width=200)
    except:
        st.warning("Imagem não encontrada.")
    if st.button("Iniciar Missão ➡️"):
        st.session_state.game_state = "phase1"
        st.rerun()
    report_bug_section()

# --- Navegação ---
st.write(f"🧭 Estado atual: {st.session_state.game_state}")
if st.session_state.game_state == "menu":
    main_menu()
elif st.session_state.game_state == "phase1":
    phase1_architecture()
elif st.session_state.game_state == "phase2":
    phase2_scaled_dot_product_attention()
elif st.session_state.game_state == "phase3":
    phase3_multi_head_attention()
elif st.session_state.game_state == "phase4":
    phase4_positional_encoding()
elif st.session_state.game_state == "phase5":
    phase5_training_results()
elif st.session_state.game_state == "summary":
    game_summary()
