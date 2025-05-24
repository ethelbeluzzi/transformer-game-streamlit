import streamlit as st
import time
import datetime
import os

# --- Inicialização de Estado ---
def init_state():
    st.session_state.setdefault("game_state", "menu")
    st.session_state.setdefault("phase1_passed", False)
    st.session_state.setdefault("phase2_passed", False)
    st.session_state.setdefault("phase3_passed", False)
    st.session_state.setdefault("phase4_passed", False)
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

# --- Fase 1 ---
def phase1_architecture():
    st.header("Fase 1: A Arquitetura Fundacional (Encoder-Decoder) 🏗️")
    st.write("Antes do Transformer, os modelos mais comuns para lidar com dados sequenciais eram as redes neurais recorrentes (RNNs) e convolucionais (CNNs). No entanto, ambos possuem limitações sérias quando se trata de capturar dependências de longo prazo em sequências e realizar o treinamento de forma paralela. O Transformer revoluciona essa abordagem ao eliminar completamente a necessidade de recorrência ou convolução, confiando exclusivamente no mecanismo de atenção.")

    with st.expander("🤔 O que é Encoder-Decoder?"):
        st.write("Essa arquitetura é composta por duas partes principais: o codificador (encoder), que processa a entrada e gera uma representação interna (vetor de contexto), e o decodificador (decoder), que utiliza essa representação para gerar uma saída. Em tarefas como tradução automática, o encoder lê a frase em uma língua e o decoder gera a tradução na outra língua.")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Redes Recorrentes Complexas 🔄", key="p1_btn_rnn"):
            st.error("❌ Resposta incorreta! As RNNs são difíceis de treinar para sequências longas e têm limitações de paralelização.")
            st.session_state.p1_attempts += 1
    with col2:
        if st.button("Redes Convolucionais 🖼️", key="p1_btn_cnn"):
            st.warning("⚠️ Parcialmente correto. As CNNs oferecem alguma paralelização, mas não capturam dependências de longo prazo de maneira eficiente.")
            st.session_state.p1_attempts += 1
    with col3:
        if st.button("Atenção Pura (Transformer) ✨", key="p1_btn_attention"):
            st.session_state.phase1_passed = True
            st.rerun()

    if st.session_state.phase1_passed:
        st.success("✅ Correto! O Transformer utiliza atenção pura — sem RNNs nem CNNs.")
        st.write("O Transformer usa pilhas de atenção e feedforward em blocos separados para codificação e decodificação, o que permite paralelização total e melhor desempenho.")
        st.image("img/transformer.png", width=300, caption="Diagrama da arquitetura do Transformer")
        if st.button("Avançar para Fase 2 ➡️", key="p1_advance_button"):
            st.session_state.game_state = "phase2"
            st.rerun()

    report_bug_section()

# --- Fase 2 ---
def phase2_scaled_dot_product_attention():
    st.header("Fase 2: Atenção Escalonada por Produto Escalar (Scaled Dot-Product Attention) 🎯")
    st.write("O mecanismo de atenção calcula a relevância entre elementos de uma sequência com base em vetores de consulta (query), chave (key) e valor (value). A versão escalonada melhora a estabilidade do treinamento ao dividir o produto escalar QK pela raiz quadrada da dimensão das chaves (\u221ad_k), evitando que a função softmax sature com valores muito grandes.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Aumentar magnitude dos produtos", key="p2_wrong"):
            st.error("❌ Incorreto! Isso pioraria o problema da saturação no softmax.")
            st.session_state.p2_attempts += 1
    with col2:
        if st.button("Evitar gradientes pequenos no softmax ✅", key="p2_correct"):
            st.session_state.phase2_passed = True
            st.rerun()

    if st.session_state.phase2_passed:
        st.success("✅ Correto! Escalar por \u221ad_k estabiliza os gradientes da atenção.")
        st.write("Isso garante que os pesos da atenção distribuídos pelo softmax fiquem em uma faixa útil para o aprendizado.")
        if st.button("Avançar para Fase 3 ➡️", key="p2_advance_button"):
            st.session_state.game_state = "phase3"
            st.rerun()

    report_bug_section()

# --- Fase 3 ---
def phase3_multi_head_attention():
    st.header("Fase 3: Atenção Multi-Cabeça (Multi-Head Attention) 💡")
    st.write("A atenção multi-cabeça divide as representações em subespaços menores e aplica atenção separada a cada um deles. Isso permite que o modelo aprenda diferentes tipos de relacionamentos simultaneamente (por exemplo: proximidade sintática, associação semântica etc.), melhorando a expressividade do modelo.")

    d_val = st.slider("Escolha d_k/d_v por cabeça (esperado: 64)", 32, 128, 64, step=32)
    if d_val == 64:
        st.session_state.phase3_passed = True
        st.rerun()
    elif 'p3_attempts' in st.session_state:
        st.session_state.p3_attempts += 1

    if st.session_state.phase3_passed:
        st.success("✅ Correto! Com d_model=512 e 8 cabeças, temos d_k = d_v = 64 por cabeça.")
        st.write("Isso mantém o custo computacional comparável ao de uma única cabeça com d_model completo.")
        if st.button("Avançar para Fase 4 ➡️", key="p3_advance_button"):
            st.session_state.game_state = "phase4"
            st.rerun()

    report_bug_section()

# --- Fase 4 ---
def phase4_positional_encoding():
    st.header("Fase 4: Codificação Posicional 📍")
    st.write("Como o Transformer não possui estrutura sequencial explícita como nas RNNs, ele precisa adicionar manualmente informações sobre a posição dos tokens. Isso é feito somando um vetor de posição ao vetor de embedding de cada palavra. O artigo original usa funções seno e cosseno com diferentes frequências para esse fim.")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Embeddings aprendidos", key="p4_wrong1"):
            st.warning("⚠️ Método alternativo possível, mas não o utilizado no artigo original.")
            st.session_state.p4_attempts += 1
    with col2:
        if st.button("Seno e Cosseno ✅", key="p4_correct"):
            st.session_state.phase4_passed = True
            st.rerun()
    with col3:
        if st.button("Hash de posição", key="p4_wrong2"):
            st.error("❌ Incorreto. Hashes não preservam relações posicionais.")
            st.session_state.p4_attempts += 1

    if st.session_state.phase4_passed:
        st.success("✅ Correto! As funções trigonométricas garantem generalização para sequências mais longas.")
        if st.button("Avançar para Fase 5 ➡️", key="p4_advance_button"):
            st.session_state.game_state = "phase5"
            st.rerun()

    report_bug_section()

# --- Fase 5 ---
def phase5_training_results():
    st.header("Fase 5: Treinamento e Resultados ⚡")
    st.write("Vamos simular o treinamento do Transformer e observar como ele se compara a modelos anteriores em termos de qualidade de tradução e custo computacional.")
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    st.success("✅ Treinamento concluído!")
    if st.button("Ver Resumo 🏆"):
        st.session_state.game_state = "summary"
        st.rerun()
    report_bug_section()

# --- Resumo Final ---
def game_summary():
    st.header("Resumo: Attention Is All You Need 🎉")
    st.markdown("""
* Arquitetura baseada exclusivamente em atenção (sem RNNs ou CNNs)  
* Treinamento altamente paralelizável  
* Mecanismo de auto-atenção (self-attention)  
* Produto escalar escalonado (scaled dot-product) para estabilidade  
* Atenção multi-cabeça (multi-head) para múltiplas perspectivas  
* Codificação posicional baseada em seno e cosseno  
* Resultados superiores em tradução automática (BLEU)  
* Capacidade de generalização para outras tarefas de linguagem natural
""")
    if st.button("Jogar Novamente 🔁"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    report_bug_section()

# --- Menu Inicial ---
def main_menu():
    st.title("🚀 A Jornada do Transformer: Atenção Desvendada! 🚀")
    st.write("Bem-vindo, engenheiro de inteligência artificial! Sua missão é guiar um modelo Transformer por cinco fases de construção e entendimento. Cada fase abordará um conceito essencial do artigo 'Attention Is All You Need'.")
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
