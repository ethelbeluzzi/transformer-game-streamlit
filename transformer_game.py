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
    st.write("Antes do Transformer, a maioria dos modelos de sequência usava RNNs ou CNNs...")
    with st.expander("🤔 O que é Encoder-Decoder?"):
        st.write("Imagine que você quer traduzir uma frase...")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Redes Recorrentes Complexas 🔄", key="p1_btn_rnn"):
            st.error("❌ Resposta incorreta!...")
            st.session_state.p1_attempts += 1
    with col2:
        if st.button("Redes Convolucionais 🖼️", key="p1_btn_cnn"):
            st.warning("⚠️ Resposta aceitável, mas não a ideal!...")
            st.session_state.p1_attempts += 1
    with col3:
        if st.button("Atenção Pura (Transformer) ✨", key="p1_btn_attention"):
            st.session_state.phase1_passed = True
            st.rerun()

    if st.session_state.phase1_passed:
        st.success("✅ Correto! O Transformer utiliza atenção pura.")
        st.image("img/transformer.png", width=300)
        if st.button("Avançar para Fase 2 ➡️", key="p1_advance_button"):
            st.session_state.game_state = "phase2"
            st.rerun()

    report_bug_section()

# --- Fase 2 ---
def phase2_scaled_dot_product_attention():
    st.header("Fase 2: Scaled Dot-Product Attention 🎯")
    st.write("A atenção mapeia uma query e pares key-value para uma saída...")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Aumentar magnitude dos produtos", key="p2_wrong"):
            st.error("❌ Incorreto!")
            st.session_state.p2_attempts += 1
    with col2:
        if st.button("Evitar gradientes pequenos no softmax ✅", key="p2_correct"):
            st.session_state.phase2_passed = True
            st.rerun()

    if st.session_state.phase2_passed:
        st.success("✅ Correto! Evitar gradientes pequenos garante estabilidade.")
        if st.button("Avançar para Fase 3 ➡️", key="p2_advance_button"):
            st.session_state.game_state = "phase3"
            st.rerun()

    report_bug_section()

# --- Fase 3 ---
def phase3_multi_head_attention():
    st.header("Fase 3: Multi-Head Attention 💡")
    st.write("Multi-head permite múltiplas representações simultâneas...")

    d_val = st.slider("Escolha d_k/d_v (esperado: 64)", 32, 128, 64, step=32)
    if d_val == 64:
        st.session_state.phase3_passed = True
        st.rerun()
    elif 'p3_attempts' in st.session_state:
        st.session_state.p3_attempts += 1

    if st.session_state.phase3_passed:
        st.success("✅ Correto! d_model/h = 512/8 = 64")
        if st.button("Avançar para Fase 4 ➡️", key="p3_advance_button"):
            st.session_state.game_state = "phase4"
            st.rerun()

    report_bug_section()

# --- Fase 4 ---
def phase4_positional_encoding():
    st.header("Fase 4: Codificação Posicional 📍")
    st.write("Sem RNNs, o Transformer precisa codificar posição...")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Embeddings aprendidos", key="p4_wrong1"):
            st.warning("⚠️ Possível, mas não o método usado.")
            st.session_state.p4_attempts += 1
    with col2:
        if st.button("Seno e Cosseno ✅", key="p4_correct"):
            st.session_state.phase4_passed = True
            st.rerun()
    with col3:
        if st.button("Hash de posição", key="p4_wrong2"):
            st.error("❌ Incorreto.")
            st.session_state.p4_attempts += 1

    if st.session_state.phase4_passed:
        st.success("✅ Correto! Seno e cosseno permitem generalização.")
        if st.button("Avançar para Fase 5 ➡️", key="p4_advance_button"):
            st.session_state.game_state = "phase5"
            st.rerun()

    report_bug_section()

# --- Fase 5 ---
def phase5_training_results():
    st.header("Fase 5: Treinamento e Resultados ⚡")
    st.write("Vamos simular o treinamento e avaliar desempenho...")
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
* Arquitetura baseada em atenção  
* Paralelização aumentada  
* Self-Attention  
* Scaled Dot-Product Attention  
* Multi-Head Attention  
* Codificação Posicional  
* Resultados superiores (BLEU)  
* Generalização para outras tarefas
""")
    if st.button("Jogar Novamente 🔁"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    report_bug_section()

# --- Menu Inicial ---
def main_menu():
    st.title("🚀 A Jornada do Transformer: Atenção Desvendada! 🚀")
    st.write("Bem-vindo, engenheiro de IA! Sua missão é guiar um modelo Transformer...")
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
