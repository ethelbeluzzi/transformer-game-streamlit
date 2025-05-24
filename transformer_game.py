import streamlit as st
import time
import datetime
import os

# --- Inicialização de Estado ---
def init_state():
    st.session_state.setdefault("game_state", "menu")
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
    st.sidebar.subheader("\U0001F41E Reportar Erro / Sugestão")
    with st.sidebar.form("bug_report_form", clear_on_submit=True):
        bug_text = st.text_area("Descreva o erro que encontrou ou sua sugestão de melhoria:")
        submitted = st.form_submit_button("Enviar Feedback ✉️", key="feedback_submit_button")
        if submitted:
            if bug_text.strip():
                log_feedback(bug_text)
            else:
                st.sidebar.warning("Por favor, escreva algo antes de enviar.")

# --- Fases do Jogo ---
def main_menu():
    st.title("🚀 A Jornada do Transformer: Atenção Desvendada! 🚀")
    st.markdown("Bem-vindo, **engenheiro de IA**! Sua missão é construir o modelo de tradução de linguagem mais eficiente e poderoso do mundo. Guie seu **Transformer** através das fases de design, treinamento e otimização.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Transformer_architecture.svg/800px-Transformer_architecture.svg.png", use_container_width=True, caption="Arquitetura do Transformer: Onde a Atenção é Tudo!")
    st.write("Prepare-se para desvendar os segredos da atenção!")
    if st.button("Iniciar Missão ➡️", key="main_menu_start_button"):
        st.session_state.game_state = "phase1"
        st.rerun()
    report_bug_section()

def phase1_architecture():
    st.header("Fase 1: A Arquitetura Fundacional (Encoder-Decoder) 🏗️")
    st.write("Antes do Transformer, a maioria dos modelos de sequência usava redes recorrentes (RNNs) ou convolucionais (CNNs) para processar informações...")
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
            st.success("✅ Correto!...")
            if st.button("Avançar para Fase 2 ➡️", key="p1_advance_button"):
                st.session_state.game_state = "phase2"
                st.rerun()
    if st.session_state.get('p1_attempts', 0) >= 3 and st.session_state.game_state != "phase2":
        st.info("💡 Dica: Lembre-se que o Transformer 'dispensa' recorrência...")
    report_bug_section()

def phase2_scaled_dot_product_attention():
    st.header("Fase 2: O Poder da Atenção (Scaled Dot-Product Attention) 🎯")
    st.write("A função de atenção mapeia uma Query (Q)...")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Aumentar a magnitude dos produtos escalares", key="p2_btn_increase"):
            st.error("❌ Incorreto!...")
            st.session_state.p2_attempts += 1
    with col2:
        if st.button("Evitar gradientes muito pequenos no softmax ✅", key="p2_btn_softmax"):
            st.success("✅ Correto!...")
            if st.button("Avançar para Fase 3 ➡️", key="p2_advance_button"):
                st.session_state.game_state = "phase3"
                st.rerun()
    if st.session_state.get('p2_attempts', 0) >= 2 and st.session_state.game_state != "phase3":
        st.info("💡 Dica: Pense no que acontece com os valores quando $d_k$ é grande...")
    report_bug_section()

def phase3_multi_head_attention():
    st.header("Fase 3: Multi-Head Attention: Múltiplas Perspectivas 💡")
    st.write("Em vez de uma única função de atenção...")
    d_val = st.slider("Escolha o valor para d_k e d_v (esperado d_model/h)", 32, 128, 64, step=32, key="p3_slider")
    if d_val == 64:
        st.success("✅ Correto!...")
        if st.button("Avançar para Fase 4 ➡️", key="p3_advance_button"):
            st.session_state.game_state = "phase4"
            st.rerun()
    else:
        st.warning("Tente novamente...")
        st.session_state.p3_attempts += 1
        if st.session_state.p3_attempts >= 2:
            st.info("💡 Dica: Divida a dimensão total (d_model) pela quantidade de cabeças (h).")
    report_bug_section()

def phase4_positional_encoding():
    st.header("Fase 4: A Importância da Posição (Positional Encoding) 📍")
    st.write("Como o Transformer não possui recorrência ou convolução...")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Embeddings Posicionais Aprendidos", key="p4_btn_learned"):
            st.warning("⚠️ O paper experimentou isso...")
            st.session_state.p4_attempts += 1
    with col2:
        if st.button("Funções Seno e Cosseno de Diferentes Frequências ✅", key="p4_btn_sin_cos"):
            st.success("✅ Correto!...")
            if st.button("Avançar para Fase 5 ➡️", key="p4_advance_button"):
                st.session_state.game_state = "phase5"
                st.rerun()
    with col3:
        if st.button("Hash de Posição", key="p4_btn_hash"):
            st.error("❌ Incorreto...")
            st.session_state.p4_attempts += 1
    if st.session_state.p4_attempts >= 2 and st.session_state.game_state != "phase5":
        st.info("💡 Dica: O método escolhido foi para permitir extrapolação...")
    report_bug_section()

def phase5_training_results():
    st.header("Fase 5: Treinamento e Otimização ⚡")
    st.write("Vamos simular o treinamento do seu Transformer...")
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    st.success("✅ Treinamento Concluído!")
    if st.button("Ver Resumo das Descobertas 🏆", key="p5_summary_button"):
        st.session_state.game_state = "summary"
        st.rerun()
    report_bug_section()

def game_summary():
    st.header("Missão Concluída! Parabéns! 🎉")
    st.markdown("* Arquitetura baseada em atenção\n* Paralelização aumentada\n* Auto-atenção\n* Multi-Head Attention\n* Codificação Posicional\n* Resultados superiores\n* Generalização...")
    if st.button("Jogar Novamente 🔁", key="summary_replay_button"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
    report_bug_section()

# --- Navegação entre fases ---
st.write(f"\U0001F9ED Estado atual: {st.session_state.game_state}")

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
