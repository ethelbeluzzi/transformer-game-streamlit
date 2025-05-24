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
    
    st.markdown("""
📘 **Conceito-chave do artigo**:
> "Nosso modelo segue a arquitetura geral do transformador como uma pilha de camadas de codificador e decodificador."  
(Vaswani et al., 2017)

A arquitetura Encoder-Decoder permite que o modelo processe a entrada por completo antes de gerar a saída, otimizando tarefas como tradução, resumo e question answering.

🔬 **Além do artigo**:
Modelos como T5, BART e muitos sistemas de tradução neural atuais usam variantes dessa arquitetura. Essa separação entre codificação e decodificação facilita transfer learning e modularidade.
    """)

    st.write("Arraste os blocos abaixo para a ordem correta da arquitetura Transformer: de entrada até a saída.")

    componentes = [
        "Mecanismo de Atenção",
        "Camada de Saída",
        "Decoder",
        "Encoder",
        "Embedding"
    ]

    ordem_correta = [
        "Embedding", "Encoder", "Mecanismo de Atenção", "Decoder", "Camada de Saída"
    ]

    dicas = [
        "Posição 1 - **Embedding**: transforma cada palavra em um vetor numérico compreensível pela IA.",
        "Posição 2 - **Encoder**: processa a frase de entrada e gera uma representação contextualizada.",
        "Posição 3 - **Mecanismo de Atenção**: decide quais palavras são mais importantes umas para as outras.",
        "Posição 4 - **Decoder**: gera a frase de saída, com base na atenção e no encoder.",
        "Posição 5 - **Camada de Saída**: traduz a saída do decoder para palavras compreensíveis."
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

    st.markdown("""
📘 **Conceito-chave do artigo**:
> "Utilizamos atenção por produto escalar escalonado, que é rápida e eficiente em termos de espaço computacional."  
(Vaswani et al., 2017)

A divisão por √dₖ evita que os valores da softmax se tornem extremos, preservando gradientes úteis para aprendizado.

🔬 **Além do artigo**:
A atenção escalonada é essencial para o bom funcionamento de grandes modelos como GPT, T5 e BERT. Pequenas variações nesse cálculo afetam o desempenho e a velocidade de convergência — especialmente em tarefas com sentenças longas ou contexto complexo.
    """)

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

    if 10 <= com_escalonamento <= 30:
        st.success("✅ Muito bem! O valor escalonado está em uma faixa ideal para o funcionamento do softmax.")
        st.info("📘 Dica: valores escalonados entre **10 e 30** mantêm a softmax funcionando bem: os pesos não ficam extremos e o modelo consegue aprender com eficiência.")
        if st.button("Avançar para Fase 3 ➡️", key="p2_advance_button"):
            st.session_state.game_state = "phase3"
            st.rerun()
    else:
        st.warning("⚠️ O valor escalonado ainda está fora do ideal. Tente ajustar Q, K ou aumentar dₖ para que o resultado fique entre **10 e 30**.")

    report_bug_section()

# --- Fase 3 ---
def phase3_multi_head_attention():
    st.header("Fase 3: Multi-Head Attention: Cabeças Paralelas 🧠")

    st.markdown("""
📘 **Conceito-chave do artigo**:
> "É benéfico projetar Q, K, V h vezes com projeções lineares diferentes aprendidas, permitindo que o modelo atenda conjuntamente a informações de diferentes subespaços de representação em diferentes posições."  
(Vaswani et al., 2017)

Cada cabeça de atenção aprende padrões diferentes — como ligações sintáticas, proximidade posicional ou coocorrência semântica — sem que essas categorias sejam pré-definidas.

🔬 **Além do artigo**:
Essa ideia inspirou arquiteturas como BERT e GPT, onde múltiplas cabeças permitem capturar nuances finas de contexto, ironia, ambiguidade, e relações de dependência a longa distância. Em tarefas como sumarização, tradução e resposta automática, essa diversidade de atenção melhora muito a performance.
    """)

    st.subheader("Mini-visualização: como múltiplas cabeças se comportam")
    frase = ["O", "modelo", "aprende", "relações", "entre", "tokens"]
    st.write("Escolha uma palavra para observar como diferentes cabeças podem reagir a ela:")

    foco = st.selectbox("Palavra de foco (query)", frase, key="p3_query")

    padroes_cabeca1 = {
        "O": ["modelo"],
        "modelo": ["O", "aprende"],
        "aprende": ["modelo"],
        "relações": ["entre"],
        "entre": ["relações"],
        "tokens": ["entre"]
    }

    padroes_cabeca2 = {
        "O": ["aprende"],
        "modelo": ["relações"],
        "aprende": ["tokens"],
        "relações": ["modelo"],
        "entre": ["aprende"],
        "tokens": ["O"]
    }

    padroes_cabeca3 = {
        "O": ["O"],
        "modelo": ["tokens"],
        "aprende": ["relações"],
        "relações": ["tokens"],
        "entre": ["modelo"],
        "tokens": ["relações"]
    }

    st.markdown("🔎 **Cabeça 1** (posição local): tende a olhar para palavras vizinhas da query.")
    st.markdown("🔎 **Cabeça 2** (ligação estrutural): pode conectar palavras com dependência gramatical.")
    st.markdown("🔎 **Cabeça 3** (semântica implícita): pode focar em termos semanticamente relacionados.")

    st.markdown("---")
    st.markdown(f"🧠 Com foco em **{foco}**, veja como cada cabeça pode responder:")

    st.write(f"**Cabeça 1:** Atenção distribuída para: {', '.join(padroes_cabeca1.get(foco, [])) or 'nenhuma palavra associada'}")
    st.write(f"**Cabeça 2:** Atenção distribuída para: {', '.join(padroes_cabeca2.get(foco, [])) or 'nenhuma palavra associada'}")
    st.write(f"**Cabeça 3:** Atenção distribuída para: {', '.join(padroes_cabeca3.get(foco, [])) or 'nenhuma palavra associada'}")

    st.success("✅ Observe como diferentes cabeças focam em padrões distintos — essa diversidade é fundamental para a riqueza das representações geradas pelo Transformer.")

    if st.button("Avançar para Fase 4 ➡️", key="p3_advance_button"):
        st.session_state.game_state = "phase4"
        st.rerun()

    report_bug_section()

# --- Fase 4 ---
def phase4_positional_encoding():
    st.header("Fase 4: A Importância da Posição (Positional Encoding) 📍")

    st.markdown("""
📘 **Conceito-chave do artigo**:
> "Como nosso modelo não possui nenhuma recorrência ou convolução, adicionamos informações de posição às embeddings de entrada em todas as camadas de codificador e decodificador."  
(Vaswani et al., 2017)

As codificações posicionais são baseadas em funções seno e cosseno de diferentes frequências. Isso permite ao modelo comparar posições relativas mesmo em sequências maiores do que as vistas no treinamento.

🔬 **Além do artigo**:
A codificação posicional senoidal permite que o modelo funcione mesmo em longos documentos, listas ou código-fonte. Em aplicações como detecção de eventos em séries temporais ou análise de DNA, a posição relativa entre tokens é fundamental.
    """)

    st.subheader("Visualização do caminho posicional")
    st.write("Use o controle abaixo para ajustar o tamanho da sequência (quantidade de tokens) e ver como a codificação posicional muda:")

    num_pontos = st.slider("Tamanho da sequência (tokens)", 5, 50, 20)
    uso_seno = st.radio("Tipo de codificação simulada:", ["Constante", "Linear", "Senoidal"], index=2)

    st.markdown("""
🔎 **O que muda quando você aumenta a sequência?**
- A **codificação constante** não representa posição alguma.
- A **linear** só distingue posições por ordem direta (ex: 1, 2, 3...).
- A **senoidal**, como no artigo, permite que o modelo compare posições relativas usando combinações harmônicas, sendo **mais robusta e generalizável**.
    """)

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(num_pontos)
    if uso_seno == "Constante":
        y = np.ones_like(x)
    elif uso_seno == "Linear":
        y = x
    else:
        y = np.sin(x / 5)

    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o')
    ax.set_title("Codificação Posicional - Visualização Simulada")
    st.pyplot(fig)

    if uso_seno == "Senoidal":
        st.success("✅ Correto! A codificação senoidal é usada no Transformer para representar posição de forma contínua e extrapolável.")
        if st.button("Avançar para Fase 5 ➡️", key="p4_advance_button"):
            st.session_state.game_state = "phase5"
            st.rerun()
    else:
        st.warning("⚠️ A codificação senoidal é a que melhor representa a posição, segundo o artigo. Tente selecioná-la.")

    report_bug_section()

# --- Fase 5 ---
def phase5_training_results():
    st.header("Fase 5: Treinamento e Otimização (Resultados e Eficiência) ⚙️")

    st.markdown("""
📘 **Conceito-chave do artigo**:
> "O Transformer atinge melhores resultados com menor custo computacional comparado a arquiteturas baseadas em convoluções ou recorrência."  
(Vaswani et al., 2017)

Com um design inteiramente baseado em atenção e sem dependências sequenciais, o Transformer permite **treinamento paralelizado** e **custo reduzido**, mesmo em grandes volumes de dados.

🔬 **Além do artigo**:
Essa eficiência transformou o campo da IA. Modelos como GPT, T5 e BERT usam essa arquitetura para serem treinados em escala massiva com clusters de GPUs/TPUs. Ajustar o número de cabeças, o tamanho do modelo e o batch size pode impactar significativamente o custo e a performance do sistema.
    """)

    st.subheader("Simule o Treinamento do seu Transformer")
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
    st.markdown(f"**Custo estimado (FLOPs):** `{custo_total:.2f}` unidades computacionais")

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
    st.header("Resumo: Descobertas do Artigo *Attention Is All You Need* 🎉")

    st.markdown("""
Você concluiu todas as fases e compreendeu os principais pilares da arquitetura Transformer. Aqui está um resumo aprofundado, diretamente alinhado ao artigo de Vaswani et al. (2017):

---

📘 **Conceitos centrais:**

1. **Atenção como mecanismo principal**  
   O Transformer substitui completamente RNNs e CNNs, baseando-se apenas em mecanismos de atenção. Isso permite maior paralelização e menor custo computacional.

2. **Auto-atenção (Self-Attention)**  
   Cada palavra se relaciona com todas as outras da sequência para gerar uma representação contextualizada.

3. **Produto escalar escalonado**  
   Divide o produto Q·K por √dₖ para manter os valores dentro de uma faixa útil ao softmax, evitando gradientes pequenos ou saturação.

4. **Multi-Head Attention**  
   Várias cabeças de atenção aprendem diferentes padrões simultaneamente, enriquecendo a compreensão contextual.

5. **Codificação Posicional Senoidal**  
   Funções seno e cosseno representam posição dos tokens sem depender de sequência recorrente.

6. **Eficiência no Treinamento**  
   O modelo alcançou melhores resultados com menos custo, superando modelos anteriores como GNMT, ByteNet e ConvS2S.

---

🔬 **Além do artigo: Impacto na IA atual**

- O Transformer se tornou a base para **BERT, GPT, T5, BART, DeBERTa**, entre outros.
- Modelos baseados nele lideram benchmarks em tradução, sumarização, geração de texto, classificação, QA e mais.
- Sua estrutura modular e escalável possibilitou o avanço dos **LLMs (Large Language Models)** e da **IA generativa** em escala global.

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
