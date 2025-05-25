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
from github import Github
import datetime
import streamlit as st

def log_feedback(feedback_text):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_text = f"[{timestamp}] {feedback_text}\n"

    token = st.secrets["GITHUB_TOKEN"]
    repo_name = st.secrets["REPO_NAME"]
    file_path = st.secrets["FILE_PATH"]

    g = Github(token)
    repo = g.get_repo(repo_name)

    try:
        contents = repo.get_contents(file_path)
        new_content = contents.decoded_content.decode() + full_text
        repo.update_file(file_path, "append feedback", new_content, contents.sha)
    except Exception:
        repo.create_file(file_path, "create feedback log", full_text)

    st.success("✅ Feedback salvo com sucesso no repositório privado!")

# --- Função lateral de bug/sugestão ---
def report_bug_section():
    st.sidebar.subheader("🐞 Reportar Erro Conceitual do Jogo")
    with st.sidebar.form("bug_report_form", clear_on_submit=True):
        bug_text = st.text_area("Descreva o erro que encontrou ou sua sugestão de melhoria:")
        submitted = st.form_submit_button("Enviar Feedback ✉️")
        if submitted:
            if bug_text.strip():
                log_feedback(bug_text)
            else:
                st.sidebar.warning("Por favor, escreva algo antes de enviar.")

# --- Função lateral de llms ---

from huggingface_hub import InferenceClient

import requests

import requests

def llm_sidebar_consultation():
    st.sidebar.subheader("🤖 Tem alguma dúvida? Pergunte aqui para a LLM! (Qwen2.5-7B-Instruct, via Hugging Face)")
    user_question = st.sidebar.text_area("Digite sua dúvida abaixo:", key="hf_chat_user_question")

    if st.sidebar.button("Enviar pergunta", key="hf_chat_submit") and user_question.strip():
        with st.spinner("Consultando a LLM..."):
            try:
                hf_token = st.secrets["HF_TOKEN"]
                headers = {
                    "Authorization": f"Bearer {hf_token}",
                    "Content-Type": "application/json"
                }

                API_URL = "https://router.huggingface.co/together/v1/chat/completions"
                payload = {
                    "model": "Qwen/Qwen2.5-7B-Instruct-Turbo",
                    "messages": [
                        {"role": "user", "content": user_question.strip()}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 300
                }

                response = requests.post(API_URL, headers=headers, json=payload, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    reply = result["choices"][0]["message"]["content"]
                    st.sidebar.success("📘 Resposta da LLM:")
                    st.sidebar.markdown(f"> {reply.strip()}")
                elif response.status_code == 429:
                    st.sidebar.error("⚠️ Ops, atingimos o limite de requests para o modelo!")
                else:
                    st.sidebar.error("❌ Ocorreu um erro inesperado ao consultar a LLM.")

            except Exception as e:
                st.sidebar.error("❌ Ocorreu um erro técnico ao tentar se conectar à LLM.")

    # 🔽 Adiciona separador entre a LLM e a caixa de feedback de erro conceitual
    st.sidebar.markdown("---")


# --- Fase 1: Mini-game de Montagem do Transformer ---
def phase1_architecture():
    st.header("Fase 1: A Arquitetura Fundacional (Encoder-Decoder) 🏗️")

    st.markdown("""
> 📘 **Conceito-chave do artigo**  
> "Nosso modelo segue a arquitetura geral do transformador como uma pilha de camadas de codificador e decodificador."  
> — *Vaswani et al., 2017*

A arquitetura Encoder-Decoder permite que o modelo processe a entrada por completo antes de gerar a saída, otimizando tarefas como tradução, resumo e question answering.
    """)

    st.write("Arraste os blocos abaixo para a ordem correta da arquitetura Transformer: da entrada até a saída.")

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

    st.markdown("""
> 🔬 **Além do artigo**  
> Modelos como **T5**, **BART** e muitos sistemas modernos de tradução neural usam variantes dessa arquitetura.  
> A separação clara entre codificação e decodificação facilita o **aprendizado transferido (transfer learning)**, a modularização e a adaptação para tarefas distintas — como sumarização, diálogo e até geração de código.
    """)

    llm_sidebar_consultation()
    report_bug_section()


# --- Fase 2 ---
def phase2_scaled_dot_product_attention():
    st.header("Fase 2: Corrida de Vetores e Escalonamento 🎯")

    st.markdown("""
> 📘 **Conceito-chave do artigo**  
> "Utilizamos atenção por produto escalar escalonado, que é rápida e eficiente em termos de espaço computacional."  
> — *Vaswani et al., 2017*

A divisão por √dₖ evita que os valores da softmax se tornem extremos, preservando gradientes úteis para aprendizado. Essa operação é fundamental para a estabilidade da rede durante o treinamento.
    """)

    with st.expander("🤔 O que são Q, K e dₖ?"):
        st.markdown("""
- **Q (Query - Consulta):** Representa o vetor da palavra que está buscando contexto.  
- **K (Key - Chave):** Representa as palavras candidatas a fornecer esse contexto.  
- **dₖ (dimensão da chave):** Tamanho dos vetores Q e K.  
- Se dₖ for grande, os produtos Q·K podem saturar a softmax. Por isso escalonamos.
        """)
        st.markdown("A fórmula da atenção é:")
        st.latex(r"Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V")

    q_val = st.slider("Valor do vetor Q (intensidade da consulta)", 1, 100, 60, step=1)
    k_val = st.slider("Valor do vetor K (intensidade da chave)", 1, 100, 80, step=1)
    d_k = st.slider("Dimensão dₖ (tamanho do vetor)", 1, 128, 64, step=1)

    produto = q_val * k_val
    com_escalonamento = produto / (d_k ** 0.5)

    st.markdown(f"**Produto Escalar (Q·K):** `{produto}`")
    st.markdown(f"**Com Escalonamento (÷ √dₖ):** `{com_escalonamento:.2f}`")

    if 10 <= com_escalonamento <= 30:
        st.success("✅ Excelente! O valor escalonado está em uma faixa ideal para o funcionamento do softmax.")
        st.info("📘 Dica: valores entre **10 e 30** mantêm a softmax balanceada e os gradientes úteis.")
        if st.button("Avançar para Fase 3 ➡️", key="p2_advance_button"):
            st.session_state.game_state = "phase3"
            st.rerun()
    else:
        st.warning("⚠️ O valor escalonado ainda está fora do ideal. Tente ajustar Q, K ou dₖ para obter resultado entre **10 e 30**.")

    st.markdown("""
> 🔬 **Além do artigo**  
> A dimensão dos vetores **Q e K** afeta a expressividade da atenção:  
> - Vetores **pequenos** (ex: 16, 32) não capturam nuances complexas.  
> - Vetores **grandes demais** (ex: 128, 256) causam produtos exagerados → saturação da softmax → aprendizado prejudicado.  
>  
> A escalagem por √dₖ **compensa esse efeito**, mantendo os gradientes estáveis.  
>  
> Na prática, isso é essencial em **modelos como GPT ou T5**, que processam sequências longas e dependem de uma atenção estável para manter coerência sem degradar o aprendizado em passos distantes.
    """)

    llm_sidebar_consultation()
    report_bug_section()

# --- Fase 3 ---
def phase3_multi_head_attention():
    st.header("Fase 3: Multi-Head Attention: Cabeças Paralelas 🧠")

    st.markdown("""
> 📘 **Conceito-chave do artigo**  
> "Ao invés de uma única atenção com vetores de dimensão dₘₒdₑₗ, projetamos Q, K, V múltiplas vezes (h cabeças) para subespaços menores, permitindo que o modelo atenda simultaneamente a diferentes informações de diferentes posições."  
> — *Vaswani et al., 2017*

A Multi-Head Attention permite que o Transformer olhe para a mesma informação de diversas maneiras simultaneamente, aprendendo padrões variados entre tokens.
    """)

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

    st.markdown("🔎 **Cabeça 1** (posição local): tende a olhar para palavras vizinhas.")
    st.markdown("🔎 **Cabeça 2** (ligação estrutural): conecta palavras com dependência gramatical.")
    st.markdown("🔎 **Cabeça 3** (semântica implícita): foca em termos semanticamente relacionados.")

    st.markdown("---")
    st.markdown(f"🧠 Com foco em **{foco}**, veja como cada cabeça pode responder:")

    st.write(f"**Cabeça 1:** Atenção distribuída para: {', '.join(padroes_cabeca1.get(foco, []))}")
    st.write(f"**Cabeça 2:** Atenção distribuída para: {', '.join(padroes_cabeca2.get(foco, []))}")
    st.write(f"**Cabeça 3:** Atenção distribuída para: {', '.join(padroes_cabeca3.get(foco, []))}")

    st.success("✅ Observe como diferentes cabeças focam em padrões distintos — essa diversidade é essencial para que o modelo compreenda múltiplas relações contextuais ao mesmo tempo.")

    if st.button("Avançar para Fase 4 ➡️", key="p3_advance_button"):
        st.session_state.game_state = "phase4"
        st.rerun()

    st.markdown("""
> 🔬 **Além do artigo**  
> Em modelos maiores como **GPT-3 ou PaLM**, o número de cabeças cresce (ex: 96 ou mais).  
> Cada uma aprende de forma independente:  
> - Algumas especializam-se em pontuação, outras em coesão, ou em longas dependências sintáticas.  
> - A diversidade entre cabeças é essencial para tarefas como sumarização, programação, tradução ou raciocínio matemático.  
>  
> Mesmo cabeças com desempenho fraco isoladamente podem ser úteis dentro do conjunto.
    """)

    llm_sidebar_consultation()
    report_bug_section()

# --- Fase 4 ---
import numpy as np
import matplotlib.pyplot as plt

def phase4_positional_encoding():
    st.header("Fase 4: Codificação Posicional (Positional Encoding) 🌐")

    st.markdown("""
> 📘 **Conceito-chave do artigo**  
> “Como o modelo não possui mecanismos recorrentes ou convolucionais, é necessário incorporar alguma informação sobre a ordem das palavras na sequência. Para isso, usamos funções senoidais que variam com a posição.”  
> — *Vaswani et al., 2017*

Transformers não têm noção da ordem dos tokens por padrão. Para isso, adicionam aos embeddings vetores de **codificação posicional** — combinações de seno e cosseno — que representam a posição de cada palavra na sequência.

Essas funções produzem padrões contínuos e diferenciáveis, permitindo que o modelo:
- Reconheça a **posição absoluta** dos tokens
- Codifique **relações de distância** entre palavras
- **Extrapole** para comprimentos de sequência maiores que os vistos no treino
""")

    st.subheader("🔢 Visualização: Senoides para representar posições")

    posicoes = np.arange(0, 50)
    dim = 16  # Exemplo com 16 dimensões
    pos_enc = np.array([
        [np.sin(p / (10000 ** (2 * i / dim))) if i % 2 == 0 else np.cos(p / (10000 ** (2 * (i - 1) / dim))) for i in range(dim)]
        for p in posicoes
    ])

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(posicoes, pos_enc[:, 0], label="Dimensão 0 (seno)")
    ax.plot(posicoes, pos_enc[:, 1], label="Dimensão 1 (cosseno)")
    ax.set_title("Variação senoidal em duas dimensões do Positional Encoding")
    ax.set_xlabel("Posição do token")
    ax.legend()
    st.pyplot(fig)

    st.markdown("Acima, vemos como diferentes dimensões oscilam de forma distinta conforme a posição muda. Isso cria um **padrão único** por posição, que pode ser aprendido pelo modelo.")

    st.subheader("🧠 Pergunta")
    resposta = st.radio("O que o Positional Encoding permite ao Transformer?", [
        "Capturar a importância semântica das palavras",
        "Aprender a ordem e a distância entre os tokens",
        "Entender a frequência de cada palavra",
        "Ignorar a posição, já que a atenção cuida disso"
    ], index=0, key="fase4_radio")

    if resposta:
        if resposta == "Aprender a ordem e a distância entre os tokens":
            st.success("✅ Correto! O Positional Encoding insere padrões que permitem ao modelo saber quem vem antes ou depois, e quão longe cada palavra está da outra.")
            if st.button("Avançar para Fase 5 ➡️", key="p4_advance_button"):
                st.session_state.game_state = "phase5"
                st.rerun()
        else:
            st.error("❌ Ainda não! Lembre-se: o objetivo do Positional Encoding é oferecer ao modelo uma forma de representar a **ordem e distância** entre tokens — algo que, sozinho, a atenção não captura.")

    st.markdown("""
> 🔬 **Além do artigo**  
> Muitos modelos modernos (como BERT e GPT) usam variantes de codificação posicional:  
> - **Fixas** (como seno/cosseno) → extrapolam para posições além do treino  
> - **Aprendidas** → mais flexíveis, mas menos interpretáveis  
>  
> A codificação posicional continua sendo uma das maiores inovações dos Transformers — e uma das razões para sua escalabilidade.
    """)

    llm_sidebar_consultation()
    report_bug_section()


# --- Fase 5 ---
def phase5_training_results():
    st.header("Fase 5: Treinamento e Otimização (Resultados e Eficiência) ⚡")

    st.markdown("""
> 📘 **Conceito-chave do artigo**  
> "O modelo Transformer atinge resultados de ponta em tradução automática, com menor custo computacional de treinamento comparado a modelos anteriores."  
> — *Vaswani et al., 2017*

A arquitetura baseada em atenção pura permite paralelismo eficiente e melhora a escalabilidade, reduzindo o tempo e custo de treinamento mesmo com grande volume de dados.
    """)

    st.subheader("Simulando Treinamento... ⏳")
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    st.success("✅ Treinamento Concluído! Seu Transformer está pronto!")

    st.write("Aqui estão os resultados comparativos do Transformer em tarefas de tradução (WMT 2014 EN-DE):")

    st.markdown("""
**Legenda:**
- 🟢 BLEU Score: quanto mais alto, melhor
- 🔵 FLOPs (Floating Point Operations): quanto menor, mais eficiente
    """)

    data = {
        "Modelo": ["ByteNet", "GNMT + RL", "ConvS2S", "Transformer (base)", "Transformer (big)"],
        "BLEU (EN-DE)": ["23.75", "24.6", "25.16", "**27.3** 🟢", "**28.4** 🟢"],
        "Custo de Treinamento (FLOPs)": ["$2.3\\cdot10^{19}$", "$1.4\\cdot10^{21}$", "$9.6\\cdot10^{18}$", "**$3.3\\cdot10^{18}$** 🔵", "**$2.3\\cdot10^{19}$**"]
    }
    st.table(data)

    # 🔝 Mini ranking
    if st.button("🔝 Destacar o melhor modelo"):
        st.info("🏆 **Transformer (big)** se destaca com **BLEU 28.4** e excelente desempenho em tradução automática!")

    st.markdown("""
> 🔬 **Além do artigo**  
> O BLEU Score é uma métrica baseada em n-gramas que compara a saída gerada com traduções humanas.  
> - Um aumento de **2 BLEU** pode representar uma diferença **perceptível na fluência e precisão**.  
> - O Transformer não só superou modelos anteriores, mas o fez com muito **menos custo de FLOPs**.  

Isso abriu caminho para aplicações em tempo real, como tradução simultânea, assistentes virtuais multilíngues e até geração de código (com adaptações).
    """)

    st.success("🚀 Sua missão foi cumprida com sucesso: você treinou um Transformer de ponta!")

    if st.button("Ver Resumo Final 🏆", key="p5_summary_button"):
        st.session_state.game_state = "summary"
        st.rerun()

    llm_sidebar_consultation()
    report_bug_section()


# --- Resumo Final + LLM ---
def game_summary():
    st.header("Missão Concluída! Recapitulação do artigo 'Attention Is All You Need' 🎉")
    st.subheader("🧠 Você demonstrou uma compreensão sólida dos fundamentos do Transformer!")

    st.markdown("""
> 📘 **Conceito central do artigo**  
> "A arquitetura Transformer depende exclusivamente de mecanismos de atenção, eliminando o uso de recorrência e convolução, permitindo paralelização eficiente."  
> — *Vaswani et al., 2017*
    """)

    st.markdown("### 🧩 Elementos centrais explorados no jogo")

    st.markdown("""
#### 1. **Arquitetura Encoder-Decoder baseada em atenção**
- O modelo é organizado em **camadas empilhadas** de codificadores e decodificadores.
- O **Encoder** transforma a entrada em uma representação contextual.
- O **Decoder** gera a saída com base nessa representação e nas posições anteriores.
- Isso permite lidar com **tarefas de tradução**, sumarização e outras sequenciais com alta flexibilidade.
""")

    st.markdown("""
#### 2. **Mecanismo de Atenção por Produto Escalar Escalonado**
- A atenção compara a *query* com todas as *keys* e pondera os *values*.
- O produto Q·K é **escalonado por √dₖ**, evitando saturação da função softmax.
- Isso mantém os **gradientes úteis** e o **treinamento estável**, mesmo em modelos grandes.
""")

    st.markdown("""
#### 3. **Atenção Multi-Cabeça (Multi-Head Attention)**
- Em vez de uma única atenção, o modelo usa múltiplas cabeças independentes.
- Cada cabeça aprende um padrão diferente: **estrutura, semântica, posição, dependências**.
- No final, os resultados são **concatenados** e projetados novamente, enriquecendo a representação.
""")

    st.markdown("""
#### 4. **Positional Encoding**
- Como o Transformer **não possui recorrência**, ele precisa saber a posição das palavras.
- Usando **funções seno e cosseno**, cada posição recebe uma curva única, contínua e extrapolável.
- Isso permite ao modelo lidar com **ordem das palavras** mesmo em contextos longos ou fora da distribuição.
""")

    st.markdown("""
#### 5. **Eficiência de Treinamento e Resultados**
- O Transformer atinge **BLEU scores superiores** a modelos anteriores com **menos FLOPs**.
- A ausência de recorrência permite **paralelização total** no treinamento.
- Sua eficiência abriu caminho para modelos massivos como BERT, GPT, T5, e muitos outros.
""")

    st.markdown("### 🌍 Impactos no mundo real")
    st.markdown("""
- Permitiu o surgimento de modelos de linguagem de código aberto e escaláveis.
- Influenciou modelos em **áudio, visão computacional, bioinformática e robótica**.
- Tornou possível o treinamento em **paralelo em GPUs e TPUs**, reduzindo drasticamente o tempo de inferência.

> 🔬 O Transformer mudou profundamente o paradigma de modelagem de linguagem — e sua missão hoje mostra que você compreende as engrenagens por trás dessa revolução.
    """)

    if st.button("Jogar novamente 🔁", key="summary_replay_button"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    llm_sidebar_consultation()
    report_bug_section()

# --- Menu Inicial ---
def main_menu():
    st.title("🚀 A Jornada do Transformer: Atenção Desvendada! 🚀")

    st.write("""
Esse é um jogo interativo pensado para ajudar você a revisar, de forma leve e engajada, os principais conceitos do paper clássico *Attention is All You Need*. [Leia o paper original](https://arxiv.org/abs/1706.03762).

Durante o jogo, você será guiado por cinco fases, cada uma com um mini-desafio sobre aspectos fundamentais do Transformer: arquitetura, atenção escalonada, atenção multi-cabeça, codificação posicional e resultados de desempenho.

No canto lateral esquerdo, você pode:
- ❓ Consultar uma **LLM integrada** sempre que tiver dúvidas sobre os conceitos apresentados.
- 🐞 Usar a **caixinha de feedback** para reportar erros conceituais ou sugerir melhorias a qualquer momento.
    """)

    # 🔽 Centralizar imagem com layout de colunas
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("img/transformer.png", use_container_width=True)

    if st.button("Iniciar Missão ➡️"):
        st.session_state.game_state = "phase1"
        st.rerun()
    report_bug_section()

# --- Navegação ---
fases_nomes = {
    "menu": "Menu Inicial",
    "phase1": "Fase 1",
    "phase2": "Fase 2",
    "phase3": "Fase 3",
    "phase4": "Fase 4",
    "phase5": "Fase 5",
    "summary": "Resumo Final"
}
estado_legivel = fases_nomes.get(st.session_state.game_state, "Desconhecido")
st.write(f"🧭 Estado atual: {estado_legivel}")

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
