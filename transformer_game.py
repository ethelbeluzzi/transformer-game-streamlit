import streamlit as st
import time
import datetime
import os

# --- Funções Auxiliares para Logs ---
LOG_FILE = "game_feedback.log"

def log_feedback(feedback_text):
    """
    Registra o feedback do usuário em um arquivo de log.
    No ambiente do Streamlit Cloud, este arquivo é temporário para a sessão.
    Para persistência real, seria necessário um serviço de armazenamento externo.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {feedback_text}\n")
    st.success("Obrigado pelo seu feedback! Ele foi registrado.")

# --- Conteúdo do Jogo ---

def main_menu():
    # Título modificado para confirmar a atualização da versão
    st.title("🚀 A Jornada do Transformer: Atenção Desvendada! (V3) 🚀")
    st.markdown("Bem-vindo, **engenheiro de IA**! Sua missão é construir o modelo de tradução de linguagem mais eficiente e poderoso do mundo. Guie seu **Transformer** através das fases de design, treinamento e otimização.")
    # Mantendo apenas a imagem principal da arquitetura
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Transformer_architecture.svg/800px-Transformer_architecture.svg.png", use_container_width=True, caption="Arquitetura do Transformer: Onde a Atenção é Tudo!")
    st.write("Prepare-se para desvendar os segredos da atenção!")
    # Mudando a key do botão inicial para evitar qualquer confusão teórica
    if st.button("Iniciar Missão ➡️", key="main_menu_start_button"):
        print("Botão 'Iniciar Missão' clicado!") # Log para depuração
        st.session_state.game_state = "phase1"
        st.rerun()

    report_bug_section()


def phase1_architecture():
    st.header("Fase 1: A Arquitetura Fundacional (Encoder-Decoder) 🏗️")
    st.write("Antes do Transformer, a maioria dos modelos de sequência usava redes recorrentes (RNNs) ou convolucionais (CNNs) para processar informações. No entanto, elas tinham limitações, especialmente na **paralelização do treinamento** para sequências longas.")
    st.write("Qual tipo de arquitetura você escolherá para o coração do seu novo modelo?")

    with st.expander("🤔 Explicar de forma simples: O que é um Encoder-Decoder?"):
        st.write("Imagine que você quer traduzir uma frase. O **Encoder** é como um *leitor* que lê a frase original inteira e a 'entende', transformando-a em um código ou representação. O **Decoder** é como um *escritor* que, com base no que o leitor entendeu (o código), escreve a frase traduzida, palavra por palavra. Eles trabalham juntos para transformar uma sequência de entrada (frase original) em uma sequência de saída (frase traduzida).")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Redes Recorrentes Complexas 🔄", key="p1_btn_rnn"):
            st.error("❌ Resposta incorreta! Redes recorrentes são inerentemente sequenciais, o que impede a paralelização eficiente dentro dos exemplos de treinamento, tornando-as lentas para sequências longas.")
            if 'p1_attempts' not in st.session_state: st.session_state.p1_attempts = 0
            st.session_state.p1_attempts += 1
    with col2:
        if st.button("Redes Convolucionais 🖼️", key="p1_btn_cnn"):
            st.warning("⚠️ Resposta aceitável, mas não a ideal! Modelos convolucionais podem processar em paralelo, mas a capacidade de relacionar posições distantes cresce linearmente ou logaritmicamente com a distância, dificultando o aprendizado de dependências de longo alcance de forma ideal.")
            if 'p1_attempts' not in st.session_state: st.session_state.p1_attempts = 0
            st.session_state.p1_attempts += 1
    with col3:
        if st.button("Atenção Pura (Transformer) ✨", key="p1_btn_attention"):
            st.success("✅ Correto! O Transformer se baseia unicamente em mecanismos de atenção, abandonando a recorrência e as convoluções inteiramente. Isso permite muito mais paralelização e um treinamento significativamente mais rápido. ****")
            st.write("O Transformer segue uma arquitetura Encoder-Decoder, usando pilhas de auto-atenção. ****")
            # Mantendo apenas a imagem principal da arquitetura aqui também
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Transformer_architecture.svg/800px-Transformer_architecture.svg.png", use_container_width=True, caption="Diagrama Simplificado do Encoder-Decoder")
            st.write("Próxima etapa: mergulhar no coração da atenção!")
            # Este botão deve aparecer APENAS se a resposta correta foi escolhida
            if st.button("Avançar para Fase 2 ➡️", key="p1_advance_button"): # Adicionado key
                print("Botão 'Avançar para Fase 2' clicado!") # Log para depuração
                st.session_state.game_state = "phase2"
                st.rerun()

    if st.session_state.get('p1_attempts', 0) >= 3 and st.session_state.game_state != "phase2":
        st.info("💡 Dica: Lembre-se que o Transformer 'dispensa' recorrência e convoluções para focar em paralelismo.")
    report_bug_section()


def phase2_scaled_dot_product_attention():
    st.header("Fase 2: O Poder da Atenção (Scaled Dot-Product Attention) 🎯")
    st.write("A função de atenção mapeia uma Query (Q) e um conjunto de pares Key-Value (K, V) para uma saída, que é uma soma ponderada dos Values. ****")
    st.write("Nosso modelo usa a 'Scaled Dot-Product Attention'. Ela calcula os produtos escalares da Query com todas as Keys, divide cada um por $\\sqrt{d_k}$, e aplica uma função softmax para obter os pesos.")

    st.latex(r"Attention(Q,K,V)=softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V \quad (1)")

    with st.expander("🤔 Explicar de forma simples: O que são Q, K, V em Atenção?"):
        st.write("  **Q (Query - Pergunta):** É como a pergunta que você está fazendo para entender uma palavra. Para cada palavra que você quer processar, ela age como uma 'pergunta' sobre a relevância de outras palavras.")
        st.write("  **K (Key - Chave):** São as 'chaves' para as respostas. Cada palavra na sequência tem uma 'chave' que a descreve ou a identifica.")
        st.write("  **V (Value - Valor):** São os 'valores' ou as informações associadas a cada 'chave'. Se uma 'chave' é relevante para a 'pergunta' (Q), seu 'valor' (V) será mais considerado na resposta final.")
        st.write("A atenção é basicamente o processo de encontrar quais 'chaves' (palavras) são mais relevantes para a 'pergunta' (palavra atual) e usar seus 'valores' para construir uma nova e melhor representação da palavra.")

    with st.expander("🤔 Explicar de forma simples: Por que a escala por $\\sqrt{d_k}$?"):
        st.write("Quando as dimensões ($d_k$) de Q e K são grandes (os vetores são longos), os produtos escalares (multiplicações de Q e K) podem se tornar números muito grandes. Isso pode 'saturar' a função `softmax`, que é usada para transformar esses produtos em pesos (probabilidades entre 0 e 1). Quando o `softmax` satura, seus gradientes (que são usados para o modelo aprender) ficam extremamente pequenos, e o modelo pode aprender muito lentamente ou até parar de aprender. Dividir por $\\sqrt{d_k}$ 'diminui' esses produtos, mantendo-os em uma faixa mais estável onde o `softmax` funciona melhor, evitando que os gradientes fiquem minúsculos. ****")

    st.write("Qual é o principal motivo para dividirmos por $\\sqrt{d_k}$?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Aumentar a magnitude dos produtos escalares", key="p2_btn_increase"):
            st.error("❌ Incorreto! Na verdade, é o oposto. Para grandes valores de $d_k$, os produtos escalares tendem a crescer muito em magnitude naturalmente.")
            if 'p2_attempts' not in st.session_state: st.session_state.p1_attempts = 0
            st.session_state.p1_attempts += 1
    with col2:
        if st.button("Evitar gradientes muito pequenos no softmax ✅", key="p2_btn_softmax"):
            st.success("✅ Correto! Para grandes valores de $d_k$, os produtos escalares crescem muito em magnitude, empurrando a função softmax para regiões com gradientes extremamente pequenos. A escala por $\\frac{1}{\\sqrt{d_k}}$ contrai esse efeito, garantindo um treinamento mais estável. ****")
            st.write("Excelente! A 'Scaled Dot-Product Attention' é fundamental para a estabilidade e o desempenho do Transformer.")
            if st.button("Avançar para Fase 3 ➡️", key="p2_advance_button"): # Adicionado key
                print("Botão 'Avançar para Fase 3' clicado!") # Log para depuração
                st.session_state.game_state = "phase3"
                st.rerun()

    if st.session_state.get('p2_attempts', 0) >= 2 and st.session_state.game_state != "phase3":
        st.info("💡 Dica: Pense no que acontece com os valores quando $d_k$ é grande e qual função é usada para obter os pesos (softmax).")
    report_bug_section()


def phase3_multi_head_attention():
    st.header("Fase 3: Multi-Head Attention: Múltiplas Perspectivas 💡")
    st.write("Em vez de uma única função de atenção, o Transformer usa 'Multi-Head Attention'. Isso permite que o modelo 'atenda conjuntamente a informações de diferentes subespaços de representação em diferentes posições'. ****")
    st.write("Projetamos Q, K e V `h` vezes linearmente para dimensões menores ($d_k$, $d_v$), executamos a atenção em paralelo, e concatenamos os resultados.")
    st.latex(r"""
    MultiHead(Q,K,V)=Concat(head_{1},...,head_{h})W^{O} \\
    \text{where } head_{i}=\text{Attention}(QW_{i}^{Q},KW_{i}^{K},VW_{i}^{V}) \quad
   """)

    with st.expander("🤔 Explicar de forma simples: Qual a vantagem de ter múltiplas cabeças de atenção?"):
        st.write("Imagine que você está tentando entender uma frase complexa. Uma única 'cabeça de atenção' pode focar em uma coisa específica (ex: qual verbo está ligado a qual sujeito). Mas se você tiver várias 'cabeças', cada uma pode focar em algo diferente ao mesmo tempo (ex: uma no sujeito-verbo, outra no objeto-verbo, outra em sinônimos ou relações mais abstratas).")
        st.write("A 'Multi-Head Attention' permite que o Transformer olhe para a mesma informação de várias maneiras diferentes ao mesmo tempo, capturando relações mais ricas, diversas e complexas entre as palavras. E a melhor parte é que, mesmo com várias cabeças, o custo computacional é mantido eficiente. ****")

    st.write("Se $d_{model}=512$ e usamos $h=8$ cabeças, quais seriam as dimensões de $d_k$ e $d_v$ para cada cabeça, para que o custo computacional total seja similar ao de uma única cabeça com dimensionalidade total?")

    d_val = st.slider("Escolha o valor para d_k e d_v (esperado d_model/h)", 32, 128, 64, step=32, key="p3_slider")

    if d_val == 64:
        st.success(f"✅ Correto! Com $d_{{model}}=512$ e $h=8$, então $d_k = d_v = d_{{model}}/h = 512/8 = 64$. ****")
        st.write("Isso permite que o modelo aprenda diferentes representações e foque em diferentes aspectos da sequência, sem aumentar significativamente o custo computacional.")
        # Mantendo apenas a imagem principal da arquitetura aqui também
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Transformer_architecture.svg/800px-Transformer_architecture.svg.png", use_container_width=True, caption="Multi-Head Attention")
        if st.button("Avançar para Fase 4 ➡️", key="p3_advance_button"): # Adicionado key
            print("Botão 'Avançar para Fase 4' clicado!") # Log para depuração
            st.session_state.game_state = "phase4"
            st.rerun()
    else:
        st.warning(f"Tente novamente. Lembre-se que o custo computacional é mantido similar ao de uma única cabeça de atenção com dimensionalidade total.")
        if 'p3_attempts' not in st.session_state: st.session_state.p3_attempts = 0
        st.session_state.p3_attempts += 1
        if st.session_state.get('p3_attempts', 0) >= 2:
            st.info("💡 Dica: Divida a dimensão total (`d_model`) pela quantidade de cabeças (`h`).")
    report_bug_section()


def phase4_positional_encoding():
    st.header("Fase 4: A Importância da Posição (Positional Encoding) 📍")
    st.write("Como o Transformer não possui recorrência ou convolução, ele precisa de uma forma de saber a ordem das palavras na sequência. ****")
    st.write("Para isso, adicionamos 'encodings posicionais' aos embeddings de entrada.")

    with st.expander("🤔 Explicar de forma simples: Por que o Positional Encoding é necessário?"):
        st.write("Modelos como RNNs processam as palavras uma de cada vez, então eles naturalmente 'sabem' a ordem em que as palavras aparecem. O Transformer, por outro lado, processa todas as palavras ao mesmo tempo e não tem uma noção 'intrínseca' de ordem.")
        st.write("Sem alguma informação de posição, ele não conseguiria diferenciar frases como 'O gato perseguiu o rato' de 'O rato perseguiu o gato', pois ambas teriam as mesmas palavras, apenas em ordens diferentes.")
        st.write("O Positional Encoding adiciona um 'carimbo de posição' único a cada palavra, que carrega informações sobre sua posição na sequência e sua distância em relação a outras palavras. Isso permite que o modelo entenda a sintaxe e a semântica da frase baseada na ordem das palavras. ****")

    st.write("Que tipo de função o paper utilizou para gerar esses encodings posicionais?")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Embeddings Posicionais Aprendidos", key="p4_btn_learned"):
            st.warning("⚠️ O paper experimentou isso, mas encontrou resultados quase idênticos ao método escolhido e optou por outro devido à capacidade de extrapolação para sequências mais longas.")
            if 'p4_attempts' not in st.session_state: st.session_state.p4_attempts = 0
            st.session_state.p4_attempts += 1
    with col2:
        if st.button("Funções Seno e Cosseno de Diferentes Frequências ✅", key="p4_btn_sin_cos"):
            st.success("✅ Correto! O paper usou funções seno e cosseno de diferentes frequências para gerar os encodings posicionais. ****")
            st.write("Cada dimensão do encoding posicional corresponde a uma sinusoide, e isso permite que o modelo aprenda facilmente a atender por posições relativas, pois os encodings formam uma progressão geométrica.")
            # **IMAGEM REMOVIDA AQUI, CONFORME SOLICITADO**
            if st.button("Avançar para Fase 5 ➡️", key="p4_advance_button"): # Adicionado key
                print("Botão 'Avançar para Fase 5' clicado!") # Log para depuração
                st.session_state.game_state = "phase5"
                st.rerun()
    with col3:
        if st.button("Hash de Posição", key="p4_btn_hash"):
            st.error("❌ Incorreto. Essa não foi a técnica utilizada no paper original para o Positional Encoding.")
            if 'p4_attempts' not in st.session_state: st.session_state.p4_attempts = 0
            st.session_state.p4_attempts += 1

    if st.session_state.get('p4_attempts', 0) >= 2 and st.session_state.game_state != "phase5":
        st.info("💡 Dica: O método escolhido foi para permitir extrapolação para sequências mais longas e não depender de treinamento.")
    report_bug_section()


def phase5_training_results():
    st.header("Fase 5: Treinamento e Otimização (Resultados e Eficiência) ⚡")
    st.write("O Transformer não só é inovador em sua arquitetura, mas também em seu desempenho de treinamento e nos resultados finais. ****")
    st.write("Vamos simular o treinamento do seu Transformer e comparar seus resultados!")

    with st.expander("🤔 Explicar de forma simples: O que é BLEU score e por que ele é importante?"):
        st.write("O BLEU (Bilingual Evaluation Understudy) score é uma métrica muito comum usada para avaliar a qualidade de uma tradução automática. Basicamente, ele compara a tradução gerada pelo seu modelo com uma ou mais 'traduções de referência' (traduções de alta qualidade feitas por humanos).")
        st.write("Quanto mais a tradução do seu modelo se sobrepõe e se parece com as traduções de referência (considerando palavras, pares de palavras, etc.), maior será o BLEU score. Um BLEU score mais alto geralmente significa que a tradução gerada pelo modelo é de melhor qualidade e mais fluida.")

    st.subheader("Simulando Treinamento... ⏳")
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.02) # Simula o tempo de treinamento
        progress_bar.progress(i + 1)
    st.success("✅ Treinamento Concluído! Seu Transformer está pronto!")

    st.write("Aqui estão os resultados comparativos do Transformer em tarefas de tradução, baseados nos dados do paper:")

    data = {
        "Model": ["ByteNet", "GNMT + RL", "ConvS2S", "Transformer (base model)", "Transformer (big)"],
        "BLEU (EN-DE)": ["23.75", "24.6", "25.16", "**27.3**", "**28.4**"],
        "Training Cost (FLOPs)": ["$2.3\\cdot10^{19}$", "$1.4\\cdot10^{21}$", "$9.6\\cdot10^{18}$", "**$3.3\\cdot10^{18}$**", "**$2.3\\cdot10^{19}$**"]
    }
    st.table(data)

    st.write("Observe como o Transformer (big) alcançou um BLEU score de **28.4** no inglês-para-alemão, superando modelos anteriores, incluindo ensembles, por mais de 2 BLEU! ****")
    st.write("E mais importante, o treinamento foi significativamente mais rápido e mais paralelizado! O *Transformer (base model)*, apesar de menor, já demonstrou um custo de treinamento (FLOPs) muito menor que os outros modelos de ponta. ****")

    st.write("Sua missão está completa, engenheiro! Você construiu e otimizou um Transformer com sucesso!")
    if st.button("Ver Resumo das Descobertas 🏆", key="p5_summary_button"): # Adicionado key
        print("Botão 'Ver Resumo' clicado!") # Log para depuração
        st.session_state.game_state = "summary"
        st.rerun()
    report_bug_section()


def game_summary():
    st.header("Missão Concluída! Principais Descobertas do Paper 'Attention Is All You Need' 🎉")
    st.write("Parabéns, engenheiro! Você demonstrou uma compreensão profunda do **Transformer**.")
    st.subheader("Principais Temas do Artigo:")
    st.markdown("""
    * **Arquitetura Baseada Apenas em Atenção:** O Transformer abandona completamente as redes recorrentes (RNNs) e convolucionais (CNNs), baseando-se unicamente em mecanismos de atenção. Isso permite um paralelismo sem precedentes no treinamento.
    * **Paralelização Aumentada:** A natureza não sequencial do Transformer permite que o treinamento seja muito mais eficiente, com tempos de treinamento significativamente menores em comparação com modelos anteriores.
    * **Mecanismo de Auto-Atenção (Self-Attention):** É o coração do Transformer, permitindo que o modelo relacione diferentes posições da mesma sequência (ex: palavras em uma frase) para calcular uma representação mais rica de cada token.
    * **Atenção Ponderada por Ponto-Produto Escalonado (Scaled Dot-Product Attention):** Uma função de atenção específica que escala os produtos escalares por $\\sqrt{d_k}$ para evitar problemas de gradiente (gradientes muito pequenos ou muito grandes) na função softmax, crucial para a estabilidade e o aprendizado do modelo.
    * **Atenção Multi-Cabeça (Multi-Head Attention):** Em vez de uma única "cabeça" de atenção, o modelo usa várias, cada uma focando em diferentes "partes" da informação ou diferentes tipos de relações. Isso enriquece as representações aprendidas e mantém a eficiência computacional.
    * **Codificação Posicional (Positional Encoding):** Essencial para injetar informações sobre a ordem e a posição das palavras na sequência, já que o modelo não processa as palavras sequencialmente. O paper utilizou funções seno e cosseno para isso, permitindo inclusive a extrapolação para sequências de tamanhos diferentes das vistas no treinamento.
    * **Resultados de Ponta (State-of-the-Art):** O Transformer alcançou resultados superiores (BLEU score) em tarefas de tradução automática (WMT 2014 English-to-German e English-to-French), superando modelos anteriores com custos de treinamento menores devido à sua alta paralelização.
    * **Generalização para Outras Tarefas:** O modelo demonstrou a capacidade de generalizar bem para outras tarefas de processamento de linguagem natural, como análise sintática (constituency parsing).
    """)
    if st.button("Jogar Novamente 🔁", key="summary_replay_button"): # Adicionado key
        print("Botão 'Jogar Novamente' clicado!") # Log para depuração
        # Limpa o estado da sessão para reiniciar o jogo
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
    report_bug_section()

def report_bug_section():
    st.sidebar.subheader("🐞 Reportar Erro / Sugestão")
    # Adicionado 'clear_on_submit=True' para limpar o campo após o envio do formulário
    with st.sidebar.form("bug_report_form", clear_on_submit=True):
        # Removido 'key' do st.text_area e do st.form_submit_button, pois causavam TypeError
        bug_text = st.text_area("Descreva o erro que encontrou ou sua sugestão de melhoria:")
        submitted = st.form_submit_button("Enviar Feedback ✉️")
        if submitted and bug_text:
            log_feedback(bug_text)
        elif submitted and not bug_text:
            st.sidebar.warning("Por favor, escreva algo antes de enviar.")

# --- Controle do Estado do Jogo ---
# Inicializa as variáveis de estado do jogo, se ainda não existirem
if 'game_state' not in st.session_state:
    st.session_state.game_state = "menu"
# Inicializa as tentativas de cada fase para evitar KeyError (garantindo que existam sempre)
if 'p1_attempts' not in st.session_state: st.session_state.p1_attempts = 0
if 'p2_attempts' not in st.session_state: st.session_state.p2_attempts = 0
if 'p3_attempts' not in st.session_state: st.session_state.p3_attempts = 0
if 'p4_attempts' not in st.session_state: st.session_state.p4_attempts = 0


# Renderiza a fase do jogo com base no estado atual
print(f"Estado do jogo atual: {st.session_state.game_state}") # Log para depuração
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
