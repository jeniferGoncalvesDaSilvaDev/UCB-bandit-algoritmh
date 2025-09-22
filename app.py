import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import nbinom
import pandas as pd

# Configuração da página
st.set_page_config(
    page_title="Simulação UCB1 Multi-Armed Bandit",
    page_icon="🎰",
    layout="wide"
)

# Título principal
st.title("🎰 Simulação Interativa do Algoritmo UCB1 Multi-Armed Bandit")
st.markdown("### Créditos: github.com/petroud/E-greedy_and_UCB_algorithms/")

# Sidebar para controles
st.sidebar.header("Parâmetros de Simulação")

# Controles interativos
k = st.sidebar.slider("Número de braços (k)", min_value=2, max_value=20, value=10, help="Número de máquinas caça-níqueis")
T = st.sidebar.slider("Número de tentativas (T)", min_value=100, max_value=20000, value=10000, step=100, help="Número total de rodadas")

# Botão para executar simulação
run_simulation = st.sidebar.button("Executar Simulação", type="primary")

# Seed para reprodutibilidade
seed = st.sidebar.number_input("Seed (opcional)", min_value=0, max_value=9999, value=42, help="Para resultados reproduzíveis")

def run_ucb1_simulation(k, T, seed=None):
    """Executa a simulação do algoritmo UCB1"""
    if seed is not None:
        np.random.seed(seed)
    
    # Vetores do modelo
    ucb_pulls = np.zeros(k)
    ucb_estimate_M = np.zeros(k)
    ucb_total_rewards = np.zeros(k)
    ucb_inst_score = np.zeros(T)
    ucb_best_score = np.zeros(T)
    ucb_alg_score = np.zeros(T)
    ucb_regret = np.zeros(T)
    ucb_optimal_action = np.zeros(T)
    
    # Definindo distribuições de probabilidade
    a = np.random.random(k)
    b = np.random.random(k)
    for i in range(k):
        if a[i] > b[i]:
            a[i], b[i] = b[i], a[i]
    
    mean = (a + b) / 2
    best = np.max(mean)
    index_best = np.where(mean == best)[0][0]
    
    def pull(i):
        """Puxa a alavanca i e retorna recompensa"""
        return np.random.uniform(a[i], b[i])
    
    success = 0
    
    def success_rate(optimal, i, total):
        """Calcula taxa de sucesso"""
        nonlocal success
        if i == index_best:
            success += 1
        optimal[total] = success / total if total > 0 else 0
    
    def update_stats_ucb(reward, i, t):
        """Atualiza estatísticas do UCB"""
        ucb_pulls[i] += 1
        ucb_inst_score[t] = reward
        ucb_total_rewards[i] += reward
        ucb_best_score[t] = ucb_best_score[t-1] + best if t > 0 else best
        ucb_alg_score[t] = ucb_alg_score[t-1] + ucb_inst_score[t] if t > 0 else ucb_inst_score[t]
        ucb_estimate_M[i] = ucb_total_rewards[i] / ucb_pulls[i]
        ucb_regret[t] = (ucb_best_score[t] - ucb_alg_score[t]) / (t + 1)
    
    # Simulação principal
    for t in range(T):
        if t == 0:
            # Primeira rodada: escolhe braço aleatório
            kth = np.random.randint(0, k)
        else:
            # UCB1: exploration bonus
            exploration_bonus = np.sqrt(np.log(t) / (ucb_pulls + 0.0001))
            kth = np.argmax(ucb_estimate_M + exploration_bonus)
        
        reward = pull(kth)
        update_stats_ucb(reward, kth, t)
        success_rate(ucb_optimal_action, kth, t)
    
    return {
        'regret': ucb_regret,
        'optimal_action': ucb_optimal_action,
        'mean_rewards': mean,
        'best_arm': index_best,
        'pulls': ucb_pulls,
        'total_rewards': ucb_total_rewards,
        'final_regret': ucb_regret[-1],
        'final_success_rate': ucb_optimal_action[-1],
        'a': a,
        'b': b
    }

# Execução da simulação
if run_simulation or 'simulation_results' not in st.session_state:
    with st.spinner('Executando simulação UCB1...'):
        results = run_ucb1_simulation(k, T, seed)
        st.session_state.simulation_results = results
        st.session_state.k = k
        st.session_state.T = T

# Verificar se há resultados para mostrar
if 'simulation_results' in st.session_state:
    results = st.session_state.simulation_results
    
    # Estatísticas principais
    st.header("📊 Estatísticas da Simulação")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Melhor Braço", f"#{results['best_arm']}")
    
    with col2:
        st.metric("Taxa de Sucesso Final", f"{results['final_success_rate']:.2%}")
    
    with col3:
        st.metric("Regret Final", f"{results['final_regret']:.4f}")
    
    with col4:
        best_reward = results['mean_rewards'][results['best_arm']]
        st.metric("Recompensa Média do Melhor Braço", f"{best_reward:.4f}")
    
    # Gráficos principais
    st.header("📈 Visualizações")
    
    # Layout em duas colunas para os gráficos
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Regret ao Longo do Tempo")
        
        # Gráfico de Regret usando Plotly
        fig_regret = go.Figure()
        fig_regret.add_trace(go.Scatter(
            x=np.arange(1, st.session_state.T + 1),
            y=results['regret'],
            mode='lines',
            name='UCB1',
            line=dict(color='red', width=2)
        ))
        
        fig_regret.update_layout(
            title=f"Regret para T={st.session_state.T} rodadas e k={st.session_state.k} bandits",
            xaxis_title="Rodada T",
            yaxis_title="Regret Total",
            hovermode='x'
        )
        
        st.plotly_chart(fig_regret, use_container_width=True)
    
    with col_right:
        st.subheader("Porcentagem de Ações Ótimas")
        
        # Gráfico de Ações Ótimas usando Plotly
        fig_optimal = go.Figure()
        fig_optimal.add_trace(go.Scatter(
            x=np.arange(1, st.session_state.T + 1),
            y=results['optimal_action'] * 100,  # Converter para porcentagem
            mode='lines',
            name='UCB1',
            line=dict(color='blue', width=2)
        ))
        
        fig_optimal.update_layout(
            title=f"Ação ótima tomada para T={st.session_state.T} rodadas e k={st.session_state.k} bandits",
            xaxis_title="Rodada T",
            yaxis_title="% Ação ótima tomada",
            hovermode='x'
        )
        
        st.plotly_chart(fig_optimal, use_container_width=True)
    
    # Gráfico da distribuição de recompensas dos braços
    st.subheader("Distribuição de Recompensas por Braço")
    
    # Criar DataFrame para melhor visualização
    arms_data = pd.DataFrame({
        'Braço': [f'Braço {i}' for i in range(st.session_state.k)],
        'Recompensa Média': results['mean_rewards'],
        'Número de Puxadas': results['pulls'],
        'Recompensa Total': results['total_rewards'],
        'É o Melhor': ['Sim' if i == results['best_arm'] else 'Não' for i in range(st.session_state.k)]
    })
    
    # Gráfico de barras das recompensas médias
    fig_arms = px.bar(
        arms_data, 
        x='Braço', 
        y='Recompensa Média',
        color='É o Melhor',
        color_discrete_map={'Sim': 'gold', 'Não': 'lightblue'},
        title="Recompensa Média por Braço"
    )
    
    fig_arms.update_layout(
        xaxis_title="Braços",
        yaxis_title="Recompensa Média",
        showlegend=True
    )
    
    st.plotly_chart(fig_arms, use_container_width=True)
    
    # Distribuição Binomial Negativa
    st.subheader("Distribuição Binomial Negativa")
    
    # Parâmetros para a distribuição binomial negativa
    st.sidebar.subheader("Parâmetros Distribuição Binomial Negativa")
    n_failures = st.sidebar.slider("Número de falhas (r)", min_value=1, max_value=50, value=10)
    prob_success = st.sidebar.slider("Probabilidade de sucesso (p)", min_value=0.01, max_value=0.99, value=0.3, step=0.01)
    
    # Gerar distribuição binomial negativa
    x_values = np.arange(0, 100)
    nb_pmf = nbinom.pmf(x_values, n_failures, prob_success)
    
    fig_nb = go.Figure()
    fig_nb.add_trace(go.Scatter(
        x=x_values,
        y=nb_pmf,
        mode='lines+markers',
        name=f'Binomial Negativa (r={n_failures}, p={prob_success})',
        line=dict(color='green', width=2),
        marker=dict(size=4)
    ))
    
    fig_nb.update_layout(
        title=f"Distribuição Binomial Negativa (r={n_failures}, p={prob_success})",
        xaxis_title="Número de sucessos",
        yaxis_title="Probabilidade",
        hovermode='x'
    )
    
    st.plotly_chart(fig_nb, use_container_width=True)
    
    # Tabela detalhada dos braços
    st.subheader("Detalhes dos Braços")
    st.dataframe(arms_data, use_container_width=True)
    
    # Informações adicionais sobre a distribuição binomial negativa
    st.subheader("Informações sobre a Distribuição Binomial Negativa")
    st.write(f"""
    A distribuição binomial negativa modela o número de sucessos em uma sequência de tentativas independentes antes de obter um número fixo de falhas.
    
    **Parâmetros atuais:**
    - **r (número de falhas):** {n_failures}
    - **p (probabilidade de sucesso):** {prob_success}
    
    **Estatísticas:**
    - **Média:** {n_failures * (1 - prob_success) / prob_success:.2f}
    - **Variância:** {n_failures * (1 - prob_success) / (prob_success ** 2):.2f}
    """)

else:
    st.info("👆 Configure os parâmetros na barra lateral e clique em 'Executar Simulação' para começar!")
    
    # Explicação do algoritmo
    st.header("ℹ️ Sobre o Algoritmo UCB1")
    st.write("""
    O **Upper Confidence Bound (UCB1)** é um algoritmo para o problema do multi-armed bandit que equilibra exploração e exploração.
    
    **Como funciona:**
    1. **Exploração vs Exploração:** O algoritmo escolhe o braço que maximiza a recompensa estimada mais um bônus de exploração
    2. **Fórmula UCB1:** Escolhe o braço i que maximiza: μᵢ + √(2ln(t)/nᵢ)
       - μᵢ: recompensa média estimada do braço i
       - t: número total de rodadas
       - nᵢ: número de vezes que o braço i foi escolhido
    3. **Bônus de Exploração:** Diminui conforme o braço é mais explorado, incentivando a exploração de braços menos testados
    
    **Métricas Importantes:**
    - **Regret:** Diferença entre a recompensa ótima e a recompensa obtida
    - **Taxa de Ações Ótimas:** Porcentagem de vezes que o melhor braço foi escolhido
    """)
