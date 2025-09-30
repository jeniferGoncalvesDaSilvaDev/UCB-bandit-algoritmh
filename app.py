import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import nbinom, beta as beta_dist
import pandas as pd

# Configuração da página
st.set_page_config(
    page_title="Simulação Multi-Armed Bandit",
    page_icon="🎰",
    layout="wide"
)

# Título principal
st.title("🎰 Simulação Interativa de Algoritmos Multi-Armed Bandit")

# Sidebar para controles
st.sidebar.header("🎯 Configurações da Simulação")

# Seleção do algoritmo
algorithm = st.sidebar.selectbox(
    "🤖 Algoritmo", 
    ["UCB1", "Epsilon-Greedy", "Thompson Sampling", "Gradient Bandit", "Comparação Todos"],
    help="Escolha o algoritmo para simulação"
)

# Seção de parâmetros básicos
st.sidebar.subheader("⚙️ Parâmetros Básicos")
k = st.sidebar.slider("Número de braços (k)", min_value=2, max_value=50, value=10, help="Número de máquinas caça-níqueis")
T = st.sidebar.slider("Número de tentativas (T)", min_value=100, max_value=50000, value=10000, step=100, help="Número total de rodadas")

# Parâmetros específicos por algoritmo
epsilon = 0.1  # Valor padrão
alpha_prior = 1.0  # Prior para Thompson Sampling
beta_prior = 1.0  # Prior para Thompson Sampling
learning_rate = 0.1  # Learning rate para Gradient Bandit

if algorithm in ["Epsilon-Greedy", "Comparação Todos"]:
    st.sidebar.subheader("🎯 Parâmetros Epsilon-Greedy")
    epsilon = st.sidebar.slider("Epsilon (ε)", min_value=0.01, max_value=0.5, value=0.1, step=0.01, help="Taxa de exploração para Epsilon-Greedy")

if algorithm in ["Thompson Sampling", "Comparação Todos"]:
    st.sidebar.subheader("🎲 Parâmetros Thompson Sampling")
    alpha_prior = st.sidebar.slider("Alpha (α) Prior", min_value=0.1, max_value=10.0, value=1.0, step=0.1, help="Prior de sucessos (Beta distribution)")
    beta_prior = st.sidebar.slider("Beta (β) Prior", min_value=0.1, max_value=10.0, value=1.0, step=0.1, help="Prior de falhas (Beta distribution)")

if algorithm in ["Gradient Bandit", "Comparação Todos"]:
    st.sidebar.subheader("📈 Parâmetros Gradient Bandit")
    learning_rate = st.sidebar.slider("Learning Rate (α)", min_value=0.01, max_value=1.0, value=0.1, step=0.01, help="Taxa de aprendizado para atualização de preferências")

# Opções avançadas
st.sidebar.subheader("🔧 Opções Avançadas")
seed = st.sidebar.number_input("Seed (opcional)", min_value=0, max_value=9999, value=42, help="Para resultados reproduzíveis")

# Personalização de exibição
show_arm_details = st.sidebar.checkbox("Mostrar detalhes dos braços", value=True, help="Exibir tabela com estatísticas detalhadas")
show_binomial = st.sidebar.checkbox("Mostrar distribuição binomial negativa", value=True, help="Exibir gráfico da distribuição binomial negativa")

# Botão para executar simulação
st.sidebar.markdown("---")
run_simulation = st.sidebar.button("🚀 Executar Simulação", type="primary")

# Informações sobre a simulação
if st.sidebar.button("ℹ️ Sobre os Algoritmos"):
    st.sidebar.info("""
    **UCB1**: Usa confiança superior (Upper Confidence Bound) para balancear exploração e exploração.
    
    **Epsilon-Greedy**: Explora aleatoriamente com probabilidade ε, senão explora o melhor braço conhecido.
    
    **Thompson Sampling**: Abordagem Bayesiana que usa distribuições Beta para modelar incerteza e amostrar ações.
    
    **Gradient Bandit**: Aprende preferências de ações usando gradiente estocástico e softmax para seleção.
    
    **Comparação Todos**: Executa todos os algoritmos simultaneamente para comparação completa.
    """)

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
    index_best = int(np.argmax(mean))  # CORRIGIDO
    
    def pull(i):
        """Puxa a alavanca i e retorna recompensa"""
        return np.random.uniform(a[i], b[i])
    
    success = 0
    
    def success_rate(optimal, i, total):
        """Calcula taxa de sucesso"""
        nonlocal success
        if i == index_best:
            success += 1
        optimal[total] = success / (total + 1)
    
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
        if t < k:
            # Primeiras k rodadas: puxar cada braço uma vez
            kth = t
        else:
            # UCB1: exploration bonus com fórmula correta
            exploration_bonus = np.sqrt(2 * np.log(t + 1) / ucb_pulls)
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

def run_epsilon_greedy_simulation(k, T, epsilon, seed=None):
    """Executa a simulação do algoritmo Epsilon-Greedy"""
    if seed is not None:
        np.random.seed(seed)
    
    # Vetores do modelo
    eg_pulls = np.zeros(k)
    eg_estimate_M = np.zeros(k)
    eg_total_rewards = np.zeros(k)
    eg_inst_score = np.zeros(T)
    eg_best_score = np.zeros(T)
    eg_alg_score = np.zeros(T)
    eg_regret = np.zeros(T)
    eg_optimal_action = np.zeros(T)
    
    # Definindo distribuições de probabilidade (mesmas do UCB para comparação justa)
    a = np.random.random(k)
    b = np.random.random(k)
    for i in range(k):
        if a[i] > b[i]:
            a[i], b[i] = b[i], a[i]
    
    mean = (a + b) / 2
    best = np.max(mean)
    index_best = int(np.argmax(mean))  # CORRIGIDO
    
    def pull(i):
        """Puxa a alavanca i e retorna recompensa"""
        return np.random.uniform(a[i], b[i])
    
    success = 0
    
    def success_rate(optimal, i, total):
        """Calcula taxa de sucesso"""
        nonlocal success
        if i == index_best:
            success += 1
        optimal[total] = success / (total + 1)
    
    def update_stats_eg(reward, i, t):
        """Atualiza estatísticas do Epsilon-Greedy"""
        eg_pulls[i] += 1
        eg_inst_score[t] = reward
        eg_total_rewards[i] += reward
        eg_best_score[t] = eg_best_score[t-1] + best if t > 0 else best
        eg_alg_score[t] = eg_alg_score[t-1] + eg_inst_score[t] if t > 0 else eg_inst_score[t]
        eg_estimate_M[i] = eg_total_rewards[i] / eg_pulls[i]
        eg_regret[t] = (eg_best_score[t] - eg_alg_score[t]) / (t + 1)
    
    # Simulação principal
    for t in range(T):
        # Epsilon-Greedy: explorar vs exploitar
        if np.random.random() < epsilon:
            # Explorar: escolher braço aleatório
            kth = np.random.randint(0, k)
        else:
            # Exploitar: escolher melhor braço conhecido
            # Braços não testados têm estimativa 0 devido à inicialização
            estimates = np.copy(eg_estimate_M)
            estimates[eg_pulls == 0] = 0  # Braços não testados têm estimativa 0
            kth = np.argmax(estimates)
        
        reward = pull(kth)
        update_stats_eg(reward, kth, t)
        success_rate(eg_optimal_action, kth, t)
    
    return {
        'regret': eg_regret,
        'optimal_action': eg_optimal_action,
        'mean_rewards': mean,
        'best_arm': index_best,
        'pulls': eg_pulls,
        'total_rewards': eg_total_rewards,
        'final_regret': eg_regret[-1],
        'final_success_rate': eg_optimal_action[-1],
        'a': a,
        'b': b
    }

def run_thompson_sampling_simulation(k, T, alpha_prior=1.0, beta_prior=1.0, seed=None):
    """Executa a simulação do algoritmo Thompson Sampling"""
    if seed is not None:
        np.random.seed(seed)
    
    # Vetores do modelo
    ts_pulls = np.zeros(k)
    ts_total_rewards = np.zeros(k)
    ts_inst_score = np.zeros(T)
    ts_best_score = np.zeros(T)
    ts_alg_score = np.zeros(T)
    ts_regret = np.zeros(T)
    ts_optimal_action = np.zeros(T)
    
    # Parâmetros Beta para Thompson Sampling
    alpha = np.full(k, alpha_prior)  # Sucessos + prior
    beta = np.full(k, beta_prior)    # Falhas + prior
    
    # Definindo distribuições de probabilidade
    a = np.random.random(k)
    b = np.random.random(k)
    for i in range(k):
        if a[i] > b[i]:
            a[i], b[i] = b[i], a[i]
    
    mean = (a + b) / 2
    best = np.max(mean)
    index_best = int(np.argmax(mean))  # CORRIGIDO
    
    def pull(i):
        """Puxa a alavanca i e retorna recompensa"""
        return np.random.uniform(a[i], b[i])
    
    success = 0
    
    def success_rate(optimal, i, total):
        """Calcula taxa de sucesso"""
        nonlocal success
        if i == index_best:
            success += 1
        optimal[total] = success / (total + 1)
    
    def update_stats_ts(reward, i, t):
        """Atualiza estatísticas do Thompson Sampling"""
        ts_pulls[i] += 1
        ts_inst_score[t] = reward
        ts_total_rewards[i] += reward
        ts_best_score[t] = ts_best_score[t-1] + best if t > 0 else best
        ts_alg_score[t] = ts_alg_score[t-1] + ts_inst_score[t] if t > 0 else ts_inst_score[t]
        ts_regret[t] = (ts_best_score[t] - ts_alg_score[t]) / (t + 1)
        
        # Atualizar distribuição Beta
        # Usar escala global [0,1] para todas as recompensas
        # Encontrar min e max globais das distribuições
        global_min = np.min(a)
        global_max = np.max(b)
        normalized_reward = (reward - global_min) / (global_max - global_min) if global_max > global_min else 0.5
        alpha[i] += normalized_reward
        beta[i] += (1 - normalized_reward)
    
    # Simulação principal
    for t in range(T):
        # Thompson Sampling: amostrar de cada distribuição Beta
        samples = [beta_dist.rvs(alpha[i], beta[i]) for i in range(k)]
        kth = np.argmax(samples)
        
        reward = pull(kth)
        update_stats_ts(reward, kth, t)
        success_rate(ts_optimal_action, kth, t)
    
    # Calcular estimativas médias
    ts_estimate_M = alpha / (alpha + beta)
    
    return {
        'regret': ts_regret,
        'optimal_action': ts_optimal_action,
        'mean_rewards': mean,
        'best_arm': index_best,
        'pulls': ts_pulls,
        'total_rewards': ts_total_rewards,
        'final_regret': ts_regret[-1],
        'final_success_rate': ts_optimal_action[-1],
        'a': a,
        'b': b
    }

def run_gradient_bandit_simulation(k, T, learning_rate=0.1, seed=None):
    """Executa a simulação do algoritmo Gradient Bandit"""
    if seed is not None:
        np.random.seed(seed)
    
    # Vetores do modelo
    gb_pulls = np.zeros(k)
    gb_total_rewards = np.zeros(k)
    gb_inst_score = np.zeros(T)
    gb_best_score = np.zeros(T)
    gb_alg_score = np.zeros(T)
    gb_regret = np.zeros(T)
    gb_optimal_action = np.zeros(T)
    
    # Preferências iniciais (todas zero = probabilidades iguais)
    preferences = np.zeros(k)
    
    # Baseline (média de recompensa)
    avg_reward = 0.0
    
    # Definindo distribuições de probabilidade
    a = np.random.random(k)
    b = np.random.random(k)
    for i in range(k):
        if a[i] > b[i]:
            a[i], b[i] = b[i], a[i]
    
    mean = (a + b) / 2
    best = np.max(mean)
    index_best = int(np.argmax(mean))  # CORRIGIDO
    
    def pull(i):
        """Puxa a alavanca i e retorna recompensa"""
        return np.random.uniform(a[i], b[i])
    
    success = 0
    
    def success_rate(optimal, i, total):
        """Calcula taxa de sucesso"""
        nonlocal success
        if i == index_best:
            success += 1
        optimal[total] = success / (total + 1)
    
    def get_action_probabilities(H):
        """Calcula probabilidades softmax das preferências"""
        exp_H = np.exp(H - np.max(H))  # Estabilidade numérica
        return exp_H / np.sum(exp_H)
    
    def update_stats_gb(reward, i, t):
        """Atualiza estatísticas do Gradient Bandit"""
        gb_pulls[i] += 1
        gb_inst_score[t] = reward
        gb_total_rewards[i] += reward
        gb_best_score[t] = gb_best_score[t-1] + best if t > 0 else best
        gb_alg_score[t] = gb_alg_score[t-1] + gb_inst_score[t] if t > 0 else gb_inst_score[t]
        gb_regret[t] = (gb_best_score[t] - gb_alg_score[t]) / (t + 1)
    
    # Simulação principal
    for t in range(T):
        # Obter probabilidades atuais
        probabilities = get_action_probabilities(preferences)
        
        # Selecionar ação baseado nas probabilidades
        kth = np.random.choice(k, p=probabilities)
        
        reward = pull(kth)
        update_stats_gb(reward, kth, t)
        success_rate(gb_optimal_action, kth, t)
        
        # Guardar baseline atual antes de atualizar
        baseline = avg_reward
        
        # Atualizar baseline (média de recompensa) para próxima iteração
        avg_reward += (reward - avg_reward) / (t + 1)
        
        # Atualizar preferências usando gradiente com baseline do passo anterior
        for arm in range(k):
            if arm == kth:
                # Ação selecionada: aumentar preferência se reward > baseline
                preferences[arm] += learning_rate * (reward - baseline) * (1 - probabilities[arm])
            else:
                # Outras ações: diminuir preferência se reward > baseline
                preferences[arm] -= learning_rate * (reward - baseline) * probabilities[arm]
    
    # Estimativas finais baseadas nas preferências
    final_probs = get_action_probabilities(preferences)
    
    return {
        'regret': gb_regret,
        'optimal_action': gb_optimal_action,
        'mean_rewards': mean,
        'best_arm': index_best,
        'pulls': gb_pulls,
        'total_rewards': gb_total_rewards,
        'final_regret': gb_regret[-1],
        'final_success_rate': gb_optimal_action[-1],
        'preferences': preferences,
        'final_probabilities': final_probs,
        'a': a,
        'b': b
    }

# Execução da simulação
if run_simulation:
    if algorithm == "UCB1":
        with st.spinner('Executando simulação UCB1...'):
            results = run_ucb1_simulation(k, T, seed)
            st.session_state.simulation_results = results
            st.session_state.algorithm = algorithm
    
    elif algorithm == "Epsilon-Greedy":
        with st.spinner('Executando simulação Epsilon-Greedy...'):
            results = run_epsilon_greedy_simulation(k, T, epsilon, seed)
            st.session_state.simulation_results = results
            st.session_state.algorithm = algorithm
    
    elif algorithm == "Thompson Sampling":
        with st.spinner('Executando simulação Thompson Sampling...'):
            results = run_thompson_sampling_simulation(k, T, alpha_prior, beta_prior, seed)
            st.session_state.simulation_results = results
            st.session_state.algorithm = algorithm
    
    elif algorithm == "Gradient Bandit":
        with st.spinner('Executando simulação Gradient Bandit...'):
            results = run_gradient_bandit_simulation(k, T, learning_rate, seed)
            st.session_state.simulation_results = results
            st.session_state.algorithm = algorithm
    
    elif algorithm == "Comparação Todos":
        with st.spinner('Executando comparação de todos os algoritmos...'):
            # Executar todos os algoritmos com a mesma seed para comparação justa
            ucb_results = run_ucb1_simulation(k, T, seed)
            eg_results = run_epsilon_greedy_simulation(k, T, epsilon, seed)
            ts_results = run_thompson_sampling_simulation(k, T, alpha_prior, beta_prior, seed)
            gb_results = run_gradient_bandit_simulation(k, T, learning_rate, seed)
            
            # Armazenar resultados de todos
            st.session_state.simulation_results = {
                'ucb': ucb_results,
                'epsilon_greedy': eg_results,
                'thompson_sampling': ts_results,
                'gradient_bandit': gb_results
            }
            st.session_state.algorithm = algorithm
    
    st.session_state.k = k
    st.session_state.T = T
    if algorithm in ["Epsilon-Greedy", "Comparação Todos"]:
        st.session_state.epsilon = epsilon
    if algorithm in ["Thompson Sampling", "Comparação Todos"]:
        st.session_state.alpha_prior = alpha_prior
        st.session_state.beta_prior = beta_prior
    if algorithm in ["Gradient Bandit", "Comparação Todos"]:
        st.session_state.learning_rate = learning_rate

# Verificar se há resultados para mostrar
if 'simulation_results' in st.session_state:
    results = st.session_state.simulation_results
    current_algorithm = st.session_state.get('algorithm', 'UCB1')
    
    if current_algorithm == "Comparação Todos":
        # Modo de comparação de todos os algoritmos
        ucb_results = results['ucb']
        eg_results = results['epsilon_greedy']
        ts_results = results['thompson_sampling']
        gb_results = results['gradient_bandit']
        
        # Estatísticas principais - Comparação
        st.header("📊 Estatísticas da Comparação - Todos os Algoritmos")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.subheader("UCB1")
            st.metric("Taxa de Sucesso", f"{ucb_results['final_success_rate']:.2%}")
            st.metric("Regret Final", f"{ucb_results['final_regret']:.4f}")
        
        with col2:
            st.subheader("Epsilon-Greedy")
            st.metric("Taxa de Sucesso", f"{eg_results['final_success_rate']:.2%}")
            st.metric("Regret Final", f"{eg_results['final_regret']:.4f}")
        
        with col3:
            st.subheader("Thompson Sampling")
            st.metric("Taxa de Sucesso", f"{ts_results['final_success_rate']:.2%}")
            st.metric("Regret Final", f"{ts_results['final_regret']:.4f}")
        
        with col4:
            st.subheader("Gradient Bandit")
            st.metric("Taxa de Sucesso", f"{gb_results['final_success_rate']:.2%}")
            st.metric("Regret Final", f"{gb_results['final_regret']:.4f}")
        
        # Gráficos de comparação
        st.header("📈 Visualizações de Comparação")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Regret ao Longo do Tempo")
            
            fig_regret = go.Figure()
            fig_regret.add_trace(go.Scatter(
                x=np.arange(1, st.session_state.T + 1),
                y=ucb_results['regret'],
                mode='lines',
                name='UCB1',
                line=dict(color='red', width=2)
            ))
            fig_regret.add_trace(go.Scatter(
                x=np.arange(1, st.session_state.T + 1),
                y=eg_results['regret'],
                mode='lines',
                name='Epsilon-Greedy',
                line=dict(color='green', width=2)
            ))
            fig_regret.add_trace(go.Scatter(
                x=np.arange(1, st.session_state.T + 1),
                y=ts_results['regret'],
                mode='lines',
                name='Thompson Sampling',
                line=dict(color='blue', width=2)
            ))
            fig_regret.add_trace(go.Scatter(
                x=np.arange(1, st.session_state.T + 1),
                y=gb_results['regret'],
                mode='lines',
                name='Gradient Bandit',
                line=dict(color='purple', width=2)
            ))
            
            fig_regret.update_layout(
                title=f"Comparação de Regret - T={st.session_state.T}, k={st.session_state.k}",
                xaxis_title="Rodada T",
                yaxis_title="Regret Médio",
                hovermode='x',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_regret, use_container_width=True)
        
        with col_right:
            st.subheader("Ações Ótimas")
            
            fig_optimal = go.Figure()
            fig_optimal.add_trace(go.Scatter(
                x=np.arange(1, st.session_state.T + 1),
                y=ucb_results['optimal_action'] * 100,
                mode='lines',
                name='UCB1',
                line=dict(color='red', width=2)
            ))
            fig_optimal.add_trace(go.Scatter(
                x=np.arange(1, st.session_state.T + 1),
                y=eg_results['optimal_action'] * 100,
                mode='lines',
                name='Epsilon-Greedy',
                line=dict(color='green', width=2)
            ))
            fig_optimal.add_trace(go.Scatter(
                x=np.arange(1, st.session_state.T + 1),
                y=ts_results['optimal_action'] * 100,
                mode='lines',
                name='Thompson Sampling',
                line=dict(color='blue', width=2)
            ))
            fig_optimal.add_trace(go.Scatter(
                x=np.arange(1, st.session_state.T + 1),
                y=gb_results['optimal_action'] * 100,
                mode='lines',
                name='Gradient Bandit',
                line=dict(color='purple', width=2)
            ))
            
            fig_optimal.update_layout(
                title=f"Comparação de Ações Ótimas - T={st.session_state.T}, k={st.session_state.k}",
                xaxis_title="Rodada T",
                yaxis_title="% Ação ótima tomada",
                hovermode='x',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_optimal, use_container_width=True)
        
        # Usar os resultados do UCB para gráficos adicionais
        results_for_arms = ucb_results
        
    elif current_algorithm == "Comparação":
        # Modo de comparação
        ucb_results = results['ucb']
        eg_results = results['epsilon_greedy']
        
        # Estatísticas principais - Comparação
        st.header("📊 Estatísticas da Comparação")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("UCB1")
            st.metric("Melhor Braço", f"#{ucb_results['best_arm']}")
            st.metric("Taxa de Sucesso Final", f"{ucb_results['final_success_rate']:.2%}")
            st.metric("Regret Final", f"{ucb_results['final_regret']:.4f}")
            best_reward_ucb = ucb_results['mean_rewards'][ucb_results['best_arm']]
            st.metric("Recompensa Média do Melhor Braço", f"{best_reward_ucb:.4f}")
        
        with col2:
            st.subheader(f"Epsilon-Greedy (ε={st.session_state.get('epsilon', 0.1)})")
            st.metric("Melhor Braço", f"#{eg_results['best_arm']}")
            st.metric("Taxa de Sucesso Final", f"{eg_results['final_success_rate']:.2%}")
            st.metric("Regret Final", f"{eg_results['final_regret']:.4f}")
            best_reward_eg = eg_results['mean_rewards'][eg_results['best_arm']]
            st.metric("Recompensa Média do Melhor Braço", f"{best_reward_eg:.4f}")
        
        # Gráficos de comparação
        st.header("📈 Visualizações de Comparação")
        
        col_left, col_right = st.columns(2)
