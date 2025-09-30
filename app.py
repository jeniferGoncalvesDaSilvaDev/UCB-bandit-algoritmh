
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import beta
import random
import math

# Configuração da página
st.set_page_config(
    page_title="🎰 Simulação Multi-Armed Bandit",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🎰 Simulação Interativa de Algoritmos Multi-Armed Bandit")
st.markdown("Uma aplicação web interativa para simular e comparar algoritmos de Multi-Armed Bandit")

# Classe para algoritmos Multi-Armed Bandit
class MultiArmedBandit:
    def __init__(self, k, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.k = k
        # Gera distribuições uniformes aleatórias para cada braço
        self.a = np.random.uniform(0, 1, k)
        self.b = self.a + np.random.uniform(0, 1, k)
        
        # Encontra o braço ótimo
        self.optimal_arm = np.argmax((self.a + self.b) / 2)
        
    def pull(self, arm):
        """Retorna recompensa do braço especificado"""
        return np.random.uniform(self.a[arm], self.b[arm])
    
    def get_expected_rewards(self):
        """Retorna recompensas esperadas de todos os braços"""
        return (self.a + self.b) / 2

# Algoritmo UCB1
def ucb1_algorithm(bandit, T):
    k = bandit.k
    pulls = np.zeros(k)
    rewards = np.zeros(k)
    total_rewards = np.zeros(k)
    regrets = []
    optimal_actions = []
    
    for t in range(T):
        if t < k:
            # Primeira rodada: puxa cada braço uma vez
            chosen_arm = t
        else:
            # UCB1: exploration bonus
            exploration_bonus = np.sqrt(2 * np.log(t + 1) / (pulls + 1e-10))
            estimates = total_rewards / (pulls + 1e-10)
            ucb_values = estimates + exploration_bonus
            chosen_arm = np.argmax(ucb_values)
        
        # Puxa o braço escolhido
        reward = bandit.pull(chosen_arm)
        pulls[chosen_arm] += 1
        total_rewards[chosen_arm] += reward
        
        # Calcula regret
        optimal_reward = bandit.get_expected_rewards()[bandit.optimal_arm]
        current_regret = optimal_reward - reward
        regrets.append(current_regret)
        
        # Verifica se escolheu ação ótima
        optimal_actions.append(chosen_arm == bandit.optimal_arm)
    
    return regrets, optimal_actions, pulls, total_rewards

# Algoritmo Epsilon-Greedy
def epsilon_greedy_algorithm(bandit, T, epsilon):
    k = bandit.k
    pulls = np.zeros(k)
    total_rewards = np.zeros(k)
    regrets = []
    optimal_actions = []
    
    for t in range(T):
        if np.random.random() < epsilon:
            # Exploração: escolha aleatória
            chosen_arm = np.random.randint(k)
        else:
            # Exploração: melhor braço atual
            estimates = total_rewards / (pulls + 1e-10)
            chosen_arm = np.argmax(estimates)
        
        # Puxa o braço escolhido
        reward = bandit.pull(chosen_arm)
        pulls[chosen_arm] += 1
        total_rewards[chosen_arm] += reward
        
        # Calcula regret
        optimal_reward = bandit.get_expected_rewards()[bandit.optimal_arm]
        current_regret = optimal_reward - reward
        regrets.append(current_regret)
        
        # Verifica se escolheu ação ótima
        optimal_actions.append(chosen_arm == bandit.optimal_arm)
    
    return regrets, optimal_actions, pulls, total_rewards

# Algoritmo Thompson Sampling
def thompson_sampling_algorithm(bandit, T, alpha_prior=1, beta_prior=1):
    k = bandit.k
    pulls = np.zeros(k)
    total_rewards = np.zeros(k)
    regrets = []
    optimal_actions = []
    
    # Parâmetros Beta para cada braço
    alpha = np.full(k, alpha_prior)
    beta_params = np.full(k, beta_prior)
    
    for t in range(T):
        # Amostra de distribuições Beta
        samples = [np.random.beta(alpha[i], beta_params[i]) for i in range(k)]
        chosen_arm = np.argmax(samples)
        
        # Puxa o braço escolhido
        reward = bandit.pull(chosen_arm)
        pulls[chosen_arm] += 1
        total_rewards[chosen_arm] += reward
        
        # Normaliza recompensa para [0,1]
        normalized_reward = (reward - bandit.a[chosen_arm]) / (bandit.b[chosen_arm] - bandit.a[chosen_arm])
        
        # Atualiza distribuição Beta
        alpha[chosen_arm] += normalized_reward
        beta_params[chosen_arm] += (1 - normalized_reward)
        
        # Calcula regret
        optimal_reward = bandit.get_expected_rewards()[bandit.optimal_arm]
        current_regret = optimal_reward - reward
        regrets.append(current_regret)
        
        # Verifica se escolheu ação ótima
        optimal_actions.append(chosen_arm == bandit.optimal_arm)
    
    return regrets, optimal_actions, pulls, total_rewards

# Algoritmo Gradient Bandit
def gradient_bandit_algorithm(bandit, T, learning_rate=0.1):
    k = bandit.k
    pulls = np.zeros(k)
    total_rewards = np.zeros(k)
    regrets = []
    optimal_actions = []
    
    # Preferências iniciais
    preferences = np.zeros(k)
    baseline = 0
    
    for t in range(T):
        # Calcula probabilidades softmax
        exp_prefs = np.exp(preferences - np.max(preferences))  # Estabilidade numérica
        probabilities = exp_prefs / np.sum(exp_prefs)
        
        # Escolhe braço baseado nas probabilidades
        chosen_arm = np.random.choice(k, p=probabilities)
        
        # Puxa o braço escolhido
        reward = bandit.pull(chosen_arm)
        pulls[chosen_arm] += 1
        total_rewards[chosen_arm] += reward
        
        # Atualiza baseline
        baseline = (baseline * t + reward) / (t + 1)
        
        # Atualiza preferências
        for arm in range(k):
            if arm == chosen_arm:
                preferences[arm] += learning_rate * (reward - baseline) * (1 - probabilities[arm])
            else:
                preferences[arm] -= learning_rate * (reward - baseline) * probabilities[arm]
        
        # Calcula regret
        optimal_reward = bandit.get_expected_rewards()[bandit.optimal_arm]
        current_regret = optimal_reward - reward
        regrets.append(current_regret)
        
        # Verifica se escolheu ação ótima
        optimal_actions.append(chosen_arm == bandit.optimal_arm)
    
    return regrets, optimal_actions, pulls, total_rewards

# Função para plotar resultados
def plot_results(regrets_dict, optimal_actions_dict, T):
    # Cria subplots verticais (um em cima do outro)
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Regret Médio Acumulado', 'Taxa de Ações Ótimas'),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    colors = {'UCB1': '#1f77b4', 'Epsilon-Greedy': '#ff7f0e', 
              'Thompson Sampling': '#2ca02c', 'Gradient Bandit': '#d62728'}
    
    for algo_name in regrets_dict.keys():
        # Regret cumulativo médio
        cumulative_regret = np.cumsum(regrets_dict[algo_name]) / np.arange(1, T + 1)
        
        # Taxa de ações ótimas
        optimal_rate = np.cumsum(optimal_actions_dict[algo_name]) / np.arange(1, T + 1) * 100
        
        # Adiciona traces
        fig.add_trace(
            go.Scatter(x=list(range(1, T + 1)), y=cumulative_regret,
                      mode='lines', name=f'{algo_name}',
                      line=dict(color=colors.get(algo_name, '#000000'))),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=list(range(1, T + 1)), y=optimal_rate,
                      mode='lines', name=f'{algo_name}',
                      line=dict(color=colors.get(algo_name, '#000000')),
                      showlegend=False),
            row=2, col=1
        )
    
    # Atualiza layout
    fig.update_xaxes(title_text="Tentativas", row=1, col=1)
    fig.update_xaxes(title_text="Tentativas", row=2, col=1)
    fig.update_yaxes(title_text="Regret Médio", row=1, col=1)
    fig.update_yaxes(title_text="Taxa de Ações Ótimas (%)", row=2, col=1)
    
    fig.update_layout(height=700, showlegend=True)
    
    return fig

# Função para plotar distribuições dos braços
def plot_arm_distributions(bandit, pulls, total_rewards):
    fig = go.Figure()
    
    expected_rewards = bandit.get_expected_rewards()
    actual_rewards = total_rewards / (pulls + 1e-10)
    
    arms = list(range(bandit.k))
    
    # Barra para recompensas esperadas
    fig.add_trace(go.Bar(
        x=arms,
        y=expected_rewards,
        name='Recompensa Esperada',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    # Barra para recompensas observadas
    fig.add_trace(go.Bar(
        x=arms,
        y=actual_rewards,
        name='Recompensa Observada',
        marker_color='darkblue',
        opacity=0.8
    ))
    
    # Destaca braço ótimo
    optimal_arm = bandit.optimal_arm
    fig.add_shape(
        type="rect",
        x0=optimal_arm-0.4, y0=0,
        x1=optimal_arm+0.4, y1=max(expected_rewards)*1.1,
        line=dict(color="red", width=3),
        fillcolor="rgba(255,0,0,0.1)"
    )
    
    fig.update_layout(
        title="Distribuição de Recompensas por Braço",
        xaxis_title="Braço",
        yaxis_title="Recompensa",
        barmode='group',
        height=400
    )
    
    return fig

# Sidebar com controles
st.sidebar.title("⚙️ Configurações")

# Seleção do modo
mode = st.sidebar.selectbox(
    "🤖 Modo de Simulação",
    ["Algoritmo Individual", "Comparação Completa"],
    help="Execute um algoritmo ou compare todos simultaneamente"
)

# Parâmetros básicos
st.sidebar.subheader("📊 Parâmetros Básicos")

k = st.sidebar.slider("Número de braços (k)", 2, 50, 10, help="Número de opções disponíveis")
T = st.sidebar.slider("Número de tentativas (T)", 100, 50000, 1000, help="Número total de rounds da simulação")
seed = st.sidebar.number_input("Seed (opcional)", value=42, help="Para resultados reproduzíveis")

if mode == "Algoritmo Individual":
    algorithm = st.sidebar.selectbox(
        "Algoritmo",
        ["UCB1", "Epsilon-Greedy", "Thompson Sampling", "Gradient Bandit"]
    )
    
    # Parâmetros específicos do algoritmo
    if algorithm == "Epsilon-Greedy":
        epsilon = st.sidebar.slider("Epsilon (ε)", 0.01, 0.5, 0.1, 0.01, help="Taxa de exploração")
    elif algorithm == "Thompson Sampling":
        alpha_prior = st.sidebar.slider("Alpha Prior (α)", 0.1, 10.0, 1.0, 0.1, help="Prior para sucessos")
        beta_prior = st.sidebar.slider("Beta Prior (β)", 0.1, 10.0, 1.0, 0.1, help="Prior para falhas")
    elif algorithm == "Gradient Bandit":
        learning_rate = st.sidebar.slider("Learning Rate (α)", 0.01, 1.0, 0.1, 0.01, help="Taxa de aprendizado")

# Controles adicionais
st.sidebar.subheader("🎛️ Controles")
show_arm_details = st.sidebar.checkbox("✅ Mostrar detalhes dos braços", value=True)
show_distribution = st.sidebar.checkbox("✅ Mostrar distribuição binomial negativa", value=False)

# Informações sobre algoritmos
with st.sidebar.expander("ℹ️ Informações sobre Algoritmos"):
    st.markdown("""
    **UCB1**: Usa intervalos de confiança para equilibrar exploração/exploração
    
    **Epsilon-Greedy**: Explora aleatoriamente com probabilidade ε
    
    **Thompson Sampling**: Abordagem Bayesiana com distribuições Beta
    
    **Gradient Bandit**: Aprende preferências usando gradiente estocástico
    """)

# Botão para executar simulação
if st.sidebar.button("🚀 Executar Simulação", type="primary"):
    with st.spinner("Executando simulação..."):
        # Cria bandit
        bandit = MultiArmedBandit(k, seed)
        
        if mode == "Algoritmo Individual":
            # Executa algoritmo individual
            if algorithm == "UCB1":
                regrets, optimal_actions, pulls, total_rewards = ucb1_algorithm(bandit, T)
            elif algorithm == "Epsilon-Greedy":
                regrets, optimal_actions, pulls, total_rewards = epsilon_greedy_algorithm(bandit, T, epsilon)
            elif algorithm == "Thompson Sampling":
                regrets, optimal_actions, pulls, total_rewards = thompson_sampling_algorithm(bandit, T, alpha_prior, beta_prior)
            elif algorithm == "Gradient Bandit":
                regrets, optimal_actions, pulls, total_rewards = gradient_bandit_algorithm(bandit, T, learning_rate)
            
            # Métricas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Regret Final", f"{np.mean(regrets):.4f}")
            with col2:
                st.metric("Taxa de Sucesso", f"{np.mean(optimal_actions)*100:.1f}%")
            with col3:
                st.metric("Braço Ótimo", f"{bandit.optimal_arm}")
            
            # Gráficos
            regrets_dict = {algorithm: regrets}
            optimal_actions_dict = {algorithm: optimal_actions}
            
            fig = plot_results(regrets_dict, optimal_actions_dict, T)
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribuição dos braços
            if show_arm_details:
                st.subheader("📊 Análise Detalhada dos Braços")
                
                fig_dist = plot_arm_distributions(bandit, pulls, total_rewards)
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Tabela de estatísticas
                df_stats = pd.DataFrame({
                    'Braço': range(k),
                    'Recompensa Esperada': bandit.get_expected_rewards(),
                    'Recompensa Observada': total_rewards / (pulls + 1e-10),
                    'Número de Puxadas': pulls.astype(int),
                    'Recompensa Total': total_rewards,
                    'É Ótimo': ['✅' if i == bandit.optimal_arm else '❌' for i in range(k)]
                })
                
                st.dataframe(df_stats, use_container_width=True)
        
        else:  # Comparação Completa
            # Executa todos os algoritmos
            algorithms = {
                'UCB1': lambda: ucb1_algorithm(bandit, T),
                'Epsilon-Greedy': lambda: epsilon_greedy_algorithm(bandit, T, 0.1),
                'Thompson Sampling': lambda: thompson_sampling_algorithm(bandit, T, 1.0, 1.0),
                'Gradient Bandit': lambda: gradient_bandit_algorithm(bandit, T, 0.1)
            }
            
            regrets_dict = {}
            optimal_actions_dict = {}
            results_summary = {}
            
            for algo_name, algo_func in algorithms.items():
                regrets, optimal_actions, pulls, total_rewards = algo_func()
                regrets_dict[algo_name] = regrets
                optimal_actions_dict[algo_name] = optimal_actions
                results_summary[algo_name] = {
                    'regret_final': np.mean(regrets),
                    'taxa_sucesso': np.mean(optimal_actions) * 100
                }
            
            # Métricas comparativas
            st.subheader("📈 Comparação de Performance")
            
            cols = st.columns(4)
            for i, (algo_name, metrics) in enumerate(results_summary.items()):
                with cols[i]:
                    st.metric(
                        f"{algo_name}",
                        f"Regret: {metrics['regret_final']:.4f}",
                        f"Sucesso: {metrics['taxa_sucesso']:.1f}%"
                    )
            
            # Gráfico comparativo
            fig = plot_results(regrets_dict, optimal_actions_dict, T)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabela de comparação
            df_comparison = pd.DataFrame(results_summary).T
            df_comparison.columns = ['Regret Final', 'Taxa de Sucesso (%)']
            df_comparison['Ranking Regret'] = df_comparison['Regret Final'].rank()
            df_comparison['Ranking Sucesso'] = df_comparison['Taxa de Sucesso (%)'].rank(ascending=False)
            
            st.dataframe(df_comparison.style.highlight_min(subset=['Regret Final'], color='lightgreen')
                                           .highlight_max(subset=['Taxa de Sucesso (%)'], color='lightgreen'),
                        use_container_width=True)

# Distribuição binomial negativa (educacional)
if show_distribution:
    st.subheader("📊 Distribuição Binomial Negativa (Educacional)")
    
    col1, col2 = st.columns(2)
    with col1:
        r = st.slider("Parâmetro r (falhas)", 1, 20, 5)
        p = st.slider("Parâmetro p (probabilidade)", 0.01, 0.99, 0.3, 0.01)
    
    with col2:
        x = np.arange(0, 50)
        pmf = stats.nbinom.pmf(x, r, p)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=pmf, name='PMF'))
        fig.update_layout(
            title=f"Distribuição Binomial Negativa (r={r}, p={p})",
            xaxis_title="Valor",
            yaxis_title="Probabilidade",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*Desenvolvido com ❤️ para demonstrar os conceitos fundamentais de Multi-Armed Bandit de forma interativa e educacional.*")
