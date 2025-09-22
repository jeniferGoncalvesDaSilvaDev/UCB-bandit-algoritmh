# 🎰 Simulação Interativa de Algoritmos Multi-Armed Bandit

Uma aplicação web interativa desenvolvida em Streamlit para simular e comparar algoritmos de Multi-Armed Bandit, incluindo UCB1 e Epsilon-Greedy.

## 📋 Sobre o Projeto

Este projeto implementa uma simulação completa do problema clássico de Multi-Armed Bandit, permitindo que usuários experimentem com diferentes algoritmos e parâmetros para entender como eles se comportam em cenários de exploração vs. exploração.

### Algoritmos Implementados

#### UCB1 (Upper Confidence Bound)
- **Fórmula**: μᵢ + √(2ln(t)/nᵢ)
- **Estratégia**: Equilibra exploração e exploração usando intervalos de confiança
- **Garantias**: Inicializa com uma puxada por braço, depois usa o índice UCB

#### Epsilon-Greedy
- **Fórmula**: Com probabilidade ε explora aleatoriamente, senão explora o melhor braço
- **Estratégia**: Simples mas efetivo para muitos cenários
- **Parâmetro**: ε controla a taxa de exploração (0.01 a 0.5)

## 🚀 Funcionalidades

### Simulação Individual
- Execute algoritmos UCB1 ou Epsilon-Greedy separadamente
- Visualize regret médio e porcentagem de ações ótimas ao longo do tempo
- Analise estatísticas detalhadas de cada braço

### Modo Comparação
- Execute ambos algoritmos simultaneamente
- Compare performance lado a lado
- Mesmo ambiente para comparação justa

### Parâmetros Configuráveis
- **Número de braços (k)**: 2 a 50
- **Número de tentativas (T)**: 100 a 50.000
- **Epsilon (ε)**: 0.01 a 0.5 (para Epsilon-Greedy)
- **Seed**: Para resultados reproduzíveis

### Visualizações Interativas
- Gráficos de regret médio ao longo do tempo
- Porcentagem de ações ótimas
- Distribuição de recompensas por braço
- Distribuição binomial negativa customizável
- Tabela detalhada com estatísticas por braço

### Controles de Interface
- ✅ Mostrar/ocultar detalhes dos braços
- ✅ Mostrar/ocultar distribuição binomial negativa
- 🤖 Seleção de algoritmo
- ⚙️ Parâmetros básicos e avançados
- ℹ️ Informações sobre algoritmos

## 🛠️ Como Usar

### 1. Configuração
1. Na barra lateral, selecione o algoritmo desejado
2. Ajuste o número de braços e tentativas
3. Configure parâmetros específicos (epsilon para Epsilon-Greedy)
4. Defina uma seed para reprodutibilidade (opcional)

### 2. Execução
1. Clique no botão "🚀 Executar Simulação"
2. Aguarde o processamento (pode levar alguns segundos para simulações grandes)
3. Analise os resultados nos gráficos e estatísticas

### 3. Análise
- **Regret Médio**: Quanto menor, melhor o algoritmo
- **Taxa de Sucesso**: Porcentagem de vezes que escolheu o braço ótimo
- **Detalhes dos Braços**: Estatísticas individuais de cada opção

## 📊 Métricas Explicadas

### Regret
O regret mede a diferença entre a recompensa ótima e a recompensa obtida:
```
Regret(t) = (Recompensa_Ótima_Acumulada - Recompensa_Obtida) / t
```

### Taxa de Ações Ótimas
Porcentagem de vezes que o algoritmo escolheu o braço com maior recompensa média:
```
Taxa_Sucesso(t) = Número_Ações_Ótimas / Total_Ações
```

## 🎯 Casos de Uso

### Educacional
- Demonstrar conceitos de exploração vs. exploração
- Comparar performance de diferentes algoritmos
- Visualizar convergência e learning curves

### Pesquisa
- Testar configurações de parâmetros
- Avaliar robustez dos algoritmos
- Análise de sensibilidade

### Prático
- Entender trade-offs em problemas de decisão
- Modelar cenários de A/B testing
- Otimização de recomendações

## 🔧 Requisitos Técnicos

### Dependências
- Python 3.11+
- Streamlit
- NumPy
- Pandas
- Plotly
- SciPy
- Matplotlib

### Configuração do Servidor
A aplicação está configurada para rodar em:
- **Endereço**: 0.0.0.0
- **Porta**: 5000
- **Modo**: Headless

## 📈 Distribuições de Recompensa

### Distribuição Uniforme (Padrão)
Cada braço tem uma distribuição uniforme U(a,b) onde:
- a e b são gerados aleatoriamente
- Garantia: a ≤ b para cada braço
- Melhor braço: aquele com maior (a+b)/2

### Distribuição Binomial Negativa (Adicional)
Visualização educacional com parâmetros configuráveis:
- **r**: Número de falhas
- **p**: Probabilidade de sucesso
- **Uso**: Demonstração de outras distribuições de probabilidade

## 🎨 Interface

### Layout Responsivo
- Sidebar com controles organizados
- Área principal com visualizações
- Colunas adaptáveis para diferentes telas

### Elementos Interativos
- Sliders para parâmetros numéricos
- Selectbox para escolhas categóricas
- Checkboxes para controles opcionais
- Botões com ações claras

### Feedback Visual
- Spinners durante processamento
- Métricas destacadas
- Gráficos interativos com hover
- Códigos de cores consistentes

## 🔬 Detalhes de Implementação

### UCB1
```python
# Primeiras k rodadas: puxar cada braço uma vez
if t < k:
    kth = t
else:
    # UCB1: exploration bonus
    exploration_bonus = sqrt(2 * log(t + 1) / pulls)
    kth = argmax(estimates + exploration_bonus)
```

### Epsilon-Greedy
```python
# Exploração vs. Exploração
if random() < epsilon:
    kth = random_choice()  # Explorar
else:
    kth = argmax(estimates)  # Explorar
```

### Comparação Justa
- Mesma seed para ambos algoritmos
- Ambiente idêntico (mesmas distribuições a,b)
- Métricas calculadas de forma consistente

## 📝 Créditos

Baseado no trabalho original de: [github.com/petroud/E-greedy_and_UCB_algorithms/](https://github.com/petroud/E-greedy_and_UCB_algorithms/)

## 🤝 Contribuição

Este projeto é educacional e está aberto para melhorias. Sugestões de novos algoritmos, visualizações ou funcionalidades são bem-vindas.

---

*Desenvolvido com ❤️ para demonstrar os conceitos fundamentais de Multi-Armed Bandit de forma interativa e educacional.*