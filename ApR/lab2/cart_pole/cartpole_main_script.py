#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['figure.figsize'] = 12, 6
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')
import CartPole_DQN

# Función para suavizar la curva de convergencia
def avg_reward(scores):
    episode_number = np.linspace(1, len(scores) + 1, len(scores) + 1)
    acumulated_rewards = np.cumsum(scores)
    return [acumulated_rewards[i] / episode_number[i] for i in \
    range(len(acumulated_rewards))]


#Parámetros
n_neurons = [
    [24,48], 
#    [40,20,40], 
#    [40,20,40], 
#    [40,20,40]
]
activations = [
    ['relu', 'relu'],
#    ['tanh', 'tanh', 'tanh'],
#    ['softsign', 'softsign', 'softsign'],
#    ['relu', 'tanh', 'softsign']
]

df = pd.DataFrame(columns=[])

def run_agent(n_neurons, activations, df):
    agent = CartPole_DQN.DQNCartPoleSolver()

    scores_DQN = agent.run()

    df['DQN_24_48_relu'] = avg_reward(scores_DQN)

    return df

for i in range(len(n_neurons)):
    df = run_agent(n_neurons[i], activations[i], df)

df.to_csv('resultados_redes_24_48_relu.csv')
df.plot(title='Recompensa promedio por episodio')
plt.ylim(0, 250)
plt.savefig('../img/cp_rec_acum_24_48_relu.png')

