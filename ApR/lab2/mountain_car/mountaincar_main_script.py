#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['figure.figsize'] = 12, 6
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')
import MountainCar_SGD
import MountainCar_SGD_Poly
import MountainCar_SGD_Poly_Dual
import MountainCar_DQN


# Función para suavizar la curva de convergencia
def avg_reward(scores):
    episode_number = np.linspace(1, len(scores) + 1, len(scores) + 1)
    acumulated_rewards = np.cumsum(scores)
    return [acumulated_rewards[i] / episode_number[i] for i in \
    range(len(acumulated_rewards))]


# Función para correr el agente
def run_agent(modelo, df, nombre):
    agent = modelo()
    scores = agent.run()
    df[nombre] = avg_reward(scores)
    return df



# ####
# # Modelo Lineal
# ####

# df = pd.DataFrame(columns=[])

# df = run_agent(MountainCar_SGD.SGDMountainCarSolver, df, 'SGD')

# #df.plot(title='Recompensa promedio por episodio')
# #plt.ylim(-250, 0)
# #plt.show()



# # ####
# # # Modelo Lineal Poly
# # ####

# # df = pd.DataFrame(columns=[])

# df = run_agent(MountainCar_SGD_Poly.SGDPolyMountainCarSolver, df, 'SGD_Poly')

# # df.plot(title='Recompensa promedio por episodio')
# # plt.ylim(-250, 0)
# # plt.show()


# # ####
# # # Modelo Lineal Poly Dual
# # ####

# # df = pd.DataFrame(columns=[])

# df = run_agent(MountainCar_SGD_Poly_Dual.SGDPolyDualMountainCarSolver, df, 'SGD_Poly_Dual')


# df.to_csv('resultados_sgd.csv')
# df.plot(title='Recompensa promedio por episodio')
# plt.savefig('../img/mc_rec_acum_sgd.png')



####
# Red Neuronal
####

#Parámetros
n_neurons = [
    [24],
    [24,48],
    [24],
    [24,48]
]
activations = [
    ['softsign'],
    ['softsign','softsign'],
    ['tanh'],
    ['tanh','tanh']
]

df = pd.DataFrame(columns=[])

def run_agent_neuron(n_neurons, activations, df):
    agent = MountainCar_DQN.DQNMountainCarSolver(n_neurons=n_neurons,
                                        activations=activations)

    scores_DQN = agent.run()

    df['DQN' + str(n_neurons) + str(activations)] = avg_reward(scores_DQN)

    df.to_csv('DQN' + str(n_neurons) + str(activations) + '.csv')

    return df

for i in range(len(n_neurons)):
    df = run_agent_neuron(n_neurons[i], activations[i], df)

df.plot(title='Recompensa promedio por episodio')
plt.ylim(-2000, 0)
plt.show()

