#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import FrozenLakeAgent as fP
import itertools
import pandas as pd

# Función para definir tablas para métricas
def crear_tablas():
    # Creamos tabla para ir guardando diferentes configuraciones
    resultados = pd.DataFrame(columns=['nombre',
                                       'policy',
                                       'algorithm',
                                       'alpha',
                                       'gamma',
                                       'epsilon o tau',
                                       'is_slippery',
                                       'cutoff_time',
                                       'penalize',
                                       'max_episode_steps']).set_index('nombre')
    # Tablas para guardar los resultados
    episode_rewards = pd.DataFrame(columns=[])
    reward_per_episode = pd.DataFrame(columns=[])
    episode_steps = pd.DataFrame(columns=[])
    steps_per_episode = pd.DataFrame(columns=[])

    return resultados, episode_rewards, reward_per_episode, episode_steps, steps_per_episode

# Función para setear agente
def set_agente(agent, params):
    agent.set_hyper_parameters(params)

    # establece el tiempo de
    agent.set_cutoff_time(params['cutoff_time'])

    # reinicializa el conocimiento del agente
    agent.restart_agent_learning()

    return agent


# Función para definir nombre a partir de los parametros
def nombre_ejecucion(params):
    if params['policy'] == 'egreedy':
        epsilon_tau = params['epsilon']
    elif params['policy'] == 'softmax':
        epsilon_tau = params['tau']
    nombre = params['policy'] + '-' + params['algorithm'] + '-' + \
             str(params['alpha']) + '-' + str(params['gamma']) + '-' + \
             str(epsilon_tau) + '-' + str(params['is_slippery']) + '-' + \
             str(params['cutoff_time']) + '-' + str(params['penalize']) + \
             '-' + str(params['max_episode_steps'])
    return nombre


# Función para cargar métricas
def cargar_metricas(agent, params):
    if params['policy'] == 'egreedy':
        epsilon_tau = params['epsilon']
    elif params['policy'] == 'softmax':
        epsilon_tau = params['tau']
    nombre = nombre_ejecucion(params)
    resultados.loc[nombre] = {'policy' : params['policy'],
                       'algorithm': params['algorithm'],
                       'alpha' : params['alpha'],
                       'gamma' : params['gamma'],
                       'epsilon o tau' : epsilon_tau,
                       'is_slippery' : params['is_slippery'],
                       'cutoff_time' : params['cutoff_time'],
                       'penalize' : params['penalize'],
                       'max_episode_steps' : params['max_episode_steps']}

    episode_rewards[nombre] = agent.reward_of_episode

    episode_number = np.linspace(1, len(episode_rewards[nombre]) + 1, len(episode_rewards[nombre]) + 1)
    acumulated_rewards = np.cumsum(episode_rewards[nombre])
    reward_per_episode[nombre] = [acumulated_rewards[i] / episode_number[i] for i in range(len(acumulated_rewards))]

    episode_steps[nombre] = agent.timesteps_of_episode

    episode_number = np.linspace(1, len(episode_steps[nombre]) + 1, len(episode_steps[nombre]) + 1)
    acumulated_steps = np.cumsum(episode_steps[nombre])
    steps_per_episode[nombre] = [acumulated_steps[i] / episode_number[i] for i in range(len(acumulated_steps))]

# Definimos hiper-parametros a probar
hiper_param = {}
hiper_param['policy'] = ['egreedy']
hiper_param['algorithm'] = ['qlearning']
hiper_param['alpha'] = [0.5]
hiper_param['gamma'] = [0.7]
hiper_param['epsilon'] = [0.1]
hiper_param['tau'] = [0.1]
hiper_param['cutoff_time'] = [100]
hiper_param['is_slippery'] = [False]
hiper_param['penalize'] = [0]
hiper_param['max_episode_steps'] = [500, 100, 50, 25, 10]

resultados, episode_rewards, reward_per_episode, episode_steps, steps_per_episode = crear_tablas()
for is_slippery in hiper_param['is_slippery']:
    # se declara una semilla aleatoria
    random_state = np.random.RandomState(47)
    agent = fP.FrozenLakeAgent()
    agent.random_state = random_state
    # declaramos como True la variable de mostrar video, para ver en tiempo real cómo aprende el agente. Borrar esta línea
    # para acelerar la velocidad del aprendizaje
    agent.display_video = True
    # inicializa el agente
    agent.init_agent(is_slippery=is_slippery)  # slippery es establecido en False por defecto
    for max_episode_steps in hiper_param['max_episode_steps']:
        for penalize in hiper_param['penalize']:
            for cutoff_time in hiper_param['cutoff_time']:
                for algorithm in hiper_param['algorithm']:
                    for policy in hiper_param['policy']:
                        for alpha in hiper_param['alpha']:
                            for gamma in hiper_param['gamma']:
                                if policy == 'egreedy':
                                    for epsilon in hiper_param['epsilon']:
                                        params = {'policy' : policy,
                                                'alpha' : alpha,
                                                'gamma' : gamma,
                                                'epsilon' : epsilon,
                                                'cutoff_time': cutoff_time,
                                                'is_slippery' : is_slippery,
                                                'algorithm' : algorithm,
                                                'penalize' : penalize,
                                                'max_episode_steps' : max_episode_steps}
                                        # Iniciamos el agente
                                        agent = set_agente(agent, params)
                                
                                        # se realiza la ejecución del agente
                                        avg_steps_per_episode = agent.run()
                                        print(avg_steps_per_episode)
                                
                                        # Se cargan las métricas
                                        cargar_metricas(agent, params)
                                        # se procede con los cálculos previos a la graficación de la matriz de valor
                                        value_matrix = np.zeros((4, 4))
                                        for row in range(4):
                                            for column in range(4):
                                
                                                state_values = []
                                
                                                for action in range(4):
                                                    state_values.append(agent.q.get((row * 4 + column, action), 0))

                                                maximum_value = max(state_values)  # como usamos epsilon-greedy, determinamos la acción que arroja máximo valor
                                                state_values.remove(maximum_value)  # removemos el ítem asociado con la acción de máximo valor
                                
                                                # el valor de la matriz para la mejor acción es el máximo valor por la probabilidad de que el mismo sea elegido
                                                # (que es 1-epsilon por la probabilidad de explotación más 1/4 * epsilon por probabilidad de que sea elegido al
                                                # azar cuando se opta por una acción exploratoria)
                                                value_matrix[row, column] = maximum_value * (1 - epsilon + epsilon/4)
                                
                                                for non_maximum_value in state_values:
                                                    value_matrix[row, column] += epsilon/4 * non_maximum_value
                                
                                        # el valor del estado objetivo se asigna en 1 (reward recibido al llegar) para que se coloree de forma apropiada
                                        value_matrix[3, 3] = 1
                                        value_matrix[1, 3] = penalize
                                        value_matrix[2, 3] = penalize
                                        value_matrix[1, 1] = penalize
                                        value_matrix[3, 0] = penalize
                                
                                        # se grafica la matriz de valor
                                        plt.imshow(value_matrix, cmap=plt.cm.RdYlGn)
                                        plt.tight_layout()
                                        plt.colorbar()
                                
                                        fmt = '.2f'
                                        thresh = value_matrix.max() / 2.
                                
                                        for row, column in itertools.product(range(value_matrix.shape[0]), range(value_matrix.shape[1])):
                                
                                            left_action = agent.q.get((row * 4 + column, 0), 0)
                                            down_action = agent.q.get((row * 4 + column, 1), 0)
                                            right_action = agent.q.get((row * 4 + column, 2), 0)
                                            up_action = agent.q.get((row * 4 + column, 3), 0)
                                
                                            arrow_direction = '↓'
                                            best_action = down_action
                                
                                            if best_action < right_action:
                                                arrow_direction = '→'
                                                best_action = right_action
                                            if best_action < left_action:
                                                arrow_direction = '←'
                                                best_action = left_action
                                            if best_action < up_action:
                                                arrow_direction = '↑'
                                                best_action = up_action
                                            if best_action == 0:
                                                arrow_direction = ''
                                
                                            #notar que column, row están invertidos en orden en la línea de abajo
                                            #porque representan a x,y del plot
                                            plt.text(column, row, arrow_direction, horizontalalignment="center")
                                
                                        plt.xticks([])
                                        plt.yticks([])
                                        plt.title('Matriz de valores para max_steps: ' + str(max_episode_steps))
                                        plt.show()
                                
                                        print('\n Matriz de valor (en números): \n\n', value_matrix)
                                if policy == 'softmax':
                                    for tau in hiper_param['tau']:
                                        params = {'policy': policy,
                                                'alpha': alpha,
                                                'gamma': gamma,
                                                'tau': tau,
                                                'cutoff_time': cutoff_time,
                                                'is_slippery': is_slippery,
                                                'algorithm': algorithm}
                                        # Iniciamos el agente
                                        agent = set_agente(agent, params)

                                        # se realiza la ejecución del agente
                                        avg_steps_per_episode = agent.run()

                                        # Se cargan las métricas
                                        cargar_metricas(agent, params)

episode_rewards.plot(title='Recompensa por episodio')
plt.show()

reward_per_episode.plot(title='Recompensa acumulada por episodio')
plt.show()

episode_steps.plot(title='Pasos (timesteps) por episodio')
plt.show()

steps_per_episode.plot(title='Pasos (timesteps) acumulados por episodio')
plt.show()


agent.destroy_agent()

# #
# # se muestra la curva de convergencia de las recompensas
# episode_rewards = np.array(agent.reward_of_episode)
# plt.scatter(np.array(range(0, len(episode_rewards))), episode_rewards, s=0.7)
# plt.title('Recompensa por episodio')
# plt.show()

# # se suaviza la curva de convergencia
# episode_number = np.linspace(1, len(episode_rewards) + 1, len(episode_rewards) + 1)
# acumulated_rewards = np.cumsum(episode_rewards)

# reward_per_episode = [acumulated_rewards[i] / episode_number[i] for i in range(len(acumulated_rewards))]

# plt.plot(reward_per_episode)
# plt.title('Recompensa acumulada por episodio')
# plt.show()

# # ---

# # se muestra la curva de aprendizaje de los pasos por episodio
# episode_steps = np.array(agent.timesteps_of_episode)
# plt.plot(np.array(range(0, len(episode_steps))), episode_steps)
# plt.title('Pasos (timesteps) por episodio')
# plt.show()

# # se suaviza la curva de aprendizaje
# episode_number = np.linspace(1, len(episode_steps) + 1, len(episode_steps) + 1)
# acumulated_steps = np.cumsum(episode_steps)

# steps_per_episode = [acumulated_steps[i] / episode_number[i] for i in range(len(acumulated_steps))]

# plt.plot(steps_per_episode)
# plt.title('Pasos (timesteps) acumulados por episodio')
# plt.show()

# # ---

# # se procede con los cálculos previos a la graficación de la matriz de valor
# value_matrix = np.zeros((4, 4))
# for row in range(4):
#     for column in range(4):

#         state_values = []

#         for action in range(4):
#             state_values.append(agent.q.get((row * 4 + column, action), 0))

#         maximum_value = max(state_values)  # como usamos epsilon-greedy, determinamos la acción que arroja máximo valor
#         state_values.remove(maximum_value)  # removemos el ítem asociado con la acción de máximo valor

#         # el valor de la matriz para la mejor acción es el máximo valor por la probabilidad de que el mismo sea elegido
#         # (que es 1-epsilon por la probabilidad de explotación más 1/4 * epsilon por probabilidad de que sea elegido al
#         # azar cuando se opta por una acción exploratoria)
#         value_matrix[row, column] = maximum_value * (1 - epsilon + 1/4 * epsilon)

#         for non_maximum_value in state_values:
#             value_matrix[row, column] += epsilon/4 * non_maximum_value

# # el valor del estado objetivo se asigna en 1 (reward recibido al llegar) para que se coloree de forma apropiada
# value_matrix[3, 3] = 1

# # se grafica la matriz de valor
# plt.imshow(value_matrix, cmap=plt.cm.RdYlGn)
# plt.tight_layout()
# plt.colorbar()

# fmt = '.2f'
# thresh = value_matrix.max() / 2.

# for row, column in itertools.product(range(value_matrix.shape[0]), range(value_matrix.shape[1])):

#     left_action = agent.q.get((row * 4 + column, 0), 0)
#     down_action = agent.q.get((row * 4 + column, 1), 0)
#     right_action = agent.q.get((row * 4 + column, 2), 0)
#     up_action = agent.q.get((row * 4 + column, 3), 0)

#     arrow_direction = '↓'
#     best_action = down_action

#     if best_action < right_action:
#         arrow_direction = '→'
#         best_action = right_action
#     if best_action < left_action:
#         arrow_direction = '←'
#         best_action = left_action
#     if best_action < up_action:
#         arrow_direction = '↑'
#         best_action = up_action
#     if best_action == 0:
#         arrow_direction = ''

#     #notar que column, row están invertidos en orden en la línea de abajo porque representan a x,y del plot
#     plt.text(column, row, arrow_direction, horizontalalignment="center")

# plt.xticks([])
# plt.yticks([])
# plt.show()

# print('\n Matriz de valor (en números): \n\n', value_matrix)


