import numpy as np
import gym
import gym_environments
from agent import QLearning
from environment2 import EducationEnv
from environment import MatematicasEnv

# Crear una instancia del entorno
env = MatematicasEnv()

# Ejemplo de un ciclo de interacción con el entorno
for _ in range(10):
    action = np.random.randint(4)  # Seleccionar una acción entre 0 y 4
    obs, reward, done, info = env.step(action)

# Reiniciar el entorno
env.reset()


