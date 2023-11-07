import gym
import numpy as np

class MatematicasEnv(gym.Env):
    def __init__(self):
        super(MatematicasEnv, self).__init__()

        # Definir el espacio de observación (features de la emoción)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,))

        # Definir el espacio de acción (categorías emocionales)
        self.action_space = gym.spaces.Discrete(5)  # Ejemplo con 5 categorías: suma, resta, multiplicación, división, porcentaje

        # Emoción actual (inicializada de manera aleatoria)
        self.current_emotion = np.random.rand(4)

        # Mapeo de nombres de categorías a vectores de categorías
        self.category_mapping = {
            "suma": [1, 0, 0, 0],
            "resta": [0, 1, 0, 0],
            "multiplicación": [0, 0, 1, 0],
            "división": [0, 0, 0, 1],
            "porcentaje": [0.25, 0.25, 0.25, 0.25]
        }

        # Parámetros BKT
        self.num_skills = 4  # Número de habilidades
        self.p_init = np.random.rand(self.num_skills)  # Probabilidad inicial de conocimiento
        self.p_learn = np.random.rand(self.num_skills)  # Habilidad de retención
        self.p_guess = np.random.rand(self.num_skills)  # Habilidad de adivinar correctamente
        self.p_slip = np.random.rand(self.num_skills)  # Habilidad de cometer un error descuidado

    def step(self, action):
        # Asegurarse de que la acción generada esté dentro del rango de categorías (0 a 4)
        if action < 0 or action >= self.action_space.n:
            raise ValueError("Acción fuera de rango")

        # Obtener el nombre de la categoría
        category = list(self.category_mapping.keys())[action]

        # Simular el proceso de aprendizaje del estudiante con BKT

        # Calcular la probabilidad de éxito en la acción actual
        p_success = self.p_init[action] * (1 - self.p_slip[action]) + (1 - self.p_init[action]) * self.p_guess[action]

        # El estudiante tiene éxito en la acción con probabilidad p_success
        success = np.random.rand() < p_success

        # Actualizar los parámetros BKT basados en el resultado
        if success:
            self.p_init[action] = self.p_init[action] + (1 - self.p_init[action]) * self.p_learn[action]
        else:
            self.p_slip[action] = self.p_slip[action] + (1 - self.p_slip[action]) * self.p_learn[action]

        # Calcular la recompensa
        reward = 1 if success else 0

        # Actualizar el estado de conocimiento
        knowledge_state = np.array([self.p_init, self.p_learn, self.p_guess, self.p_slip])

        # Definir si el episodio termina (puede ser un límite de tiempo o un número fijo de pasos)
        done = False

        return knowledge_state, reward, done, {}

    def reset(self):
        # Reiniciar el entorno con una nueva emoción aleatoria
        self.current_emotion = np.random.rand(4)
        return np.array([self.p_init, self.p_learn, self.p_guess, self.p_slip])

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Emociones actuales: {self.current_emotion}")
            print(f"Conocimiento actual: P_init = {self.p_init}, P_learn = {self.p_learn}, P_guess = {self.p_guess}, P_slip = {self.p_slip}")
        else:
            super(MatematicasEnv, self).render(mode=mode)