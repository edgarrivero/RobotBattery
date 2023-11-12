import gym
import numpy as np

class MatematicasEnv(gym.Env):
    def __init__(self):
        super(MatematicasEnv, self).__init__()
        self.observation_space = gym.spaces.Discrete(4)  # Emociones: feliz, triste, neutro, asustado
        self.action_space = gym.spaces.Discrete(5)  # Acciones: suma, resta, multiplicacion, division, porcentaje
        #Inicializar la emoción actual y el contador de pasos
        self.current_emotion = None
        self.current_step = 0
        self.max_happiness_category = None
        #Llamar al método de reinicio para inicializar el entorno
        self.reset()

    def reset(self):
        self.current_emotion = np.random.choice(self.observation_space.n)  # Emoción aleatoria al inicio
        self.current_step = 0
        self.max_happiness_category = None
        return self.current_emotion

    def step(self, action):
        # Tomar una acción y obtener la recompensa

        # Mapeo de acciones a categorías de estudio
        categories = {0: 'suma', 1: 'resta', 2: 'multiplicacion', 3: 'division', 4: 'porcentaje'}
        study_category = categories[action]

        # Lógica para determinar la recompensa basada en la emoción y la categoría de estudio
        reward = self.calculate_reward(study_category)

        # Actualizar el estado (emoción) de manera aleatoria para el siguiente paso
        self.current_emotion = np.random.choice(self.observation_space.n)

        # Incrementar el contador de pasos
        self.current_step += 1

        # Determinar la categoría que produce la mayor felicidad
        self.max_happiness_category = self.determine_max_happiness_category(categories)
        print("Emoción:", self.get_emotion_name(), "Recompensa:", reward, "Categoría de mayor felicidad:", self.get_category_name())

        # Devolver observación (emoción), recompensa, indicador de finalización y categoría de mayor felicidad
        return self.current_emotion, reward, False, {'max_happiness_category': self.max_happiness_category}


    def calculate_reward(self, study_category):
        if self.current_emotion == 0:  # Si el estudiante está feliz
            if study_category == 'suma':
                return 1.0  # Recompensa positiva
            else:
                return -0.5  # Recompensa negativa para otras categorías
        elif self.current_emotion == 1:  # Si el estudiante está triste
            if study_category == 'resta':
                return -1.0  # Recompensa muy mala
            else:
                return -0.5  # Recompensa negativa para otras categorías
        elif self.current_emotion == 2:  # Si el estudiante está neutro
            if study_category == 'multiplicacion':
                return 0.5  # Recompensa positiva para la multiplicación
            else:
                return 0.0  # Recompensa neutra para otras categorías
        elif self.current_emotion == 3:  # Si el estudiante está confundido
            if study_category == 'division':
                return -0.2  # Recompensa ligeramente negativa para la división
            else:
                return -0.1  # Recompensa negativa para otras categorías
        elif self.current_emotion == 3:  # Si el estudiante está confundido
            if study_category == 'porcentaje':
                return -0.2  # Recompensa ligeramente negativa para la división
            else:
                return -0.1  # Recompensa negativa para otras categorías
            
    
    def determine_max_happiness_category(self,categories):
        happiness_scores = {}
        for category in range(self.action_space.n):
            happiness_scores[category] = self.calculate_reward(categories[category])

        max_happiness_category = max(happiness_scores, key=happiness_scores.get)
        return max_happiness_category
    
    def get_emotion_name(self):
        emotion_names = ['feliz', 'triste', 'neutro', 'confundido']
        return emotion_names[self.current_emotion]
    
    def get_category_name(self):
        category_names = ['suma', 'resta', 'multiplicacion', 'division', 'porcentaje']
        return category_names[self.max_happiness_category]