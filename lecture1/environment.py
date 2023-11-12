import gym
import numpy as np

class MatematicasEnv(gym.Env):
    def __init__(self):
        super(MatematicasEnv, self).__init__()
        self.observation_space = gym.spaces.Discrete(4)  # Emociones: feliz, triste, neutro, confundido
        self.action_space = gym.spaces.Discrete(5)  # Acciones: suma, resta, multiplicacion, division, porcentaje
        
        self.current_emotion = None
        self.current_step = 0
        self.max_happiness_category = None  
        self.reset()

    def reset(self):
        self.current_emotion = np.random.choice(self.observation_space.n)  # Emoción aleatoria al inicio
        self.current_step = 0
        self.max_happiness_category = None 
        return self.current_emotion

    def step(self, action):
        categories = {0: 'suma', 1: 'resta', 2: 'multiplicacion', 3: 'division', 4: 'porcentaje'}
        study_category = categories[action]

        is_correct_answer = np.random.choice([True, False])
        reward = self.calculate_reward(study_category, is_correct_answer)
        self.current_emotion = np.random.choice(self.observation_space.n)
        self.current_step += 1
        self.max_happiness_category = self.determine_max_happiness_category(categories)
        print("Emoción:", self.get_emotion_name(), "Recompensa:", reward, "Categoría de mayor felicidad:", self.get_category_name())
        return self.current_emotion, reward, False, {'max_happiness_category': self.max_happiness_category}

    def calculate_reward(self, study_category, is_correct_answer):
        base_reward = 0.5  # Valor base para todas las categorías
        emotion_adjustment = 0
        category_adjustment = 0
         # Ajuste de la recompensa según la emoción del estudiante
        if self.current_emotion == 0:  # Si el estudiante está feliz
            emotion_adjustment = 0.2
        elif self.current_emotion == 1:  # Si el estudiante está neutro
            emotion_adjustment = 0.1
        elif self.current_emotion == 2:  # Si el estudiante está confundido
            emotion_adjustment = -0.1
        elif self.current_emotion == 3:  # Si el estudiante está triste
            emotion_adjustment = -0.2

        # Ajuste de la recompensa según la categoría de la pregunta
        if study_category == "suma":  # Si la categoría es suma
            category_adjustment = 0.1
        elif study_category == "resta":  # Si la categoría es resta
            category_adjustment = 0.1
        elif study_category == "multiplicacion":  # Si la categoría es multiplicación
            category_adjustment = 0.1
        elif study_category == "division":  # Si la categoría es división
            category_adjustment = 0.1
        elif study_category == "porcentaje":  # Si la categoría es porcentaje
            category_adjustment = 0.1

        # Ajuste de la recompensa según la respuesta correcta o incorrecta
        if is_correct_answer:  # Si la respuesta es correcta
            answer_adjustment = emotion_adjustment + category_adjustment
        else:  # Si la respuesta es incorrecta
            answer_adjustment = - (emotion_adjustment + category_adjustment)

        # Cálculo de la recompensa final
        return base_reward + answer_adjustment
            
    def determine_max_happiness_category(self, categories):
        rewards = {}
        # Iterar sobre las categorías y calcular la recompensa para cada una
        for category in categories:
            is_correct_answer_Other = np.random.choice([True, False])
            reward = self.calculate_reward(category, is_correct_answer_Other)
            # Guardar la recompensa en el diccionario con la categoría como clave
            rewards[category] = reward

        # Encontrar la categoría con la mayor recompensa
        max_category = max(rewards, key=rewards.get)

        # Devolver la categoría con la mayor recompensa
        return max_category

    def get_emotion_name(self):
        emotion_names = ['feliz', 'triste', 'neutro', 'confundido']
        return emotion_names[self.current_emotion]

    def get_category_name(self):
        category_names = ['suma', 'resta', 'multiplicacion', 'division', 'porcentaje']
        return category_names[self.max_happiness_category]