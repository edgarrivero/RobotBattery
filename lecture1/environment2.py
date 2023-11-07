import gym
import numpy as np

class EducationEnv(gym.Env):
    def __init__(self):
        super(EducationEnv, self).__init__()

        # Definir el espacio de observación (características de emoción y conocimiento)
        self.observation_space = gym.spaces.Dict({
            'emotion': gym.spaces.Box(low=0, high=1, shape=(4,)),
            'knowledge': gym.spaces.Box(low=0, high=1, shape=(3,))
        })

        # Definir el espacio de acción (categorías emocionales y respuestas a preguntas)
        self.action_space = gym.spaces.Dict({
            'emotional_category': gym.spaces.Discrete(5),  # Ejemplo con 5 categorías emocionales
            'answer': gym.spaces.Discrete(2)  # Respuesta a una pregunta (0 para incorrecto, 1 para correcto)
        })

        # Emoción y conocimiento iniciales (pueden ser aleatorios)
        self.current_emotion = np.random.rand(4)
        self.current_knowledge = np.random.rand(3)

        # Mapeo de categorías emocionales a nombres
        self.emotional_category_mapping = {
            0: "positiva",
            1: "negativa",
            2: "neutra",
            3: "feliz",
            4: "triste"
        }

    def step(self, action):
        # Asegurarse de que las acciones generadas estén dentro del rango de categorías
        if action['emotional_category'] < 0 or action['emotional_category'] >= self.action_space['emotional_category'].n:
            raise ValueError("Categoría emocional fuera de rango")
        if action['answer'] < 0 or action['answer'] >= self.action_space['answer'].n:
            raise ValueError("Respuesta fuera de rango")

        # Obtener el nombre de la categoría emocional
        emotional_category = self.emotional_category_mapping[action['emotional_category']]

        # Simular la influencia de la acción en las emociones y el conocimiento
        self.current_emotion += self.influence_emotion(emotional_category)
        self.current_knowledge += self.influence_knowledge(action['answer'])

        # Calcular la recompensa en función de las emociones y el conocimiento
        reward = self.calculate_reward()

        # Definir si el episodio termina (puede ser un límite de tiempo o un número fijo de pasos)
        done = False

        return {
            'emotion': self.current_emotion,
            'knowledge': self.current_knowledge
        }, reward, done, {}

    def reset(self):
        # Reiniciar el entorno con emociones y conocimiento iniciales aleatorios
        self.current_emotion = np.random.rand(4)
        self.current_knowledge = np.random.rand(3)
        return {
            'emotion': self.current_emotion,
            'knowledge': self.current_knowledge
        }

    def render(self, mode='human'):
        # Calcular las categorías emocionales que le gustan más al estudiante en función de las recompensas
        best_emotional_categories = self.find_best_emotional_categories()

        print(f"Emociones actuales: {self.current_emotion}")
        print(f"Conocimiento actual: {self.current_knowledge}")
        print(f"Categorías emocionales preferidas: {', '.join(best_emotional_categories)}")

    def influence_emotion(self, emotional_category):
        # Simular cómo las acciones influyen en las emociones del estudiante
        influence = np.zeros(4)
        if emotional_category == 'positiva':
            influence[0] += 0.1
        elif emotional_category == 'negativa':
            influence[1] -= 0.1
        return influence

    def influence_knowledge(self, answer):
        # Simular cómo las respuestas influyen en el conocimiento del estudiante
        influence = np.zeros(3)
        if answer == 1:  # Respuesta correcta
            influence[0] += 0.1
        return influence

    def calculate_reward(self):
        # Calcular la recompensa en función de las emociones y el conocimiento
        reward = np.dot(self.current_emotion, [1, -1, 0, 0]) + np.dot(self.current_knowledge, [0, 0, 1])
        return reward

    def find_best_emotional_categories(self):
        # Calcular las categorías emocionales preferidas en función de las recompensas
        emotional_categories = list(self.emotional_category_mapping.values())
        rewards = [self.calculate_reward() for _ in emotional_categories]
        best_reward = max(rewards)
        best_emotional_categories = [category for category, reward in zip(emotional_categories, rewards) if reward == best_reward]
        return best_emotional_categories