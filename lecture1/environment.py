import gym
import numpy as np

class MatematicasEnv(gym.Env):
    def __init__(self):
        super(MatematicasEnv, self).__init__()
        self.observation_space = gym.spaces.Discrete(4)  
        self.action_space = gym.spaces.Discrete(5)          
        self.current_emotion = None
        self.current_step = 0
        self.max_happiness_category = None  
        self.reset()

    def reset(self):
        self.current_emotion = np.random.choice(self.observation_space.n)  
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

        #Condicionales con las emociones y la accion(categoria) 
        base = 0.5  
        emotion_adjustment = 0
        category_adjustment = 0
        
        if self.current_emotion == 0:  
            emotion_adjustment = 0.2
        elif self.current_emotion == 1:  
            emotion_adjustment = 0.1
        elif self.current_emotion == 2:  
            emotion_adjustment = -0.1
        elif self.current_emotion == 3:  
            emotion_adjustment = -0.2

        
        if study_category == "suma":  
            category_adjustment = 0.1
        elif study_category == "resta":  
            category_adjustment = 0.1
        elif study_category == "multiplicacion":  
            category_adjustment = 0.1
        elif study_category == "division":  
            category_adjustment = 0.1
        elif study_category == "porcentaje":  
            category_adjustment = 0.1

        if is_correct_answer:  
            answer_adjustment = emotion_adjustment + category_adjustment
        else:  
            answer_adjustment = - (emotion_adjustment + category_adjustment)

        result = base + answer_adjustment

        #probabilidad de correcto


        #siguiente emocion

        print("Emoción:", self.get_emotion_name(), "Recompensa:", reward, "Categoría de mayor felicidad:", self.get_category_name())
        return self.current_emotion, reward, False, {'max_happiness_category': self.max_happiness_category}

    def calculate_reward(self):
        #si responde correcto es 1
        #caso contrario 0
        base_reward = 0.5  
        
        if self.current_emotion == 1:  
            base_reward = 1
        else :
            base_reward = 0
        
        return base_reward
            
    def determine_max_happiness_category(self, categories):
        rewards = {}
        
        for category in categories:
            is_correct_answer_Other = np.random.choice([True, False])
            reward = self.calculate_reward(category, is_correct_answer_Other)
            rewards[category] = reward

        
        max_category = max(rewards, key=rewards.get)
        return max_category

    def get_emotion_name(self):
        emotion_names = ['feliz', 'triste', 'neutro', 'confundido']
        return emotion_names[self.current_emotion]

    def get_category_name(self):
        category_names = ['suma', 'resta', 'multiplicacion', 'division', 'porcentaje']
        return category_names[self.max_happiness_category]