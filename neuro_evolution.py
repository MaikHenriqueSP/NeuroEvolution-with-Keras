import sys
#Insert the path of the game
sys.path.insert(0, 'path_to_the_game')

from environment import Environment
from neural_network import Neural_controller
import numpy as np
import winsound
import os
import matplotlib.pyplot as plt
import pickle

experiment_name = 'project_individual_final_version'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
    
class Neuro_evolution():
    def __init__(self, hidden_nodes, population_size, generations, mutation_rate):
        self.neural_network = Neural_controller(hidden_nodes)

        self.environment = Environment(experiment_name = experiment_name,
                          enemies = [2, 5],
                          level = 2,
                          contacthurt = "player",
                          playermode = "ai",
                          timeexpire = 500,
                          multiplemode = "yes",
                          player_controller = self.neural_network,
                          enemymode = "static",
                          speed = "fastest")
        
        self.environment.state_to_log()
        
        self.generation_counter = 0
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        self.population = []
        self.population_fitness = []
        self.solutions = [self.population, self.population_fitness, hidden_nodes]
        
        self.plot_x_generations = []
        self.plot_y_highest_fitness = []
        self.plot_y_average_fitness = []
        
    #Create N random networks
    def initialize_population(self, total):
        for i in range(total):
            weights = self.neural_network.get_weights()                
            fitness, player_life, enemy_life, game_run_time = self.environment.play(pcont = self.neural_network)

            self.population_fitness.append(fitness)     
            self.population.append(weights)
           
            self.neural_network.reset_weights()
    
    #To be implemented, Roullete wheel selection of the parents
    #Current just take the fittest individual and a random one
    def crossover(self):
        parent1_index = np.random.randint(0, len(self.population))
        parent2_index = np.random.randint(0, len(self.population))
        child_weights = []
        
        while parent1_index == parent2_index:
            parent2_index = np.random.randint(0, len(self.population))
        
        child_weights.append(self.generate_child(parent1_index, parent2_index, 0))
        child_weights.append(self.generate_child(parent1_index, parent2_index, 1))
        child_weights.append(self.generate_child(parent1_index, parent2_index, 2))
        child_weights.append(self.generate_child(parent1_index, parent2_index, 3))
        
        self.mutation(child_weights)
        
        self.evaluate_net([child_weights[0], child_weights[1]], [child_weights[2], child_weights[3]])   
    
    #Method to generate child weights
    def generate_child(self, parent1_index, parent2_index, index_weight):
        midpoint = np.random.randint(0, len(self.population[parent1_index][index_weight]))
       
        child_weights = np.concatenate([(self.population[parent1_index][index_weight][:midpoint]),
                                                    (self.population[parent2_index][index_weight][midpoint:])])
        return child_weights
    
    #Evaluate agents after crossover
    def evaluate_net(self, weights_first_layer, weights_second_layer):
        self.neural_network.setting_weights(weights_first_layer, weights_second_layer)
        fitness, player_life, enemy_life, game_run_time = self.environment.play(pcont = self.neural_network)
        
        self.population.append(self.neural_network.get_weights())
        self.population_fitness.append(fitness)
            
    def mutation(self, neural_weights):
        self.mutate_layer(neural_weights, 0)
        self.mutate_layer(neural_weights, 2)
        self.mutate_bias(neural_weights, 1)
        self.mutate_bias(neural_weights, 3)

    #Support Functions
    def mutate_layer(self, neural_weights, index):
        for i in range(len(neural_weights[index])):
            for j in range(len(neural_weights[index][0])):
                if np.random.random() <= self.mutation_rate:
                      sup_var = np.random.uniform(-0.1, 0.1)
                      if (neural_weights[index][i][j] + sup_var) <= 1 and (neural_weights[index][i][j] + sup_var) >= -1: 
                          neural_weights[index][i][j] = neural_weights[index][i][j] + sup_var

    def mutate_bias(self, neural_weights, index):
        for i in range(len(neural_weights[index])):
            if np.random.random() <= self.mutation_rate:
                sup_var = np.random.uniform(-0.3, 0.3)
                if (neural_weights[index][i] + sup_var) <= 1 and (neural_weights[index][i] + sup_var) >= -1:
                    neural_weights[index][i] = neural_weights[index][i] + sup_var
            
    #Survival Tournament
    def selection(self):
        contender1_index = np.random.randint(0, len(self.population))
        contender2_index = np.random.randint(0, len(self.population))
        loser_index = contender1_index
        
        while contender1_index == contender2_index:
            contender2_index = np.random.randint(0, len(self.population))    
        
        if self.population_fitness[contender1_index] > self.population_fitness[contender2_index]:
            loser_index = contender2_index

        self.population.pop(loser_index)
        self.population_fitness.pop(loser_index)
        
            
    #Control evolution
    def evolve_network(self):
        self.initialize_population(self.population_size)
        
        for i in range(self.generations):
            #Crossover
            for j in range(int(self.population_size * 0.75)):
                self.crossover()
            
            #Create 10 % of random agents
            self.initialize_population(int(self.population_size * 0.10) + 1)
            
            #Selection
            while len((self.population)) != self.population_size:
                self.selection()
            
            self.generation_counter += 1
            
            print("================================ CURRENT GENERATION: {} ================================".format(self.generation_counter))
            print("HIGHEST FITNESS: {}".format(max(self.population_fitness)))           

            self.plot_x_generations.append(self.generation_counter)
            self.plot_y_highest_fitness.append(max(self.population_fitness))
            self.plot_y_average_fitness.append(np.mean(self.population_fitness))
            
        #Beep to notify end of the training
        winsound.Beep(2400, 1500)
        self.plot_graphics()
        self.save()
        self.fittest_agent()
        
    def plot_graphics(self):
        plt.plot(self.plot_x_generations, self.plot_y_highest_fitness, label = "Highest fitness")
        plt.plot(self.plot_x_generations, self.plot_y_average_fitness, label = "Average Fitness")
        plt.xlabel("Gerações")
        plt.ylabel("Fitness")
        plt.title("Gerações x Fitness")
        plt.legend()
        plt.show()
        
                
    #Testing the fittest agent against all enemies
    def fittest_agent(self):
        self.environment.update_parameter('speed','normal')
        self.environment.update_parameter('multiplemode','yes')
        self.environment.update_parameter('enemies', [1, 2, 3, 4, 5, 6, 7, 8])
        
        print("FITTEST PLAYER AGAISNT ALL ENEMIES")
        
        index_fittest_player = self.population_fitness.index(max(self.population_fitness))        
        weight_first_layer = self.population[index_fittest_player][0], self.population[index_fittest_player][1]
        weight_second_layer = self.population[index_fittest_player][2], self.population[index_fittest_player][3]
        
        self.neural_network.setting_weights(weight_first_layer, weight_second_layer)        
        self.environment.play(pcont = self.neural_network)
        
    #Save results in pickle file
    def save(self): 
        outfile = open("saved_population","wb")
        pickle.dump(self.solutions, outfile)
        outfile.close()
        
        
nn = Neuro_evolution(17, 50, 15, 0.20 )
#Initialize simulation
nn.evolve_network()
