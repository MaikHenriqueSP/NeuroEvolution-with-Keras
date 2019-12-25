import pickle
import sys
sys.path.insert(0,'path_to_the_game')
from environment import Environment
from neural_network import Neural_controller

infile = open("saved_population", "rb")
solution = pickle.load(open("saved_population", "rb"))
infile.close()

index_fittest_player = solution[1].index(max(solution[1]))

first_layer = [solution[0][index_fittest_player][0] , solution[0][index_fittest_player][1]]
second_layer = [solution[0][index_fittest_player][2] , solution[0][index_fittest_player][3]]

neural_network = Neural_controller(solution[2])
neural_network.setting_weights(first_layer, second_layer)

environment = Environment(
            enemies = [1, 2, 3, 4, 5, 6, 7, 8],
            level = 2,
            contacthurt = "player",
            playermode = "ai",
            multiplemode = "yes",
            player_controller = neural_network,
            enemymode = "static",
            speed = "normal")

fitness, player_life, enemy_life, game_run_time = environment.play(pcont = neural_network)
print("FITNESS {}".format(fitness))
