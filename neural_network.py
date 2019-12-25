import sys
#Insert the path to the game
sys.path.insert(0, 'path_to_the_game')

#Uses CPU to process the network instead of GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


from controller import Controller
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation


class Neural_controller(Controller):
    def __init__(self, hidden_nodes):
        self.hidden_nodes = hidden_nodes
        self.model = Sequential()
        self.define_model(self.model, self.hidden_nodes)
    
    #Return the weights separeted by layer
    def get_weights(self):
        return (self.model.layers[0].get_weights()[0], self.model.layers[0].get_weights()[1], 
                self.model.layers[2].get_weights()[0], self.model.layers[2].get_weights()[1])
    
    #Initialize the network
    def define_model(self, model, hidden_nodes):
        self.model.add(Dense(units = self.hidden_nodes, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', input_shape = [20]))
        self.model.add(Activation('sigmoid'))
        
        self.model.add(Dense(units = 6, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform' ))
        self.model.add(Activation('sigmoid'))
    
    #Predict an output given the input
    def prediction(self, input_nodes):
        return self.model.predict(np.array([input_nodes]))[0]

    #Restart the weights with random values
    def reset_weights(self):
        (old_weights_first_layer, old_weights_first_layer_bias,
         old_weights_second_layer, old_weights_second_layer_bias) = self.get_weights()

        #Reseting the weights with random values
        new_weights_first_layer = [np.random.uniform(-1,1, [len(old_weights_first_layer), len(old_weights_first_layer[0])]),
					np.random.uniform(-1,1, [1, len(old_weights_first_layer_bias)])[0]]				
        
        new_weights_second_layer = [np.random.uniform(-1,1, [len(old_weights_second_layer), len(old_weights_second_layer[0])]),
					np.random.uniform(-1,1, [1, len(old_weights_second_layer_bias)])[0]]
    
        self.setting_weights(new_weights_first_layer, new_weights_second_layer)
    
    #Set the neural network weights
    def setting_weights(self, weights_first_layer, weights_second_layer):
        self.model.layers[0].set_weights(weights_first_layer) 
        self.model.layers[2].set_weights(weights_second_layer) 


    def control(self, input_nodes, controller):
        #Normalize inputs
        input_nodes = (input_nodes - min (input_nodes)) / float((max(input_nodes) -min(input_nodes)))

        actions_array = self.prediction(input_nodes)
        decision_list = [0] * len(actions_array)

        for i in range(len(actions_array)):
            if actions_array[i] > 0.50:
                decision_list[i] = 1
                
        return decision_list
