import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes #56
        self.hidden_nodes = hidden_nodes #2
        self.output_nodes = output_nodes #1

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes)) #(56,2)

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes)) #(2,1)
        self.lr = learning_rate
        
        
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))  
        
        
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer (128,56) (56,2) - > (128,2)
        
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer (128,2)

        
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer (128,2) (2,1) -> (128,1)
        
        final_outputs = final_inputs # signals from final output layer (128,1)
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        
        ### Backward pass ###

        
        error = y - final_outputs 
        
        output_error_term = error #(128,1)
        
        
        hidden_error = np.dot(error, self.weights_hidden_to_output.T) # (128,1) (1,2) -> (128,2)
        
        
        
        hidden_error_term = hidden_error * (hidden_outputs*(1-hidden_outputs)) # (128 , 2)
        
        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:, None]  #(56,128) * (128,2) - > (56,2)
        # Weight step (hidden to output)
        delta_weights_h_o += output_error_term * hidden_outputs[:, None] #(2,128) * (128,1) - > (2,1)
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += (1/n_records) * self.lr * delta_weights_h_o  # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += (1/n_records) * self.lr * delta_weights_i_h # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer 
        
        return final_outputs


#########################################################
# hyperparameters
##########################################################
iterations = 3000  
learning_rate = 1  
hidden_nodes = 11 
output_nodes = 1 
