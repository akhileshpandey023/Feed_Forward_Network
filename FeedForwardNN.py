from __future__ import division 
import numpy as np
import logging
import os
from utils import setup_logger


class FeedforwardNN(object):

    
    def __init__(self, n_inputs, hidden_sizes, n_output, loss, activation_input):
    
        self.loss = loss
        self.num_hidden = len(hidden_sizes)
        self.n_output = n_output
        
        all_dims = hidden_sizes.copy()
        all_dims.insert(0, n_inputs)
        all_dims.append(n_output)
        
        np.random.seed(1234)
        
        ## Create a dict of activations --> Can add more!
        self.activation = {'tanh': self.tanh, 'sigmoid': self.sigmoid, 'softmax': self.stable_softmax}
        self.grad_activation = {'tanh': self.grad_tanh, 'sigmoid': self.grad_sigmoid, 'softmax': self.grad_softmax}
        
        self.activation_input = activation_input
        
        ## Initialise Objects 
        self.weights = {}
        self.biases = {}
        ## Weight -- (n_out, n_input)
        ## Bias - (n_out, 1)
        for x in range(len(all_dims)-1):
            self.weights[x+1] = np.random.randn(all_dims[x+1], all_dims[x])*np.sqrt(1/all_dims[x+1])
            self.biases[x+1] = np.zeros((all_dims[x+1], 1))

        ## Initialise Preactivation and output of each layer    
        self.A = {}
        self.H = {}

        # Add dimension checks 

        ## Initialise Gradients for preactivation, activation, weights and bias 
        self.grad_A = {}
        self.grad_H = {}
        self.grad_weights = {}
        self.grad_biases = {}
        

        
    def normalise_fit(self, train_data):
        ## Normalise the training data and save the min and (max - min) value for each column/variable 
        self.train_min = train_data.min(0)
        self.train_max_min_diff = train_data.ptp(0)
        
        train_norm = (train_data - self.train_min) / self.train_max_min_diff
        
        return train_norm
    
    def normalise_transform(self, test_data):
        # Normalise the test data according to the values found in the training data to ensure the values are uniformly
        # normalised in both the datasets 
        test_norm = (test_data - self.train_min) / self.train_max_min_diff
        
        return test_norm
    
    # Implement Activation and Gradients 
    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))
    
    def grad_sigmoid(self, a):
        return self.sigmoid(a)*(1 - self.sigmoid(a))

    def tanh(self, a):
        return np.tanh(a)

    def grad_tanh(self, a):
        return 1.0 - np.square(self.tanh(a))
    
    def relu(self, a):
        return np.maximum(a, 0, a)

    def stable_softmax(self, X):
        exps = np.exp(X - np.max(X))
        return exps / np.sum(exps, axis = 0)
    
    def grad_softmax(self, softmax):
        # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
        s = softmax.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)
    
    
    ### Forward Pass
    def forwardPass(self, X_input):
  
        self.H[0] = X_input.T
        output_layer = self.num_hidden+1

        for i in range(0, self.num_hidden):    
            self.A[i+1] = np.dot(self.weights[i+1], self.H[i]) + self.biases[i+1]
            self.H[i+1] = self.activation[self.activation_input](self.A[i+1])
            
        self.A[output_layer] = np.dot(self.weights[output_layer], self.H[output_layer-1]) + self.biases[output_layer]
        
        # Last Layer will always have softmax
        self.H[output_layer] = self.stable_softmax(self.A[output_layer])

        return self.H[output_layer]
    
    
    def delta_cross_entropy(self, X, y):

        y_one_hot = self.one_hot(y, self.n_output)
        grad = X - y_one_hot
        
        return grad.T
    
    def delta_square_error(self, X, y):
        y_one_hot = self.one_hot(y, self.n_output)
        grad = (X- y_one_hot) * X * (1 - X)
        
        return grad.T

    
    ### Backward Pass
    def backwardPass(self, y_input, weights):
        L =  self.num_hidden + 1
        num_samples = y_input.shape[0]
        
        ## Calculate the gardient of loss
        ### Add Square loss!!!!!
        if self.loss == 'ce':
            self.grad_A[L] = self.delta_cross_entropy(self.H[L].T, y_input)
        elif self.loss == 'sq':
            self.grad_A[L] = self.delta_square_error(self.H[L].T, y_input)

        for k in reversed(range(2,L+1)):
            self.grad_weights[k] = np.matmul(self.grad_A[k], self.H[k-1].T)/num_samples
            self.grad_biases[k] = np.sum(self.grad_A[k], keepdims= True, axis=1)/num_samples

            self.grad_H[k-1] = np.matmul(weights[k].T, self.grad_A[k])
            self.grad_A[k-1] = np.multiply(self.grad_H[k-1], self.grad_activation[self.activation_input](self.A[k-1]))

        self.grad_weights[1] = np.matmul(self.grad_A[1], self.H[0].T)/num_samples
        self.grad_biases[1] = np.sum(self.grad_A[1], keepdims= True, axis=1)/num_samples
        
        return self
        
        
    def one_hot(self, a, num_classes):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
    
    def cross_entropy(self, predictions, targets):
        
        predictions = np.where(predictions==0, 0.000001, predictions)
        targets = self.one_hot(targets, self.n_output)
        N = predictions.shape[0]
        ce = -np.sum(targets * np.log(predictions))/N
        
        return ce
    
    def mean_square(self, predictions, targets):
        
        targets_one_hot = self.one_hot(targets, self.n_output)
        
        return np.mean(np.square(targets_one_hot - predictions))
    
    def get_class(self, predictions):
        idx = np.argmax(predictions, axis=-1)
        predictions = np.zeros(predictions.shape)
        predictions[ np.arange(predictions.shape[0]), idx] = 1
        
        return predictions
    
    def get_accuracy(self, class_predictions, class_actual):
        
        class_actual_one_hot = self.one_hot(class_actual, self.n_output)
        
        total_correct = np.sum(np.all(np.equal(class_predictions, class_actual_one_hot), axis=1))
        total_predictions = class_actual_one_hot.shape[0]
        
        return round((100*(total_correct/total_predictions)),2)

    
    def fit(self, X_train, y_train, X_validation, y_validation, batch_size, max_epochs, learning_rate, 
            update_rule = 'gd', gamma = 0.9, beta_1 = 0.9, beta_2 = 0.999, anneal = False):
        
        
        # first file logger
        train_logger = setup_logger('train_logger', os.getcwd() + '/Logs/train.log')
        
        # second file logger
        test_logger = setup_logger('test_logger', os.getcwd() + '/Logs/test.log')
        
        update_weights = {}
        update_bias = {}
        
        prev_total_validation_loss = 1000000000 # Inititalise to large value
        anneal_count = 0
        eps = 0.00000001
        
        ##Add history to save weights!
        if anneal:
            weights_prev_iter = {}
            bias_prev_iter = {}
            for i in range(1, len(self.weights)+1):
                weights_prev_iter[i] =  self.weights[i]
                bias_prev_iter[i] = self.biases[i]
        
    
        if update_rule == 'nag':
            weight_lookahead = {}
            bias_lookahead = {} 
            
            for i in range(1, len(self.weights)+1):
                update_weights[i] =  np.zeros(self.weights[i].shape)
                weight_lookahead[i] = np.zeros(self.weights[i].shape)
                
                update_bias[i] =  np.zeros(self.biases[i].shape)
                bias_lookahead[i] = np.zeros(self.biases[i].shape)
        
        elif update_rule == 'momentum':
            
            for i in range(1, len(self.weights)+1):
                update_weights[i] =  np.zeros(self.weights[i].shape)
                update_bias[i] =  np.zeros(self.biases[i].shape)
            
        elif update_rule  == 'adam':
            m_w = {}
            m_b = {}

            v_w = {}
            v_b = {}

            m_hat_w = {}
            m_hat_b = {}

            v_hat_w = {}
            v_hat_b = {}
            
            for i in range(1, len(self.weights)+1):
                m_w[i] = np.zeros(self.weights[i].shape)
                m_b[i] = np.zeros(self.biases[i].shape)

                v_w[i] = np.zeros(self.weights[i].shape)
                v_b[i] = np.zeros(self.biases[i].shape)

                m_hat_w[i] = np.zeros(self.weights[i].shape)
                m_hat_b[i] = np.zeros(self.biases[i].shape)

                v_hat_w[i] = np.zeros(self.weights[i].shape)
                v_hat_b[i] = np.zeros(self.biases[i].shape)
        
        ## Calculate total number of steps for a given batch_size 
        steps = int(X_train.shape[0]/batch_size)
        epoch = 1
        
        while epoch < max_epochs + 1:      
            step = 1
            total_validation_loss = 0
            while step < steps + 1:
                
                index_start = index_end if step > 1 else 0
                index_end = index_start + batch_size - 1
                last_sample = X_train.shape[0]

                if index_end > last_sample:
                    index_end = last_sample

                # Create Batch
                X_batch = X_train[index_start:index_end,::]
                y_batch = y_train[index_start:index_end,:]
            
                # Forward Pass
                _ = self.forwardPass(X_batch)
       
                # Backward Pass
                if update_rule != 'nag':
                    self.backwardPass(y_batch, self.weights)

                ## To Implement: NAG, Adam
                if update_rule == 'gd':
                    for i in range(1, len(self.weights)+1):
                        update_weights[i] = learning_rate*self.grad_weights[i]
                        update_bias[i] = learning_rate*self.grad_biases[i]
                
                elif update_rule == 'momentum':

                    for i in range(1, len(self.weights)+1):                            
                            update_weights[i] =  gamma*update_weights[i] + learning_rate*self.grad_weights[i]
                            update_bias[i] =  gamma*update_bias[i] + learning_rate*self.grad_biases[i]
                
                
                elif update_rule == 'nag':
                      
                    for i in range(1, len(self.weights)+1):
                        weight_lookahead[i] = self.weights[i] - gamma*update_weights[i]
                        bias_lookahead[i] = bias_lookahead[i] - gamma*update_bias[i]                         
        
                    self.backwardPass(y_batch, weight_lookahead)
                    
                    for i in range(1, len(self.weights)+1):
                        update_weights[i] = gamma*update_weights[i] + learning_rate*self.grad_weights[i]
                        update_bias[i] =  gamma*update_bias[i] + learning_rate*self.grad_biases[i]
                        
                        
                        
                elif update_rule == 'adam':
                    
                    for i in range(1, len(self.weights)+1):
                        m_w[i] = beta_1*m_w[i] + (1-beta_1)*self.grad_weights[i]
                        v_w[i] = beta_2*v_w[i] + (1-beta_2)*np.power(self.grad_weights[i], 2)
                        
                        m_hat_w[i] = m_w[i]/(1- np.power(beta_1,step))
                        v_hat_w[i] = v_w[i]/(1- np.power(beta_2, step))
                        
                        update_weights[i] = (learning_rate/(np.sqrt(v_hat_w[i]) + eps))*m_hat_w[i]
                        
                        m_b[i] = beta_1*m_b[i] + (1-beta_1)*self.grad_biases[i]
                        v_b[i] = beta_2*v_b[i] + (1-beta_2)*np.power(self.grad_biases[i], 2)
                        
                        m_hat_b[i] = m_b[i]/(1-np.power(beta_1, step))
                        v_hat_b[i] = v_b[i]/(1-np.power(beta_2, step))
                        
                        update_bias[i] = (learning_rate/(np.sqrt(v_hat_b[i])+eps))*m_hat_b[i]
                    
                
                ## Update the weights     
                for i in range(1, len(self.weights)+1):                    
                    self.weights[i] = self.weights[i] - update_weights[i]
                    self.biases[i] = self.biases[i] - update_bias[i]
                    
                if step % 100 == 0:
                    
                    output_train = self.forwardPass(X_train)
                    output_validation = self.forwardPass(X_validation)

                    if self.loss == 'ce':
                        train_loss = self.cross_entropy(output_train.T, y_train)
                        validation_loss = self.cross_entropy(output_validation.T, y_validation)

                    elif self.loss == 'sq':
                        train_loss = self.mean_square(output_train.T, y_train)
                        validation_loss = self.mean_square(output_validation.T, y_validation)
                    
                    total_validation_loss += validation_loss

                    class_output_train = self.get_class(output_train.T)
                    class_output_validation = self.get_class(output_validation.T)

                    accuracy_train = self.get_accuracy(class_output_train, y_train)
                    accuracy_validation = self.get_accuracy(class_output_validation, y_validation)
                
                    train_logger.info('Epoch ' + str(epoch) + ', ' + 'Step ' + str(step) + ', ' + 'Train Loss '
                          + str(train_loss)+ ', ' + 'Train Accuracy ' + str(accuracy_train) + ', ' + 'lr '
                          + str(learning_rate))
                    
                    test_logger.info('Epoch ' + str(epoch) + ', ' + 'Step ' + str(step) + ', ' + 'Validation Loss ' 
                          + str(validation_loss)+ ', ' + 'Validation Accuracy ' + str(accuracy_validation) + ', ' + 'lr '
                          + str(learning_rate))
                
                step +=1
        
            ## TO DO: Implement Annealing 
            # Half the learning rate and reset the epoch if the validation loss increases
            if anneal:
                if total_validation_loss > prev_total_validation_loss:
                    anneal_count +=1
                    for i in range(1, len(self.weights)+1):
                        self.weights[i] = weights_prev_iter[i]
                        self.biases[i] = bias_prev_iter[i]
                    learning_rate = learning_rate/2
                    epoch = epoch - 1
                else:
                    for i in range(1, len(self.weights)+1):
                        weights_prev_iter[i] = self.weights[i]
                        bias_prev_iter[i] = self.biases[i]
                
                prev_total_validation_loss = total_validation_loss

                    # Stuck around a local minima: stop iterating
            if anneal_count > 15:
                break

            epoch += 1

        return self
    
    
    def predict(self, test_data):
        
        predicted_score = self.forwardPass(test_data.T)
        predicted_label = self.get_class(predicted_score.T)
        
        return predicted_label
