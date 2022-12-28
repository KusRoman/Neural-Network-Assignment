#   COMPENG 4SL4 Assignment #05
#   
#   Roman Kus - kusr - 400212354
#
#   This assignment experiments with neural network classifiers with two hidden
#   layers for binary classification. The banknote authentication data set 
#   is used. There are four predictor variables (i.e., features) and the goal 
#   is to predict if a banknote is authentic (class 0) or a forgery (class 1).

import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn import model_selection as sk
import matplotlib.pyplot as plt
import random
from itertools import chain

STUDENT_NUM_SEED = 2354

class FNN:

    def __init__(self, num_inputs = 3, num_hidden=[3, 3], num_outputs = 2):

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # list where each entry is the qty of neurons in the layer represented
        # by its index
        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        # initialize required variable arrays
        weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

        activations = []
        for i in range (len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range (len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives

    # compute forward prop starting with data inputs
    def forward_propogate(self, inputs):

        activations = inputs
        self.activations[0] = inputs

        for i, w in enumerate(self.weights):
            
            net_inputs = np.matmul(activations, w)
           
            if i != len(self.weights) - 1:
                activations = self.ReLU(net_inputs)
            else:
                activations = self.sigmoid(net_inputs)

            # print("New Activation: {}".format(activations))
            self.activations[i+1] = activations

            # a_3 = ReLU(h_3)
            # h_3 = a_2 * W_2
        
        return activations
    
    def back_propogate(self, error, verbose=False):
        
        # EXAMPLE OF SIGMOID @ OUTPUT AND ReLU AT HIDDEN LAYERS
        #
        # dE/dW_i = (t/a_[i+1]) + (1-t)/(1-a_[i+1]) * s'(h_[i+1]) * a_i
        # s'(h_[i+1]) = s(h_[i+1])(1 - s(h_[i+1])) 
        # s(h_[i+1]) = a_[i+1]
        #
        # dE/dW_[i-1] = (t/a_[i+1]) + (1-t)/(1-a_[i+1]) * s'(h_[i+1]) * W_i 
        #   * ReLU'(h_[i]) * a_[i-1]

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
        
            
            if i == len(self.derivatives) - 1:
                product = error*self.sigmoid_derivative(activations)
            else:
                product = error*self.ReLU_derivative(activations)
            
          
            product_reshaped = product.reshape(product.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations_reshaped, product_reshaped)
            error = np.dot(product, self.weights[i].T)

            if verbose:
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))
        
        return error
    
    # train model using forward and backward prop
    def train(self, inputs_train, targets_train, inputs_val, targets_val, epochs, learning_rate, tau, verbose = False):
        dict_train_error = {}
        dict_val_error = {}
        for i in range(epochs):
            error_train = 0
            error_val = 0
            set = list(zip(inputs_train, targets_train, inputs_val, targets_val))
            random.shuffle(set)

            for input_train, target_train, input_val, target_val in set:
                
                #forward prop
                output_train = self.forward_propogate(input_train)
                            
                #calculate error derivative
                error =  -(target_train/output_train) + ((1-target_train)/(1-output_train))
                
                #back prop
                self.back_propogate(error)

                #SGD
                self.gradient_descent(learning_rate)

                error_train += self.CEloss(target_train,output_train)
                
                output_val = self.forward_propogate(input_val)
                
                error_val += self.CEloss(target_val,output_val)
            
            if verbose:
                print("Train Error: {} at epoch {}".format(error_train/len(inputs_train), i))
                print("Val Error: {} at epoch {}".format(error_val/len(inputs_val), i))
                print("Learning Rate: {} at epoch {}".format(learning_rate, i))
            
            # Plotting Dictionary
            dict_train_error[i] = np.float(error_train/len(inputs_train))
            dict_val_error[i] = np.float(error_val/len(inputs_train))

            #Adaptive learning rate
            if i < tau:
                learning_rate = self.LR_decay(learning_rate, tau, i)
        
        return [dict_train_error, dict_val_error, output_val]
        
    
    def test(self, inputs, targets, epochs):
        for i, target in enumerate(targets):
            output = self.forward_propogate(inputs)
            sum_error += self.CEloss(target,output[i])
        print(sum_error/len(inputs))
        return sum_error/len(inputs)

    # Stochastic gradient descent, processing one input at a time. Takes
    # learning rate as input
    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
    
            derivatives = self.derivatives[i]
            weights = weights - derivatives * learning_rate
            self.weights[i] = weights
           
    # Cross entropy loss function
    def CEloss(self,target,output):
        return -(target*np.log(output))-((1 - target)*np.log(1 + 1e-15 - output))         
    
    # ReLU activation function
    def ReLU(self, arg):
        return np.maximum(0,arg)
    
    # ReLU derivative function
    def ReLU_derivative(self, arg):
        arg[arg<=0] = 0
        arg[arg>0] = 1
        return arg

    # sigmoid activation function for output  
    def sigmoid(self, arg):
        return 1 / (1 + np.exp(-arg))

    # derivative of sigmoid function
    def sigmoid_derivative(self, arg):
        return arg*(1-arg) 
    
    #learning rate decay 
    def LR_decay(self, learning_rate_init, tau, epoch):
        epoch += 1
        return ((1 - (epoch/tau))*learning_rate_init) + (epoch/tau)*(learning_rate_init/100)

     
    # classify(model, threshold) will create a matrix of classifications according to a
    # specified threshold
    def classify(self, model, threshold):
        copy = model
        for i in range(copy.shape[0]):
            if copy[i] >= self.sigmoid(threshold):
                copy[i] = 1
            else:
                copy[i] = 0
        return copy

    #Misclassification
    def misclassification(self, label, target):
        diff = np.subtract(label, target)
        misclassification = np.sum(np.absolute(diff))/diff.shape[0]
        return misclassification
    
    # determine class given activation value according to sigmoid function
    def classify(self, activations, threshold):
        copy = activations
        for i in range(copy.shape[0]):
            if copy[i] >= self.sigmoid(threshold):
                copy[i] = 1
            else:
                copy[i] = 0
        return copy


if __name__ == "__main__":
    
    np.random.seed(24)

    #Allocate space
    X = np.zeros(shape=(1372, 5))

    # Input data into a matrix
    with open('data_banknote_authentication.txt') as f:
        for i, line in enumerate(f):
            
            #Get each line from txt file, delimited by commas and remove newspace char
            row = line.strip().split(",")  

            #Turn into numpy array and convert strings into int
            row = np.array(row).astype(float)

            X[i] = row

    #Assign last column of text file as targets, then get rid of them from data set       
    t = X[:,4:]; X = np.delete(X, 4, 1)

    #Split data set into training and test sets
    X_train, X_test, y_train, y_test, = sk.train_test_split(
        X, t, test_size = 0.4, random_state = STUDENT_NUM_SEED)

    #Use half of the test set as a validation set (effectively 60/20/20 split)
    X_test, X_val, y_test, y_val, = sk.train_test_split(
        X_test, y_test, test_size = 0.5, random_state = STUDENT_NUM_SEED)
    
    #Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_val = sc.transform(X_val)

    # for N1 in range(2,5):
    #     for N2 in range(2,5):
    NN = FNN(4, [2, 4], 1)
    output = NN.train(X_train, y_train, X_test, y_test, 200, 0.01, 100, verbose=True)
    out = NN.forward_propogate(X_val)
    out = NN.classify(out, 0.5)
    print(NN.misclassification(out, y_val)) 
    
    #plt.plot(output[0].keys(), output[0].values())
    plt.plot(output[1].keys(), output[1].values())
    plt.title("Cross Entropy Loss, [n1,n2] = [{},{}]".format(2,4), loc='left')
    plt.title("alpha = 0.005", loc='right')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend(('Test Error',),
    loc='upper center', shadow=True)    
    plt.show()
    

    # NN = FNN(4, [2, 2], 1)
    # output = NN.train(X_train, y_train, X_val, y_val, 200, 0.005, 0, verbose=True)

    # plt.plot(output[0].keys(), output[0].values())
    # plt.plot(output[1].keys(), output[1].values())
    # plt.show()

    # out = NN.forward_propogate(X_val)
    # out = NN.classify(out, 0.5)
    # print(NN.misclassification(out, y_val))
 
    



    
    





        