import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

class Perceptron:
    def __init__(self, eta, epochs):  # self makes variable to access globally within the class
        self.weights = np.random.randn(3)*1e-4
        logging.info(f'Initial weights before training: \n {self.weights}')
        self.eta = eta        # Learning Rate
        self.epochs = epochs  # Iteration applied to reduce total loss

    def activationFunction(self, inputs, weights):
        z = np.dot(inputs, weights)  # Z = w1x1+w2x2+w0*bias (Where x are inputs and w are weights and bias is constant)
        return np.where(z>0, 1, 0)
    
    def fit(self,X,y): #X and y train values
        self.X = X
        self.y = y 

        X_with_bias = np.c_[self.X , -np.ones((len(self.X),1))]
        logging.info(f"X along with bias: \n {X_with_bias}")

        for epoch in tqdm(range(self.epochs), total=self.epochs, desc="Training the model"):
            logging.info("--"*10)
            logging.info(f"for epoch {epoch}")
            logging.info("--"*10)

            y_hat = self.activationFunction(X_with_bias, self.weights)  # Forward propagation
            logging.info(f"Predicted the output for forward pass: \n {y_hat}")
            self.error = self.y - y_hat
            logging.info(f"Error: \n {self.error}")
            self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error) # Backward propagation
            logging.info(f"Updated weights after epoch: \n {epoch}/{self.epochs} : \n {self.weights}")
            logging.info("####"*10) 

    def predict(self, X):
        X_with_bias = np.c_[X , -np.ones((len(X),1))]
        return self.activationFunction(X_with_bias,self.weights)

    def total_loss(self):
        total_loss = np.sum(self.error)
        logging.info(f"total loss: \n {total_loss}")
        return total_loss