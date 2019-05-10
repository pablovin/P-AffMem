# -*- coding: utf-8 -*-
"""
Associative GWR based on Marsland et al. (2002)'s
          Grow-When-Required network and Parisi et al. (2017) associative learning.

adapted from: https://github.com/giparisi/gwr-tb/blob/master/agwr.py

"""

import scipy.spatial
import numpy as np
import math
from heapq import nsmallest
from metrics import concordance_correlation_coefficient_python

class AffectiveMemory:
    
    def __init__(self):
        self.iterations = 0
    
    def init_network(self, ds, labels, random):
        # Number of neurons
        self.num_nodes = 2
        # Dimensionality of weights
        self.dimension = ds.vectors.shape[1]
        # Start with two neurons
        self.weights = np.zeros((self.num_nodes, self.dimension))
        # Habituation counters
        self.habn = np.ones(self.num_nodes)
        # Connectivity matrix
        self.edges = np.ones((self.num_nodes, self.num_nodes))
        # Age matrix
        self.ages = np.zeros((self.num_nodes, self.num_nodes))
        # Label histogram
        self.alabelsArousal = np.zeros((self.num_nodes, 1))
        self.alabelsValence = np.zeros((self.num_nodes, 1))
        # Initialize weights
        self.random = random
        
        if self.random:
            init_ind = np.random.randint(0, ds.shape[0], 2)
        else:
            init_ind = list(range(0, self.num_nodes))

        for i in range(0, len(init_ind)):           
            self.weights[i] = ds[init_ind[i]]
            self.alabels[i, int(labels[i])] = 1

    def find_bmus(self, input_vector, **kwargs):
        second_best = kwargs.get('s_best', False)
        distances = np.zeros(self.num_nodes)
        for i in range(0, self.num_nodes):
            distances[i] = self.compute_distance(self.weights[i], 
                                                 input_vector, self.dis_metric)
        
        if second_best:
            # Compute best and second-best matching units
            return self.find_bs(distances)
        else:
            b_index = distances.argmin()
            b_distance = distances[b_index]
            return b_index, b_distance

    def compute_distance(self, x, y, m):
        return np.linalg.norm(x-y) if m else scipy.spatial.distance.cosine(x, y)

    def add_node(self, b_index, input_vector):
        new_weight = np.array([np.dot(self.weights[b_index] + input_vector,
                                      self.new_node)])
        self.weights = np.concatenate((self.weights, new_weight), axis=0)
        self.num_nodes += 1

    def habituate_node(self, index, tau, **kwargs):
        new_node = kwargs.get('new_node', False)
        if not new_node:
            self.habn[index] += (tau * 1.05 * (1. - self.habn[index]) - tau)
        else:
            self.habn.resize(self.num_nodes) 
            self.habn[index] = 1
            
    def update_neighbors(self, input, index, epsilon):
        b_neighbors = np.nonzero(self.edges[index])
        for z in range(0, len(b_neighbors[0])):
            neIndex = b_neighbors[0][z]

            distances = self.compute_distance(self.weights[neIndex],
                                              input, self.dis_metric)
            # Compute activity
            a = math.exp(-distances)

            self.update_weight(input, neIndex, epsilon, a)
            self.habituate_node(neIndex, self.tau_n, new_node=False)
 
    def update_weight(self, input, index, epsilon, activation):

        delta = np.array([np.dot((input-self.weights[index]), 
                                  epsilon)]) * self.habn[index]*(1-activation)
        self.weights[index] = self.weights[index] + delta
        
    def update_labels(self, bmu, label, **kwargs):
        new_node = kwargs.get('new_node', False)        

 
       if not label[0] > 1: # Guarantees that a image without a valid label does not update the network's labels.
		if not new_node:
		        if self.alabelsArousal[bmu] < label[0]:
		            self.alabelsArousal[bmu] += self.a_inc*self.alabelsArousal[bmu]
		        else:
		            self.alabelsArousal[bmu] -= self.a_inc * self.alabelsArousal[bmu]


		        if self.alabelsValence[bmu] < label[0]:
		            self.alabelsValence[bmu] += self.a_inc*self.alabelsValence[bmu]
		        else:
		            self.alabelsValence[bmu] -= self.a_inc * self.alabelsValence[bmu]
		else:
		    self.alabels = np.concatenate((self.alabelsArousal, label[0]), axis=0)
		    self.alabels = np.concatenate((self.alabelsValence, label[1]), axis=0)


    def update_edges(self, fi, si, **kwargs):
        new_index = kwargs.get('new_index', False)
        self.ages += 1
        if not new_index:
            self.edges[fi, si] = 1  
            self.edges[si, fi] = 1
            self.ages[fi, si] = 0
            self.ages[si, fi] = 0
        else:
            self.edges.resize((self.num_nodes, self.num_nodes))
            self.ages.resize((self.num_nodes, self.num_nodes))
            self.edges[fi, si] = 0
            self.edges[si, fi] = 0
            self.ages[fi, si] = 0
            self.ages[si, fi] = 0
            self.edges[fi, new_index] = 1
            self.edges[new_index, si] = 1
      
    def remove_old_edges(self):
        for i in range(0, self.num_nodes):
            neighbours = np.nonzero(self.edges[i])
            for j in neighbours[0]:
                if self.ages[i, j] >  self.max_age:
                    self.edges[i, j] = 0
                    self.edges[j, i] = 0
                    self.ages[i, j] = 0
                    self.ages[j, i] = 0
                              
    def remove_isolated_nodes(self):
        ind_c = 0
        rem_c = 0
        while (ind_c < self.num_nodes):
            neighbours = np.nonzero(self.edges[ind_c])
            if len(neighbours[0]) < 1:
                self.weights = np.delete(self.weights, ind_c, axis=0)
                self.alabels = np.delete(self.alabels, ind_c, axis=0)
                self.edges = np.delete(self.edges, ind_c, axis=0)
                self.edges = np.delete(self.edges, ind_c, axis=1)
                self.ages = np.delete(self.ages, ind_c, axis=0)
                self.ages = np.delete(self.ages, ind_c, axis=1)
                self.habn = np.delete(self.habn, ind_c)
                self.num_nodes -= 1
                rem_c += 1
            else:
                ind_c += 1
        print ("(-- Removed %s neuron(s))" % rem_c)
 
    def find_bs(self, dis):
        bs = nsmallest(2, ((k, i) for i, k in enumerate(dis)))
        return bs[0][1], bs[0][0], bs[1][1]
               
    def train_agwr(self, ds, labels, epochs, a_threshold, l_rates):
        
        self.samples = ds.shape[0]
        
        self.max_epochs = epochs
        self.a_threshold = a_threshold   
        self.epsilon_b, self.epsilon_n = l_rates
        
        self.hab_threshold = 0.2
        self.tau_b = 0.3
        self.tau_n = 0.1
        self.max_nodes = self.samples # OK for batch, bad for incremental
        self.dis_metric = 1 # 1 = Euclidean, 0 = Cosine
        self.max_neighbors = 6
        self.max_age = 600
        self.new_node = 0.4 
        self.a_inc = 0.4 # labeling factor
  
        # Start training
        error_counter = np.zeros(self.max_epochs)
        for epoch in range(0, self.max_epochs):
            for iteration in range(0, self.samples):
                
                # Generate input sample
                input = ds[iteration]
                label = labels[iteration]
                
                # Find the best and second-best matching neurons
                b_index, b_distance, s_index = self.find_bmus(input, s_best=True)
                
                # Quantization error
                error_counter[epoch] += b_distance
                
                # Compute network activity
                a = math.exp(-b_distance)
                
                if (a < self.a_threshold
                    and self.habn[b_index] < self.hab_threshold
                    and self.num_nodes < self.max_nodes):
                   
                    # Add new neuron
                    n_index = self.num_nodes
                    self.add_node(b_index, input)
        
                    # Add label histogram
                    self.update_labels(n_index, label, new_node=True)

                    # Update edges and ages
                    self.update_edges(b_index, s_index, new_index=n_index)

                    # Habituation counter                    
                    self.habituate_node(n_index, self.tau_b, new_node=True)
                    
                else:
                    # Habituate BMU
                    self.habituate_node(b_index, self.tau_b)

                    # Update BMU's weight vector
                    self.update_weight(input, b_index, self.epsilon_b, a)

                    # Update BMU's edges and ages
                    self.update_edges(b_index, s_index) 

                    # Update BMU's neighbors
                    self.update_neighbors(input, b_index, self.epsilon_n)
                    
                    # Update BMU's label histogram
                    self.update_labels(b_index, label)
                    
                self.iterations += 1

            # Remove old edges
            self.remove_old_edges()

            # Average quantization error (AQE)
            error_counter[epoch] /= self.samples
            
            print ("(Epoch: %s, NN: %s, AQE: %s)" % 
                   (epoch+1, self.num_nodes, error_counter[epoch]))
            
        # Remove isolated neurons
        self.remove_isolated_nodes()

    def evaluate_affectiveMemory(self, test_ds, labels):
        self.bmus_index = -np.ones(self.samples)
        self.bmus_label = -np.ones(self.samples)
        self.bmus_activation = np.zeros(self.samples)
        arousals = []
        valences = []
        for i in range(0, test_ds.vectors.shape[0]):
            input = test_ds[i]
            b_index, b_distance = self.find_bmus(input)
            self.bmus_index[i] = b_index
            self.bmus_activation[i] = math.exp(-b_distance)
            arousals.append(self.alabelsArousal[b_index])
	    valences.append(self.alabelsValences[b_index])
            

    return concordance_correlation_coefficient_python(arousals, labels[0]), concordance_correlation_coefficient_python(valences, labels[1])
