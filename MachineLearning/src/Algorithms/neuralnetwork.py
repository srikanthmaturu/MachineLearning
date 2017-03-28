# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

from random import randrange
from math import exp
from math import pow
from DataFileLoaders.dataset import *
import time
import numpy as np
import threading
from Algorithms.threadenvironment import *

class NeuralNetwork:
    def __init__(self, network_config = ''):
        self.network_config = network_config
        self.layers = []
        self.layers_size = self.network_config.layers_size
        self.input_layer = ''
        self.output_layer = ''
        self.network_layers = []
        self.training_stop_criteria = ''
        self.preferred_validation_set_error = ''
        self.training_iteration = 0
        self.print_mode = False
        self.eta = ''
    
    def set_print_mode(self, mode):
        self.print_mode = mode
        
    def set_config(self, network_config):
        self.network_config = network_config
    
    def get_config(self):
        return self.network_config
    
    def build_network(self):
        if self.network_config == '':
            print('No network configuration is initialized.. returning...')
            return
        self.network_layers = []
        upstream_layer = ''
        
        for layer_config in self.network_config.layers_configs:
            layer = self.build_layer(layer_config, upstream_layer)
            self.network_layers.append(layer)
            upstream_layer = layer
        
        self.training_stop_criteria = self.network_config.training_stop_criteria
        
    def build_layer(self, layer_config, upstream_layer):
        if(isinstance(layer_config, InputLayerConfig)):
            self.input_layer = self.build_input_layer(layer_config)
            return self.input_layer
        elif(isinstance(layer_config, OutputLayerConfig)):
            self.output_layer = self.build_output_layer(layer_config, upstream_layer) 
            upstream_layer.set_downstream_layer(self.output_layer)
            return self.output_layer
        elif(isinstance(layer_config, HiddenLayerConfig)):
            layer = self.build_hidden_layer(layer_config, upstream_layer)
            upstream_layer.set_downstream_layer(layer)
            return layer
    
    def build_input_layer(self, layer_config):
        input_layer = InputLayer(layer_config)
        input_layer.build_layer()
        return input_layer
    
    def build_output_layer(self, layer_config, upstream_layer):
        output_layer = OutputLayer(layer_config, upstream_layer)
        output_layer.build_layer()
        return output_layer
    
    def build_hidden_layer(self, layer_config, upstream_layer):
        hidden_layer = HiddenLayer(layer_config, upstream_layer)
        hidden_layer.build_layer()
        return hidden_layer
    
    def set_training_dataset(self, training_dataset):
        self.input_layer.set_training_dataset(training_dataset)
        self.output_layer.set_training_dataset(training_dataset)
        self.training_iteration = 0
    
    def set_validation_dataset(self, validation_dataset):
        self.input_layer.set_validation_dataset(validation_dataset)
        self.output_layer.set_validation_dataset(validation_dataset)
        self.current_validation_set_error = ''
        
    def set_testing_dataset(self, testing_dataset):
        self.input_layer.set_testing_dataset(testing_dataset)
        self.output_layer.set_testing_dataset(testing_dataset)
    
    def set_mode(self, mode):
        self.set_input_layer_mode(mode)
        self.set_output_layer_mode(mode)
        
    def set_input_layer_mode(self, mode):
        self.input_layer.set_input_layer_mode(mode)
    
    def set_output_layer_mode(self, mode):
        self.output_layer.set_output_layer_mode(mode)
        
    def initialize_input_layer(self):
        self.input_layer.reset_nodes_points_counters()
        self.input_layer.initialize_nodes_points_counters()
    
    def initialize_output_layer(self):
        self.output_layer.reset_nodes_expected_label_index_counters()
        self.output_layer.initialize_nodes_expected_label_index_counters()
    
    def initialize_network(self):
        for layer in self.network_layers:
            layer.initialize_layer()
        self.initialize_network_error_trackers()
    
    def set_eta(self, eta):
        self.eta = eta
        for layer in self.network_layers:
            if(not isinstance(layer, InputLayer)):
                layer.set_eta(eta)
        
    def train(self, eta, training_dataset=''):
        if(self.print_mode):
            print('Training of the neural network using training dataset is started....')
            
        self.build_network()
        
        if(training_dataset !=''):
            self.set_training_dataset(training_dataset)
            
        self.set_mode('training')
        self.initialize_network()
        self.set_eta(eta)
        self.preferred_validation_error = 1.1
        if(self.print_mode):
            print('Neural network training stopping criteria is set to :', self.training_stop_criteria['type'], '=', self.training_stop_criteria['value'])
        self.highlevel_iteration = 0
        iterations_results = {'validation_errors':[],'training_times':[],'training_errors':[],'training_iterations':[]}
        for i in range(1, self.training_stop_criteria['max_iterations']+1):
            self.highlevel_iteration = i
            if(self.print_mode):
                print('High level Iteration: ', self.highlevel_iteration)
            training_start_time = time.process_time()
            
            result = self.train_using_weights_change()
            training_end_time = time.process_time()
            training_time = (training_end_time - training_start_time)
            iterations_results['training_times'].append(training_time)
            iterations_results['training_errors'].append(result)
            iterations_results['training_iterations'].append(self.training_iteration)
            if(self.print_mode):
                print('---Training time: ', training_time, ' Final Training error: ', result)
            
            self.confusion_matrix = self.get_confusion_matrix()

            self.validate()
            validation_error = self.get_error()
            if(self.print_mode):
                print('---Validation error: ', validation_error)
            iterations_results['validation_errors'].append(validation_error)
            if(self.highlevel_iteration == 1 or self.preferred_validation_error > validation_error):
                self.preferred_network_layers = self.network_layers
                self.preferred_input_layer = self.input_layer
                self.preferred_output_layer = self.output_layer
                self.preferred_validation_error = validation_error
                self.preferred_confusion_matrix = self.confusion_matrix
                    
            if(self.highlevel_iteration != self.training_stop_criteria['max_iterations']):
                self.build_network()
                self.set_mode('training')
                self.initialize_network()
                self.set_eta(self.eta)
                if(self.print_mode):
                    print('Resetting neural network for next round of training(high level iteration: '+str(self.highlevel_iteration)+')...')
                if(training_dataset !=''):
                    self.set_training_dataset(training_dataset)
            else:
                if(self.preferred_validation_error < validation_error):
                    self.network_layers = self.preferred_network_layers
                    self.input_layer = self.preferred_input_layer
                    self.output_layer = self.preferred_output_layer
                    self.confusion_matrix = self.preferred_confusion_matrix
                        
        if(self.print_mode):
            print('Training Ended.....')
        self.training_iterations_results = iterations_results
        #print(iterations_results)
        
    def train_using_weights_change(self):
        self.training_iteration = 0
        while(not self.check_stopping_criteria()):
            self.training_iteration += 1
            self.set_mode('training')
            self.initialize_input_layer()
            self.initialize_output_layer()
            self.initialize_network_error_trackers()
            p = 0
            while(self.next_point_available()):
                self.feed_forward()
                self.back_propagate()
                self.update_error()
                self.move_to_next_point()
                p += 1
                if(p % 100 == 0):
                    print("processing point: ", p)
            error = self.get_error()
            result = error
            if(self.print_mode):
                self.compute_network_weights_change()
                print('Iteration ', self.training_iteration , ': ', ' Training error - ', self.get_error(), 'Network weigths change: ', self.get_network_weights_change())
            if(error == 0.0):
                if(self.print_mode):
                    print('Training error became zero, training will be stopped....')
                break
            
        return result
        
    def check_stopping_criteria(self, validation_dataset = ''):
        if(self.training_iteration == 0):
            return False
        
        if(self.training_iteration == 1000):
            return True
        
        if(self.training_stop_criteria['type'] == 'weights_change'):
            return self.check_weights_criteria()
        elif(self.training_stop_criteria['type'] == 'max_iterations'):
            return True if self.training_iteration == self.training_stop_criteria['value'] else False
        
    def check_weights_criteria(self):
        min_weight_change = self.training_stop_criteria['value']
        self.compute_network_weights_change()
        #print(self.get_network_weights_change())
        return not ((min_weight_change - self.get_network_weights_change()) < 0)
    
    def check_validation_set_error_criteria(self, validation_dataset = ''):
        error_change = self.get_validation_set_error_change()
        if(error_change < self.training_stop_criteria['value']):
            return False
        
    def get_validation_set_error_change(self, validation_dataset = ''):
        self.validate()
        prev_validation_error = self.current_validation_error
        self.current_validation_error = self.get_error()
        return  (prev_validation_error - self.current_validation_error)
    
    def feed_forward(self):
        self.input_layer.propagate_inputs()
        
    def back_propagate(self):
        self.output_layer.back_propagate_errors()
    
    def test(self, testing_dataset=''):
        #print('Testing started....')
        if(testing_dataset != ''):
            self.set_testing_dataset(testing_dataset)
            
        self.set_mode('testing')
        self.initialize_input_layer()
        self.initialize_output_layer()
        self.initialize_network_error_trackers()
        while(self.next_point_available()):
                self.feed_forward()
                self.update_error()
                self.move_to_next_point()
        #print('Testing Error: ', self.get_error())
        #print('Testing complete....')
    
    def validate(self, validation_dataset=''):
        #print('Validation Started...')
        if(validation_dataset != ''):
            self.set_validation_dataset(validation_dataset)
            
        self.set_mode('validation')
        self.initialize_input_layer()
        self.initialize_output_layer()
        self.initialize_network_error_trackers()
        while(self.next_point_available()):
            self.feed_forward()
            self.update_error()
            self.move_to_next_point()
        #print('Validation error: ', self.get_error())
        #print('Validation complete....')
        
    def compute_network_weights_change(self):
        for layer in self.network_layers:
            if(not isinstance(layer, InputLayer)):
                layer.compute_layer_weights_change()
        self.network_weights_change = sum([layer.get_layer_weights_change() for layer in self.network_layers if(not isinstance(layer, InputLayer))])
    
    def initialize_network_error_trackers(self):
        self.output_layer.reset_error_tracker()
        self.output_layer.initialize_error_tracker()
        
    def update_error(self):
        self.output_layer.update_error()
    
    def get_error(self):
        return self.output_layer.get_error()
        
    def get_network_weights_change(self):
        return self.network_weights_change
    
    def next_point_available(self):
        return self.input_layer.next_point_available() 
    
    def move_to_next_point(self):
        self.input_layer.move_to_next_point()
        self.output_layer.move_to_next_expected_label()
    
    def print_predictions(self):
        pass
    
    def get_confusion_matrix(self):
        return self.output_layer.get_confusion_matrix()
    
    def display_confusion_matrix(self, confusion_matrix=''):
        if(confusion_matrix == ''):
            confusion_matrix = self.get_confusion_matrix()
        label_indexes_to_names_mapping = {value:key for key, value in self.input_layer.get_training_dataset().label.indexed_unique_nominal_values.items()}
        label_names = [label_indexes_to_names_mapping[key] for key in sorted(label_indexes_to_names_mapping.keys())]
        #print('Printing confusion matrix: ')
        label_names_lengths = [len(label_name) for label_name in label_names]
        max_label_name_length = max(label_names_lengths + [16])
        field_size = max_label_name_length + 3
        label_string = '{:' + str(field_size) + '}'
        for label_name in label_names:
            label_string = label_string + '{:' + str(field_size) + '}'
        header = ['Actual\\Predicted'] + label_names
        
        print(label_string.format(*header))
        print(('{:-<'+str(len(label_string.format(*header)))+'}').format('-'))
        
        current_line = ''
        for label_name, label_composition in zip(label_names, confusion_matrix):
            current_line += '{:'+ str(field_size) +'}'
            for label_count in label_composition:
                current_line += '{:<'+ str(field_size) +'}'
            print(current_line.format(label_name, *label_composition))
            current_line = ''
        print('')
        
    def get_true_positive_rates(self):
        confusion_matrix = self.confusion_matrix
        instance_sizes = [sum(row) for row in confusion_matrix]
        return [confusion_matrix[i][i]/instance_sizes[i] for i in range(0, len(confusion_matrix[0]))]
        
    def get_false_positive_rates(self):
        confusion_matrix = self.confusion_matrix
        no_of_classes = len(confusion_matrix[0])
        instance_sizes = [sum(row) for row in confusion_matrix]
        false_positive_errors = []
        for current_class in range(0, no_of_classes):
            other_classes_instances_size = sum([instance_sizes[class_index] for class_index in range(0, no_of_classes) if(class_index != current_class)])
            false_positive_count = 0
            for j in range(0, no_of_classes):
                if(current_class!=j):
                    false_positive_count += confusion_matrix[j][current_class]
            false_positive_errors.append(false_positive_count/other_classes_instances_size)
        return false_positive_errors
    
    def print_tp_fp_rates(self):
        tp_rates = self.get_true_positive_rates()
        fp_rates = self.get_false_positive_rates()
        label_indexes_to_names_mapping = {value:key for key, value in self.input_layer.get_training_dataset().label.indexed_unique_nominal_values.items()}
        label_names = [label_indexes_to_names_mapping[key] for key in sorted(label_indexes_to_names_mapping.keys())]
        
        for i in range(0, len(tp_rates)):
            print('Class ',label_names[i],': tp rate = ', tp_rates[i], ' fp rate = ', fp_rates[i])
        print('')
    
    def get_tp_fp_rates(self):
        tp_rates = self.get_true_positive_rates()
        fp_rates = self.get_false_positive_rates()
        return (tp_rates, fp_rates)
    
    def set_threshold(self, threshold):
        self.output_layer.set_threshold(threshold)
    
    def print_95_percent_confidence_interval(self, error='', dataset_size=''):
        if(error == '' and dataset_size == ''):
            error = self.get_error()
            dataset_size = self.input_layer.get_current_dataset().size
            
        standard_deviation = 1.96 * ((error * (1 - error)) / dataset_size)**0.5
        lower_bound = error - standard_deviation
        upper_bound = error + standard_deviation
        print('Approximately with 95% confidence, generalization_error lies in between ({},{})'.format(lower_bound, upper_bound))
    
    def print_network_information(self):
        network_information = self.get_network_information()
        print('Neural Network Information')
        print('')
        print('Input Layer:')
        print('{:-<70}'.format('-'))
        print('No of input nodes: ', network_information['input_layer']['no_of_input_nodes_used'])
        name_string = 'Input nodes attribute names: \n'
        for node in self.input_layer.nodes:
            name_string += ('-----{}\n')
        name_string = name_string.format(*network_information['input_layer']['attribute_names'])
        print(name_string)
        print('')
        print('Hidden Layers:')
        print('{:-<70}'.format('-'))
        print('No of hidden layers: ', network_information['hidden_layers']['no_of_hidden_layers'])
        for hidden_layer_index in range(1, len(self.network_layers)-1):
            print('\nHidden layer index: ', hidden_layer_index)
            hidden_layer_nodes_weights = network_information['hidden_layers']['hidden_layers_nodes_weight_vectors'][hidden_layer_index-1]
            print('No of hidden nodes: ', len(hidden_layer_nodes_weights))
            for node_index in range(0, len(hidden_layer_nodes_weights)):
                print('-----Hidden node index: ', node_index, '(weight_vector_length:'+str(len(hidden_layer_nodes_weights[node_index]))+')','Hidden node weight vector: ', hidden_layer_nodes_weights[node_index])
        print('')
        print('Output Layer:')
        print('{:-<70}'.format('-'))
        name_string = 'Output labels: \n'
        for output_label in network_information['output_layer']['output_labels']:
            name_string += ('-----{}\n')
        name_string = name_string.format(*network_information['output_layer']['output_labels'])
        print(name_string)
        print('No of Output nodes: ', network_information['output_layer']['no_of_output_nodes'])
        output_layer_nodes_weights = network_information['output_layer']['output_layer_nodes_weight_vectors']
        
        for node_index in range(0, len(output_layer_nodes_weights)):
                print('-----Output node index: ', node_index,'(weight_vector_length:'+str(len(output_layer_nodes_weights[node_index]))+')', 'Output node weight vector: ', output_layer_nodes_weights[node_index])
        print('')
        print('{:-<70}'.format('-'))
        print('Training information:')
        print('Learning rate eta: ', network_information['training_details']['eta'], ' Training stop criteria = ', network_information['training_details']['training_stop_criteria'])
    
    def get_network_information(self):
        network_information = {}
        network_information['output_layer'] = { 'no_of_output_nodes':(len(self.output_layer.nodes))}
        network_information['output_layer']['output_layer_nodes_weight_vectors'] = self.output_layer.get_layer_nodes_weight_vectors()
        
        label_indexes_to_names_mapping = {value:key for key, value in self.input_layer.get_training_dataset().label.indexed_unique_nominal_values.items()}
        label_names = [label_indexes_to_names_mapping[key] for key in sorted(label_indexes_to_names_mapping.keys())]
        
        network_information['output_layer']['output_labels'] = label_names
        network_information['input_layer'] = {'no_of_input_nodes_used':(len(self.input_layer.nodes))}
        network_information['input_layer']['attribute_names'] = [node.get_node_att_name() for node in self.input_layer.nodes]
        network_information['hidden_layers'] = {'no_of_hidden_layers':(len(self.network_layers)-2)}
        hidden_layers_nodes_weight_vectors = []
        for hidden_layer in self.network_layers[1:-1]:
            #print(hidden_layer)
            hidden_layers_nodes_weight_vectors.append(hidden_layer.get_layer_nodes_weight_vectors())
        network_information['hidden_layers']['hidden_layers_nodes_weight_vectors'] = hidden_layers_nodes_weight_vectors
        network_information['training_details'] = {'eta':self.eta, 'training_stop_criteria':self.training_stop_criteria}
        return network_information
        
class Node:
    def __init__(self):
        self.inputs = ''
        self.output = ''
        
    def set_inputs(self, inputs=''):
        self.inputs = inputs
        self.compute_output()
    
    def compute_output(self):
        pass
    
    def get_output(self):
        return self.output
    
class InputNode(Node):
    def __init__(self, attributes_set = [], mode = ''):
        super().__init__()
        
        self.training_attribute, self.testing_attribute, self.validation_attribute = attributes_set
        self.training_size = len(self.training_attribute.attribute)
        self.testing_size = len(self.testing_attribute.attribute)
        self.validation_size = len(self.validation_attribute.attribute)
        self.current_training_point_index = -1
        self.next_training_point_available = False
        self.current_testing_point_index = -1
        self.next_testing_point_available = False
        self.current_validation_point_index = -1
        self.next_validation_point_available = False
            
    def set_training_attribue(self, training_attribute):
        self.training_attribute = training_attribute
        self.training_size = len(self.training_attribute.attribute)
        self.current_training_point_index = -1
        self.next_training_point_available = False
    
    def set_validation_attribute(self, validation_attribute):
        self.validaiton_attribute = validation_attribute
        self.validation_size = len(self.validation_attribute.attribute)
        self.current_validation_point_index = -1
        self.next_validation_point_available = False
    
    def set_testing_attribute(self, testing_attribute):
        self.testing_attribute = testing_attribute
        self.testing_size = len(self.testing_attribute.attribute)
        self.current_testing_point_index = -1
        self.next_testing_point_available = False
    
    def set_mode(self, mode):
        self.mode = mode
        self.reset_training_points_counter()
        self.reset_validation_points_counter()
        self.reset_testing_points_counter()
        
    def get_mode(self):
        return self.mode
    
    def reset_points_counter(self):
        if(self.mode == 'training'):
            self.reset_training_points_counter()
        elif(self.mode == 'testing'):
            self.reset_testing_points_counter()
        elif(self.mode == 'validation'):
            self.reset_validation_points_counter()
    
    def reset_training_points_counter(self):
        self.current_training_point_index = -1
        self.next_training_point_available = False
    
    def reset_validation_points_counter(self):
        self.current_validation_point_index = -1
        self.next_validation_point_available = False
        
    def reset_testing_points_counter(self):
        self.current_testing_point_index = -1
        self.next_testing_point_available = False
    
    def initialize_points_counter(self):
        if(self.mode == 'training'):
            if(self.training_size > 0):
                self.current_training_point_index = 0
                self.next_training_point_available = True
        elif(self.mode == 'testing'):
            if(self.testing_size > 0):
                self.current_testing_point_index = 0
                self.next_testing_point_available = True
        elif(self.mode == 'validation'):
            if(self.validation_size > 0):
                self.current_validation_point_index = 0
                self.next_validation_point_available = True
                
    def next_point_available(self):
        if(self.mode == 'training'):
            return self.next_training_point_available
        elif(self.mode == 'testing'):
            return self.next_testing_point_available
        elif(self.mode == 'validation'):
            return self.next_validation_point_available
    
    def move_to_next_point(self):
        if(self.mode == 'training'):
            self.move_to_next_training_point()
        elif(self.mode == 'testing'):
            self.move_to_next_testing_point()
        elif(self.mode == 'validation'):
            self.move_to_next_validation_point()
    
    def get_current_point(self):
        if(self.mode == 'training'):
            return self.get_current_training_point()
        elif(self.mode == 'testing'):
            return self.get_current_testing_point()
        elif(self.mode == 'validation'):
            return self.get_current_validation_point()
    
    def get_current_training_point(self):
        if(self.next_training_point_available):
            training_point = self.training_attribute.attribute[self.current_training_point_index]
            #self.move_to_next_training_point()
            return training_point
        else:
            return 'no_points_available'
    
    def move_to_next_training_point(self):
        if(self.next_training_point_available):
            self.current_training_point_index += 1
            if(self.current_training_point_index == self.training_size):
                self.next_training_point_available = False
        else:
            return False
    
    def get_current_testing_point(self):
        if(self.next_testing_point_available):
            testing_point = self.testing_attribute.attribute[self.current_testing_point_index]
            #self.move_to_next_testing_point()
            return testing_point
        else:
            return 'no_points_available'
    
    def move_to_next_testing_point(self):
        if(self.next_testing_point_available):
            self.current_testing_point_index += 1
            if(self.current_testing_point_index == self.testing_size):
                self.next_testing_point_available = False
        else:
            return False
    
    def get_current_validation_point(self):
        if(self.next_validation_point_available):
            validation_point = self.validation_attribute.attribute[self.current_validation_point_index]
            #self.move_to_next_validation_point()
            return validation_point
        else:
            return 'no_points_available'
    
    def move_to_next_validation_point(self):    
        if(self.next_validation_point_available):
            self.current_validation_point_index += 1
            if(self.current_validation_point_index == self.validation_size):
                self.next_validation_point_available = False
        else:
            return False
            
    def compute_output(self):
        self.output = self.get_current_point()
    
    def get_node_att_name(self):
        return self.training_attribute.attrname
        
class Neuron(Node):
    def __init__(self, no_of_inputs=''):
        super().__init__()
        self.no_of_inputs = no_of_inputs + 1
        self.weights = np.array([])
        self.prev_weights = np.array([])
        self.prev_cur_weights_euclidean_distance = ''
        self.sigmoid_value = ''
        self.weighted_sum = ''
        self.inputs = np.array([])
        self.delta = ''
        self.eta = ''
        self.bias = 1
    
    def set_eta(self, eta):
        self.eta = eta
    
    def non_numpy_set_inputs(self, inputs):
        self.inputs = [1] + inputs
        self.compute_output()
    
    def set_inputs(self, inputs):
        self.inputs = np.append([1], inputs)
        self.compute_output()
    
    def compute_output(self):
        self.process_inputs()
        
    def process_inputs(self):
        self.compute_weighted_sum()
        self.compute_sigmoid()
        
    def compute_weighted_sum(self):
        #print(self.weights, self.inputs)
        #print(self.weights, '\n\n')
        #print(self.inputs)
        #self.weighted_sum = sum([weight*value for weight, value in zip(self.weights, self.inputs)])
        self.weighted_sum = np.sum(self.weights*self.inputs)
        #print(self.weights, self.inputs, self.weighted_sum)
    
    def compute_sigmoid(self):
        self.sigmoid_value = 1/(1 + exp(-1 * self.weighted_sum))
        self.output = self.sigmoid_value
        
    def set_random_weights(self):
        self.weights = [randrange(0, 100000)/100000 for i in range(0, self.no_of_inputs)]
    
    def get_weights(self):
        return self.weights
    
    def compute_delta(self):
        pass
    
    def get_delta(self):
        return self.delta
    
    def non_numpy_update_weights(self):
        self.prev_weights = [self.weights[i] for i in range(0, len(self.weights))]
        #print(self.weights, self.prev_weights)
        for index in range(0, len(self.weights)):
            self.weights[index] += self.eta * self.delta * self.inputs[index]
        #print(self.weights, self.prev_weights)
        self.compute_prev_cur_weights_euclidean_distance()
        self.prev_cur_weights_euclidean_distance = self.get_prev_cur_weights_euclidean_distance()
    
    def update_weights(self):
        self.prev_weights = np.copy(self.weights)
        self.weights = self.prev_weights + (self.eta*self.delta)*np.array(self.inputs)
        #self.weights = ( 1 + (self.eta*self.delta))*self.prev_weights
        self.compute_prev_cur_weights_euclidean_distance()
        self.prev_cur_weights_euclidean_distance = self.get_prev_cur_weights_euclidean_distance()
    
    def non_nump_py_compute_prev_cur_weights_euclidean_distance(self):
        self.prev_cur_weights_euclidean_distance = pow(sum([(self.prev_weights[i] - self.weights[i]) ** 2 for i in range(0, len(self.weights))]), 0.5)
    
    def compute_prev_cur_weights_euclidean_distance(self):
        self.prev_cur_weights_euclidean_distance = pow(np.sum((self.prev_weights - self.weights)**2), 0.5)
    
    def get_prev_cur_weights_euclidean_distance(self):
        return self.prev_cur_weights_euclidean_distance
    
        
class OutputNode(Neuron):
    def __init__(self, no_of_inputs, labels, multi_label_mode, labels_from_vectors = ''):
        super().__init__(no_of_inputs)
        self.training_label , self.validation_label, self.testing_label = labels
        self.multi_label_mode = multi_label_mode
        if(self.multi_label_mode):
            self.training_expected_label, self.validation_expected_label, self.testing_expected_label = labels_from_vectors
        else:
            self.training_expected_label, self.validation_expected_label, self.testing_expected_label = self.training_label.label, self.validation_label.label, self.testing_label.label
        
        self.expected_training_label_size = len(self.training_expected_label)
        self.current_expected_training_label_index = -1
        self.next_expected_training_label_available = False
        self.expected_validation_label_size = len(self.validation_expected_label)
        self.current_expected_validation_label_index = -1
        self.next_expected_validation_label_available = False
        self.expected_testing_label_size = len(self.testing_expected_label)
        self.current_expected_testing_label_index = -1
        self.next_expected_testing_label_available = False
        
    def set_training_label(self, training_label, training_label_from_label_vector=''):
        self.training_label = training_label
        
        if(self.multi_label_mode):
            self.training_expected_label = training_label_from_label_vector
        else:
            self.training_expected_label = training_label.label
    
        self.expected_training_label_size = len(self.training_expected_label)
        self.current_expected_training_label_index = -1
        self.next_expected_training_label_available = False
    
    def set_validation_label(self, validation_label, validation_label_from_label_vector=''):
        self.validation_label = validation_label
        
        if(self.multi_label_mode):
            self.validation_expected_label = validation_label_from_label_vector
        else:
            self.validation_expected_label = validation_label.label
    
        self.expected_validation_label_size = len(self.validation_expected_label)
        self.current_expected_validation_label_index = -1
        self.next_expected_validation_label_available = False
    
    def set_testing_label(self, testing_label, testing_label_from_label_vector=''):
        self.testing_label = testing_label
        
        if(self.multi_label_mode):
            self.testing_expected_label = testing_label_from_label_vector
        else:
            self.testing_expected_label = testing_label.label
    
        self.expected_testing_label_size = len(self.testing_expected_label)
        self.current_expected_testing_label_index = -1
        self.next_expected_testing_label_available = False
    
    def set_mode(self, mode):
        self.mode = mode
        self.reset_expected_training_label_index_counter()
        self.reset_expected_validation_label_index_counter()
        self.reset_expected_testing_label_index_counter()
        
    def get_mode(self):
        return self.mode
    
    def reset_expected_label_index_counter(self):
        if(self.mode == 'training'):
            self.reset_expected_training_label_index_counter()
        elif(self.mode == 'validation'):
            self.reset_expected_validation_label_index_counter()
        elif(self.mode == 'testing'):
            self.reset_expected_testing_label_index_counter()
        
    def reset_expected_training_label_index_counter(self):
        self.current_expected_training_label_index = -1
        self.next_expected_training_label_available = False
    
    def reset_expected_validation_label_index_counter(self):
        self.current_expected_validation_label_index = -1
        self.next_expected_validation_label_available = False
        
    def reset_expected_testing_label_index_counter(self):
        self.current_expected_testing_label_index = -1
        self.next_expected_testing_label_available = False
    
    def initialize_expected_label_index_counter(self):
        if(self.mode == 'training'):
            if(self.expected_training_label_size > 0):
                self.current_expected_training_label_index = 0
                self.next_expected_training_label_available = True
        elif(self.mode == 'validation'):
            if(self.expected_validation_label_size > 0):
                self.current_expected_validation_label_index = 0
                self.next_expected_validation_label_available = True
        elif(self.mode == 'testing'):
            if(self.expected_testing_label_size > 0):
                self.current_expected_testing_label_index = 0
                self.next_expected_testing_label_available = True
                
    def next_expected_label_available(self):
        if(self.mode == 'training'):
            return self.next_expected_training_label_available
        elif(self.mode == 'validation'):
            return self.next_expected_validation_label_available
        elif(self.mode == 'testing'):
            return self.next_expected_testing_label_available
    
    def move_to_next_expected_label(self):
        if(self.mode == 'training'):
            self.move_to_next_expected_training_label()
        elif(self.mode == 'validation'):
            self.move_to_next_expected_validation_label()
        elif(self.mode == 'testing'):
            self.move_to_next_expected_testing_label()
    
    def get_current_expected_label(self):
        if(self.mode == 'training'):
            return self.get_current_expected_training_label()
        elif(self.mode == 'validation'):
            return self.get_current_expected_validation_label()
        elif(self.mode == 'testing'):
            return self.get_current_expected_testing_label()
    
    def get_current_expected_training_label(self):
        if(self.next_expected_training_label_available):
            training_label = self.training_expected_label[self.current_expected_training_label_index]
            #self.move_to_next_expected_training_label()
            return training_label
        else:
            return 'no_labels_available'
    
    def move_to_next_expected_training_label(self):
        if(self.next_expected_training_label_available):
            self.current_expected_training_label_index += 1
            if(self.current_expected_training_label_index == self.expected_training_label_size):
                self.next_expected_training_label_available = False
        else:
            return False
    
    def get_current_expected_validation_label(self):
        if(self.next_expected_validation_label_available):
            validation_label = self.validation_expected_label[self.current_expected_validation_label_index]
            #self.move_to_next_expected_validation_label()
            return validation_label
        else:
            return 'no_labels_available'
    
    def move_to_next_expected_validation_label(self):
        if(self.next_expected_validation_label_available):
            self.current_expected_validation_label_index += 1
            if(self.current_expected_validation_label_index == self.expected_validation_label_size):
                self.next_expected_validation_label_available = False
        else:
            return False
    
    def get_current_expected_testing_label(self):
        if(self.next_expected_testing_label_available):
            testing_label = self.testing_expected_label[self.current_expected_testing_label_index]
            #self.move_to_next_expected_testing_label()
            return testing_label
        else:
            return 'no_labels_available'
    
    def move_to_next_expected_testing_label(self):
        if(self.next_expected_testing_label_available):
            self.current_expected_testing_label_index += 1
            if(self.current_expected_testing_label_index == self.expected_testing_label_size):
                self.next_expected_testing_label_available = False
        else:
            return False
    
    def compute_delta(self):
        r = self.get_current_expected_label()
        y = self.get_output()
        #print(r, y)
        self.delta = y*(1-y)*(r-y)
        
class HiddenNode(Neuron):
    def __init__(self, no_of_inputs=''):
        super().__init__(no_of_inputs)
    
    def compute_delta(self, downstream_nodes_delta_weight_vectors):
        y = self.get_output()
        wd = 0.0
        
        for node_delta, node_weight in downstream_nodes_delta_weight_vectors:
            wd += node_delta*node_weight
        
        self.delta = y * (1-y) * wd

class Layer:
    def __init__(self, layer_config, upstream_layer='', downstream_layer=''):
        self.layer_config = layer_config
        self.upstream_layer = upstream_layer
        self.downstream_layer = downstream_layer
    
    def set_layer_outputs(self, layer_outputs):
        self.set_layer_outputs = layer_outputs
        
    def get_layer_nodes(self):
        return self.nodes
    
    def get_no_of_layer_nodes(self):
        return len(self.nodes)
    
    def set_downstream_layer(self, downstream_layer):
        self.downstream_layer = downstream_layer
        
    def propagate_inputs(self, layer_inputs=''):
        layer_outputs = []
        #print(layer_inputs, '\n\n')
        for node in self.nodes:
            node.set_inputs(layer_inputs)
            #node.compute_output()
            layer_outputs.append(node.get_output())
        #print(layer_inputs, layer_outputs)
        self.layer_outputs = np.array(layer_outputs)
        if(self.__class__ != InputLayer):
            pass
            #print('Node 1 weights:', self.nodes[0].get_weights())
        if(self.downstream_layer != ''):
            self.downstream_layer.propagate_inputs(self.layer_outputs)
    
    def threaded_propagate_inputs(self, layer_inputs=[]):
        if(layer_inputs == []):
            self.non_threaded_propagate_inputs()
            return
        
        layer_outputs = []
        #len(self.nodes)
        layer_outputs = process_task(self.nodes, layer_inputs, 4)
        self.layer_outputs = np.array(layer_outputs)
        if(self.downstream_layer != ''):
            self.downstream_layer.propagate_inputs(self.layer_outputs)
        
    
    def back_propagate_errors(self, downstream_layer_nodes_deltas = '', downstream_layer_nodes_weight_vectors = ''):
        if(self.__class__ == OutputLayer):
            downstream_layer_nodes_deltas = []
            downstream_layer_nodes_weight_vectors = []
            for node in self.nodes:
                node.compute_delta()
                delta = node.get_delta()
                node_weights = node.get_weights()
                #print('\n \n ', node_weights, '\n')
                node.update_weights()
                #print(node.get_weights())
                downstream_layer_nodes_deltas.append(delta)
                downstream_layer_nodes_weight_vectors.append(node_weights[1:])
            if(self.upstream_layer  != ''):
                self.upstream_layer.back_propagate_errors(downstream_layer_nodes_deltas, downstream_layer_nodes_weight_vectors)
        elif(self.__class__ == HiddenLayer):
            no_of_downstream_nodes = len(downstream_layer_nodes_deltas)
            for node_index in range(0, len(self.nodes)):
                node = self.nodes[node_index]
                downstream_nodes_delta_weight_vectors = []
                for downstream_node_index in range(0, no_of_downstream_nodes):
                    downstream_nodes_delta_weight_vectors.append([downstream_layer_nodes_deltas[downstream_node_index], downstream_layer_nodes_weight_vectors[downstream_node_index][node_index]])
                
                node.compute_delta(downstream_nodes_delta_weight_vectors)
                
            downstream_layer_nodes_deltas = []
            downstream_layer_nodes_weight_vectors = []
            for node in self.nodes:
                delta = node.get_delta()
                node_weights = node.get_weights()
                #print('\n \n ', node_weights, '\n')
                node.update_weights()
                #print(node.get_weights())
                downstream_layer_nodes_deltas.append(delta)
                downstream_layer_nodes_weight_vectors.append(node_weights[1:])
            if(self.upstream_layer != ''):
                self.upstream_layer.back_propagate_errors(downstream_layer_nodes_deltas, downstream_layer_nodes_weight_vectors)
                
class InputLayer(Layer):
    def __init__(self, layer_config):
        super().__init__(layer_config)
        self.training_dataset = layer_config.training_dataset
        self.testing_dataset = layer_config.testing_dataset
        self.validation_dataset = layer_config.validation_dataset
        self.no_of_nodes = len(self.training_dataset.attributes) + 1
        
    def build_layer(self):
        zipped_attributes = zip(self.training_dataset.attributes, self.testing_dataset.attributes, self.validation_dataset.attributes)
        self.nodes = [InputNode(attributes_set) for attributes_set in zipped_attributes]
    
    def set_training_dataset(self, training_dataset):
        self.layer_config.training_dataset = training_dataset
        self.training_dataset = training_dataset
        result = [node.set_training_attribute(training_attribute) for node, training_attribute in zip(self.nodes, self.training_dataset.attributes)]
    
    def set_validation_dataset(self, validation_dataset):
        self.layer_config.validation_dataset = validation_dataset
        self.validation_dataset = validation_dataset
        result = [node.set_validation_attribute(validation_attribute) for node, validation_attribute in zip(self.nodes, self.validation_dataset.attributes)]
        
    def set_testing_dataset(self, testing_dataset):
        self.layer_config.testing_dataset = testing_dataset
        self.testing_dataset = testing_dataset
        result = [node.set_testing_attribute(testing_attribute) for node, testing_attribute in zip(self.nodes, self.testing_dataset.attributes)]
    
    def get_current_dataset(self):
        if(self.mode == 'training'):
            return self.get_training_dataset()
        elif(self.mode == 'validation'):
            return self.get_validation_dataset()
        elif(self.mode == 'testing'):
            return self.get_testing_dataset()
    
    def get_training_dataset(self):
        return self.training_dataset
    
    def get_validation_dataset(self):
        return self.validation_datset
    
    def get_testing_dataset(self):
        return self.testing_dataset
    
    def set_input_layer_mode(self, mode):
        self.mode = mode
        for node in self.nodes:
            node.set_mode(mode)
    
    def initialize_nodes_points_counters(self):
        for node in self.nodes:
            node.initialize_points_counter()
    
    def reset_nodes_points_counters(self):
        for node in self.nodes:
            node.reset_points_counter()
    
    def initialize_layer(self):
        self.initialize_nodes_points_counters()
    
    def next_point_available(self):
            return self.nodes[0].next_point_available()
    
    def move_to_next_point(self):
        for node in self.nodes:
            node.move_to_next_point()
        
class OutputLayer(Layer):
    def __init__(self, layer_config, upstream_layer ):
        super().__init__(layer_config, upstream_layer)
        self.training_dataset = layer_config.training_dataset
        self.testing_dataset = layer_config.testing_dataset
        self.validation_dataset = layer_config.validation_dataset
        self.training_error = ''
        self.curr_training_error = ''
        self.validation_error = ''
        self.curr_validation_error = ''
        self.testing_error = ''
        self.curr_testing_error = ''
        self.training_confusion_matrix = ''
        self.validation_confusion_matrix = ''
        self.testing_confusion_matrix = ''
        
        label = layer_config.training_dataset.label
        self.threshold = layer_config.threshold
        if(isinstance(label, CategoricalLabel)):
            if(label.multi_label):
                self.no_of_nodes = label.label_vector_size
            else:
                self.no_of_nodes = 1
        else:
            self.no_of_nodes = 1
    
    def set_threshold(self, threshold):
        self.threshold = threshold
        
    def build_layer(self):
        labels = (self.training_dataset.label, self.validation_dataset.label, self.testing_dataset.label)
        if(self.no_of_nodes == 1):
            self.nodes = [OutputNode(self.upstream_layer.get_no_of_layer_nodes(), labels, False) for i in range(0, self.no_of_nodes)]
        elif(self.no_of_nodes > 1):
            self.nodes = []
            for node_index in range(0, self.no_of_nodes):
                labels_from_vectors = (self.training_dataset.label.get_single_label_from_label_vector(node_index), self.validation_dataset.label.get_single_label_from_label_vector(node_index), self.testing_dataset.label.get_single_label_from_label_vector(node_index))
                output_node = OutputNode(self.upstream_layer.get_no_of_layer_nodes(), labels, True, labels_from_vectors)
                self.nodes.append(output_node)
        
    def set_training_dataset(self, training_dataset):
        self.layer_config.training_dataset = training_dataset
        self.training_dataset = training_dataset
        training_label = self.training_dataset.label
        if(self.training_dataset.label.multi_label):
            for node_index in range(0, self.no_of_nodes):
                training_label_from_label_vector = self.training_dataset.label.get_single_label_from_label_vector(node_index)
                self.nodes[node_index].set_training_label(training_label, training_label_from_label_vector)
        else:
            self.nodes[0].set_training_label(training_label)
        
        self.reset_training_error_tracker()
    
    def set_validation_dataset(self, validation_dataset):
        self.layer_config.validation_dataset = validation_dataset
        self.validation_dataset = validation_dataset
        validation_label = self.validation_dataset.label
        if(self.validation_dataset.label.multi_label):
            for node_index in range(0, self.no_of_nodes):
                validation_label_from_label_vector = self.validation_dataset.label.get_single_label_from_label_vector(node_index)
                self.nodes[node_index].set_validation_label(validation_label, validation_label_from_label_vector)
        else:
            self.nodes[0].set_validation_label(validation_label)
        
        self.reset_validation_error_tracker()
            
    def set_testing_dataset(self, testing_dataset):
        self.layer_config.testing_dataset = testing_dataset
        self.testing_dataset = testing_dataset
        testing_label = self.testing_dataset.label
        if(self.testing_dataset.label.multi_label):
            for node_index in range(0, self.no_of_nodes):
                testing_label_from_label_vector = self.testing_dataset.label.get_single_label_from_label_vector(node_index)
                self.nodes[node_index].set_testing_label(testing_label, testing_label_from_label_vector)
        else:
            self.nodes[0].set_testing_label(testing_label)
            
        self.reset_testing_error_tracker()
            
    def set_output_layer_mode(self, mode):
        self.mode = mode
        for node in self.nodes:
            node.set_mode(mode)
    
    def initialize_nodes_expected_label_index_counters(self):
        for node in self.nodes:
            node.initialize_expected_label_index_counter()
    
    def reset_nodes_expected_label_index_counters(self):
        for node in self.nodes:
            node.reset_expected_label_index_counter()
            
    def initialize_layer(self):
        self.initialize_nodes_expected_label_index_counters()
        self.initialize_node_weights()
    
    def initialize_node_weights(self):
        for node in self.nodes:
            node.set_random_weights()
    
    def set_eta(self, eta):
        for node in self.nodes:
            node.set_eta(eta)
            
    def compute_layer_weights_change(self):
        self.compute_layer_nodes_prev_cur_weights_euclidean_distance_sum()
        
    def compute_layer_nodes_prev_cur_weights_euclidean_distance_sum(self):
        self.layer_nodes_prev_cur_weights_euclidean_distance_sum =  sum([node.get_prev_cur_weights_euclidean_distance() for node in self.nodes])
        self.layer_weights_change = self.layer_nodes_prev_cur_weights_euclidean_distance_sum
        
    def get_layer_weights_change(self):
        return self.layer_weights_change
    
    def initialize_error_tracker(self):
        if(self.mode == 'training'):
            self.training_error = 0
            self.curr_training_error = False
            self.training_confusion_matrix = self.get_initial_confusion_matrix()
        elif(self.mode == 'validation'):
            self.validation_error = 0
            self.curr_validation_error = False
            self.validation_confusion_matrix = self.get_initial_confusion_matrix()
        elif(self.mode == 'testing'):
            self.testing_error = 0
            self.curr_testing_error = False
            self.testing_confusion_matrix = self.get_initial_confusion_matrix()
    
    def reset_error_tracker(self):
        if(self.mode == 'training'):
            self.reset_training_error_tracker()
        elif(self.mode == 'validation'):
            self.reset_validation_error_tracker()
        elif(self.mode == 'testing'):
            self.reset_testing_error_tracker()
    
    def reset_training_error_tracker(self):
        self.training_error = ''
        self.curr_training_error = ''
        self.training_confusion_matrix = ''
        
    def reset_validation_error_tracker(self):
        self.validation_error = ''
        self.curr_validation_error = ''
        self.validation_confusion_matrix = ''
            
    def reset_testing_error_tracker(self):
        self.testing_error = ''
        self.curr_testing_error = ''
        self.testing_confusion_matrix = ''
    
    def update_error(self):
        self.compute_error()
    
    def compute_error(self):
        if(self.mode == 'training'):
            self.compute_training_error()
        elif(self.mode == 'validation'):
            self.compute_validation_error()
        elif(self.mode == 'testing'):
            self.compute_testing_error()
    
    def compute_training_error(self):
        e = False
        if(len(self.nodes) > 1):
            node_expected_labels = [node.get_current_expected_label() for node in self.nodes]
            node_outputs = [node.get_output() for node in self.nodes]
            predicted_label = node_outputs.index(max(node_outputs))
            actual_label = node_expected_labels.index(max(node_expected_labels))
            if(node_expected_labels[node_outputs.index(max(node_outputs))] != 1):
                e = True
                self.training_error += 1
                self.current_training_error = 1
                self.training_confusion_matrix[actual_label][predicted_label] += 1
            else:
                self.training_confusion_matrix[actual_label][actual_label] += 1
                
            '''
            if(e):
                print(node_expected_labels, node_outputs, -1)
            else:
                print(node_expected_labels, node_outputs)'''
        else:
            node_expected_label = self.nodes[0].get_current_expected_label()
            node_output = 1 if self.nodes[0].get_output() > 0.5 else 0
            if(node_output != node_expected_label):
                e = True
                self.training_error +=1
                self.current_training_error = 1
                self.training_confusion_matrix[node_expected_label][node_output] += 1
            else:
                self.training_confusion_matrix[node_expected_label][node_expected_label] += 1
            '''
            if(e):
                print(node_expected_label, node_output, -1)
            else:
                print(node_expected_label, node_output)'''
    
    def compute_validation_error(self):
        if(len(self.nodes) > 1):
            node_expected_labels = [node.get_current_expected_label() for node in self.nodes]
            node_outputs = [node.get_output() for node in self.nodes]
            predicted_label = node_outputs.index(max(node_outputs))
            actual_label = node_expected_labels.index(max(node_expected_labels))
            if(node_expected_labels[node_outputs.index(max(node_outputs))] != 1):
                self.validation_error += 1
                self.current_validation_error = 1
                self.validation_confusion_matrix[actual_label][predicted_label] += 1
            else:
                self.validation_confusion_matrix[actual_label][actual_label] += 1    
        else:
            node_expected_label = self.nodes[0].get_current_expected_label()
            node_output = 1 if self.nodes[0].get_output() > self.threshold else 0
            if(node_output != node_expected_label):
                self.validation_error +=1
                self.current_validation_error = 1
                self.validation_confusion_matrix[node_expected_label][node_output] += 1
            else:
                self.validation_confusion_matrix[node_expected_label][node_expected_label] += 1
    
    def compute_testing_error(self):
        if(len(self.nodes) > 1):
            node_expected_labels = [node.get_current_expected_label() for node in self.nodes]
            node_outputs = [node.get_output() for node in self.nodes]
            predicted_label = node_outputs.index(max(node_outputs))
            actual_label = node_expected_labels.index(max(node_expected_labels))
            e = False
            if(node_expected_labels[node_outputs.index(max(node_outputs))] != 1):
                self.testing_error += 1
                self.current_testing_error = 1
                e = True
                self.testing_confusion_matrix[actual_label][predicted_label] += 1
            else:
                self.testing_confusion_matrix[actual_label][actual_label] += 1
            '''
            if(e):
                print(node_expected_labels, node_outputs)
            else:
                print(node_expected_labels, node_outputs, -1)'''
        else:
            node_expected_label = self.nodes[0].get_current_expected_label()
            node_output = 1 if self.nodes[0].get_output() > self.threshold else 0
            #print(self.nodes[0].get_output(), node_output, self.threshold)
            if(node_output != node_expected_label):
                self.testing_error +=1
                self.current_testing_error = 1
                self.testing_confusion_matrix[node_expected_label][node_output] += 1
            else:
                self.testing_confusion_matrix[node_expected_label][node_expected_label] += 1
        
    def get_error(self):
        if(self.mode == 'training'):
            return self.get_training_error()
        elif(self.mode == 'validation'):
            return self.get_validation_error()
        elif(self.mode == 'testing'):
            return self.get_testing_error()
    
    def get_training_error(self):
        no_of_training_points = self.nodes[0].current_expected_training_label_index + 1
        return self.training_error/no_of_training_points
    
    def get_validation_error(self):
        no_of_validation_points = self.nodes[0].current_expected_validation_label_index + 1
        return self.validation_error/no_of_validation_points
    
    def get_testing_error(self):
        no_of_testing_points = self.nodes[0].current_expected_testing_label_index + 1
        return self.testing_error /no_of_testing_points
    
    def move_to_next_expected_label(self):
        for node in self.nodes:
            node.move_to_next_expected_label()
    
    def get_initial_confusion_matrix(self):
        size = 2 if self.no_of_nodes == 1 else self.no_of_nodes
        confusion_matrix = [[ 0 for j in range(0, size)] for i in range(0,size)]
        #print(confusion_matrix)
        return confusion_matrix
    
    def get_confusion_matrix(self):
        if(self.mode == 'training'):
            return self.training_confusion_matrix
        elif(self.mode == 'validation'):
            return self.validation_confusion_matrix
        elif(self.mode == 'testing'):
            return self.testing_confusion_matrix
    
    def get_layer_nodes_weight_vectors(self):
        layer_nodes_weight_vectors = []
        for node in self.nodes:
            layer_nodes_weight_vectors.append(node.get_weights())
        return layer_nodes_weight_vectors
    
class HiddenLayer(Layer):
    def __init__(self, layer_config, upstream_layer):
        super().__init__(layer_config, upstream_layer)
    
    def build_layer(self):
        self.nodes = [HiddenNode(self.upstream_layer.get_no_of_layer_nodes()) for i in range(0, self.layer_config.no_of_nodes)]
    
    def initialize_layer(self):
        self.initialize_node_weights()
    
    def initialize_node_weights(self):
        for node in self.nodes:
            node.set_random_weights()
    
    def set_eta(self, eta):
        for node in self.nodes:
            node.set_eta(eta)
    
    def compute_layer_weights_change(self):
        self.compute_layer_nodes_prev_cur_weights_euclidean_distance_sum()
        
    def compute_layer_nodes_prev_cur_weights_euclidean_distance_sum(self):
        #print([node.get_prev_cur_weights_euclidean_distance() for node in self.nodes])
        self.layer_nodes_prev_cur_weights_euclidean_distance_sum =  np.sum([node.get_prev_cur_weights_euclidean_distance() for node in self.nodes])
        self.layer_weights_change = self.layer_nodes_prev_cur_weights_euclidean_distance_sum
        
    def get_layer_weights_change(self):
        return self.layer_weights_change
    
    def get_layer_nodes_weight_vectors(self):
        layer_nodes_weight_vectors = []
        for node in self.nodes:
            layer_nodes_weight_vectors.append(node.get_weights())
        return layer_nodes_weight_vectors
        
class NetworkConfig:
    def __init__(self, layers_configs = []):
        self.input_layer_config_available = False
        self.output_layer_config_available = False
        self.hidden_layers_configs_size = 0
        self.layers_size = 0
        self.layers_configs = []
        result = [set_layer_config(layer_config) for layer_config in layers_configs]
                
    def set_layer_config(self, layer_config):
        if(isinstance(layer_config, InputLayerConfig)):
            self.layers_configs.insert(0,layer_config)
            self.input_layer_config_available = True
            self.layers_size += 1
            return True
        elif(isinstance(layer_config, OutputLayerConfig)):
            self.layers_configs.append(layer_config)
            self.output_layer_config_available = True
            self.layers_size += 1
            return True
        elif(isinstance(layer_config, HiddenLayerConfig)):
            self.hidden_layers_configs_size += 1
            self.layers_size += 1
            
            if(self.output_layer_config_available):
                self.layers_configs.insert(self.layers_size-1, layer_config)
            else:
                self.layers_configs.append(layer_config)
                
            return True
        else:
            return False
        
class LayerConfig:
    def __init__(self, no_of_nodes=''):
        self.no_of_nodes = no_of_nodes
    
class HiddenLayerConfig(LayerConfig):
    def __init__(self, no_of_nodes):
        super().__init__(no_of_nodes)

class InputLayerConfig(LayerConfig):
    def __init__(self, datasets):
        self.training_dataset, self.testing_dataset, self.validation_dataset = datasets
        
class OutputLayerConfig(LayerConfig):
    def __init__(self, datasets):
        self.training_dataset, self.testing_dataset, self.validation_dataset = datasets
        
if __name__ == "__main__":
    print("This is Neural Network code.. run hw2.py")
