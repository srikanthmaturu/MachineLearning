# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
from Algorithms.neuralnetwork import *
from Algorithms.naivebayes import *
from Algorithms.decisontree import *
import time

def neural_network_experiment(dataset_description, datasets,  ann_architecture, training_parameters, training_stop_criteria):
    network_config = NetworkConfig()
    network_config.training_stop_criteria = training_stop_criteria
    input_layer_config = InputLayerConfig(datasets)
    network_config.set_layer_config(input_layer_config)
    for index in range(0, ann_architecture['no_of_hidden_layers']):
        hidden_layer_config = HiddenLayerConfig(ann_architecture['no_of_nodes_per_hidden_layer'])
        network_config.set_layer_config(hidden_layer_config)
    output_layer_config = OutputLayerConfig(datasets)
    output_layer_config.threshold = 0.5
    network_config.set_layer_config(output_layer_config)
    
    neural_network = NeuralNetwork(network_config)
    neural_network.set_print_mode(True)
    annbp_training_start_time = time.process_time()
    print('Training the Neural Network....please wait as it may take a long time to complete...\n')
    neural_network.train(training_parameters['eta'])
    annbp_training_end_time = time.process_time()
    annbp_training_time = annbp_training_end_time - annbp_training_start_time
    print('ANN\'s Training time: ', annbp_training_time)
    print(neural_network.training_iterations_results)
    
    print('\nNeural network\'s training error on the training dataset at the end of training:',neural_network.training_iterations_results['training_errors'][-1])

    print('\nNeural network\'s Confusion matrix on the training dataset at the end of training\n')
    neural_network.display_confusion_matrix(neural_network.confusion_matrix)
    print('tp and fp rates for each individual class on the training dataset')
    neural_network.print_tp_fp_rates()
    #neural_network.validate()
    print('{:-<200}'.format('-'))
    print('\nTesting Neural Network on testing dataset:\n')
    annbp_testing_start_time = time.process_time()
    neural_network.test()
    annbp_testing_end_time = time.process_time()
    annbp_testing_time = annbp_testing_end_time - annbp_testing_start_time
    print('ANN\'s Testing time: ', annbp_testing_time)
    print('\nNeural network\'s testing error on the testing dataset is:',neural_network.get_error())
    print('')
    print('Neural network\'s Confusion matrix on the testing dataset\n')
    neural_network.display_confusion_matrix()
    print('tp and fp rates for each individual class on testing dataset')
    print('')
    neural_network.print_tp_fp_rates()
    neural_network.print_95_percent_confidence_interval()
    print('')
    neural_network.print_network_information()
    return neural_network.get_error()

def naive_bayes_experiment(dataset_description, datasets, m):
    training_dataset, testing_dataset, validation_dataset = datasets
    plot_mode = False
    if(plot_mode):
        print('{:-<200}'.format('-'))
        print('\nPrinting training, validation and testing datasets: \n')
        print('Printing Training Dataset:')
        training_dataset.print_dataset()
        print('{:-<200}'.format('-'))
        print('Printing Testing Dataset: ')
        testing_dataset.print_dataset()
    
    naive_bayes = NaiveBayes(training_dataset, testing_dataset, m)
    print('\nInput training dataset is ', dataset_description['datasetname'])
    print('----Training size: ', training_dataset.size)
    print('----Testing size: ', testing_dataset.size)
    print('')
    print('pseudo count m = ', m)
    print('{:-<200}'.format('-'))
    print('Training NaiveBayes....')
    naive_bayes.train()
    print('Training complete....')
    naive_bayes.test(training_dataset)
    naive_bayes.print_error()
    print('Confusion matrix: \n')
    naive_bayes.display_confusion_matrix()    
    print('\nPrinting tp and fp rates for each class:')
    naive_bayes.print_tp_fp_rates()
    print('')
    print('{:-<200}'.format('-')) 
    print('Testing using testing dataset...')
    naive_bayes.test(testing_dataset)
    naive_bayes.print_error()
    naive_bayes.print_95_percent_confidence_interval()
    print('\nConfusion matrix: \n')
    naive_bayes.display_confusion_matrix()    
    print('\nPrinting tp and fp rates for each class:')
    naive_bayes.print_tp_fp_rates()
    print('\nPrinting prior probabilities: ')
    naive_bayes.print_prior_probabilities()


def decison_tree_experiment(dataset_description, datasets):
    print('')
    print('{:*<200}'.format('*'))
    print('{:*<200}'.format('*'))
    print('')
    
    discretized_training_dataset, discretized_testing_dataset, discretized_validation_dataset = datasets
    
    id3 = ID3(discretized_training_dataset, discretized_testing_dataset, 0.2)
    print('')
    print('Training ID3...\n')
    id3_training_start_time = time.process_time()
    id3.train()
    id3_trainng_end_time = time.process_time() 
    id3_time = id3_trainng_end_time - id3_training_start_time
    print('ID3\'s Training time: ', id3_time)
    
    
    id3.set_testing_dataset(discretized_training_dataset)
    nu_error = id3.test()
    print('\nID3\'s Classification error: ', nu_error)
    print('\nPrinting Confusion matrix: \n')
    id3.compute_confusion_matrix()
    id3.display_confusion_matrix()
    id3.compute_tp_fp_rates()
    id3_tp_fp_rates = id3.get_tp_fp_rates()
    print('\nPrinting tp and fp rates for each class:\n')
    id3.print_tp_fp_rates()
    print('{:-<200}'.format('-'))
    print('\nTesting on testing dataset: \n')
    id3.set_testing_dataset(discretized_testing_dataset)
    id3_testing_start_time = time.process_time()
    nu_error = id3.test()
    id3_testing_end_time = time.process_time() 
    id3_testing_time = id3_testing_end_time - id3_testing_start_time
    print('ID3\'s Testing time: ', id3_testing_time)
    print('\nID3\'s Classification error: ', nu_error)
    id3.print_95_percent_confidence_interval()
    print('\nConfusion matrix: \n')
    id3.compute_confusion_matrix()
    id3.display_confusion_matrix()
    id3.compute_tp_fp_rates()
    id3_tp_fp_rates = id3.get_tp_fp_rates()
    print('\nPrinting tp and fp rates for each class:\n')
    id3.print_tp_fp_rates()

if __name__ == "__main__":
    print("Loading final project experiments code...")
