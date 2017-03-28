# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
from Algorithms.neuralnetwork import *
from Algorithms.decisontree import ID3
from DataSetFilters.datasetfilters import *
#import matplotlib.pyplot as plt
import time
def experiment1(dataset_description, dataset, eta_range, no_of_hidden_layers, no_of_nodes_per_hidden_layer, training_stop_criteria, ratios):
    training_dataset, testing_dataset, validation_dataset = split_dataset(dataset, ratios)
    neural_network_validation_dataset_size = 30
    neural_network_training_dataset_size = training_dataset.size - neural_network_validation_dataset_size
    selection_indices = [i for i in range(0, training_dataset.size)]
    neural_network_datasets = (training_dataset.subset(selection_indices[:neural_network_training_dataset_size]), testing_dataset, training_dataset.subset(selection_indices[neural_network_training_dataset_size:]))

    plt.title('varying eta_range...')
    plt.xlabel('index')
    plt.ylabel('training_error')
    errors = []
    x_values = []
    for i in range(0, len(eta_range)):
        x_values.append(eta_range[i])
        error = experiment(dataset_description, neural_network_datasets, eta_range[i], no_of_hidden_layers, no_of_nodes_per_hidden_layer, training_stop_criteria)
        print('Current setting: eta_value:-',eta_range[i],'Error: ',error)
        errors.append(error)
    
    plt.title('Dataset: '+dataset_description['datasetname']+'\nTraining Size: '+str(training_dataset.size)+'\nNeuralNetwork Architecture: '+'Number of hidden layers: '+str(no_of_hidden_layers)+', Number of hidden nodes per layer: '+str(no_of_nodes_per_hidden_layer)+'\nVarying learning rate: '+str(eta_range[0])+' to '+str(eta_range[-1]))
    
    plt.plot(x_values, errors,label='neural_network', marker='p')
    plt.xlabel('learning_rate(eta)')
    plt.ylabel('training_error')
    plt.show()

def experiment2(dataset_description, dataset, eta, no_of_hidden_layers_range, no_of_nodes_per_hidden_layer, training_stop_criteria, ratios):
    training_dataset, testing_dataset, validation_dataset = split_dataset(dataset, ratios)
    neural_network_validation_dataset_size = 30
    neural_network_training_dataset_size = training_dataset.size - neural_network_validation_dataset_size
    selection_indices = [i for i in range(0, training_dataset.size)]
    neural_network_datasets = (training_dataset.subset(selection_indices[:neural_network_training_dataset_size]), testing_dataset, training_dataset.subset(selection_indices[neural_network_training_dataset_size:]))

    print('Experiment 2: varying no of hidden layers: [',no_of_hidden_layers_range[0],',',no_of_hidden_layers_range[-1],']')
    plt.title('varying number of layers ....')
    plt.xlabel('index')
    plt.ylabel('training_error')
    errors = []
    x_values = []
    for i in range(0, len(no_of_hidden_layers_range)):
        print('Current setting: no_of_hidden_layers: ', no_of_hidden_layers_range[i])
        x_values.append(i)
        error = experiment(dataset_description, neural_network_datasets, eta, no_of_hidden_layers_range[i], no_of_nodes_per_hidden_layer, training_stop_criteria)
        errors.append(error)
    
    plt.title('Dataset: '+dataset_description['datasetname']+'\nTraining Size: '+str(training_dataset.size)+'\nNeuralNetwork Architecture: '+'Learning rate: '+str(eta)+', Number of hidden nodes per layer: '+str(no_of_nodes_per_hidden_layer)+'\nVarying number of hidden layers: '+str(no_of_hidden_layers_range[0])+' to '+str(no_of_hidden_layers_range[-1]))
    plt.plot(x_values, errors,label='neural_network', marker='p')
    plt.xlabel('learning_rate(eta)')
    plt.ylabel('training_error')
    plt.plot(x_values, errors)
    plt.show()

def experiment3(dataset_description, dataset, eta, no_of_hidden_layers, no_of_nodes_per_hidden_layer_range, training_stop_criteria, ratios):
    training_dataset, testing_dataset, validation_dataset = split_dataset(dataset, ratios)
    neural_network_validation_dataset_size = 30
    neural_network_training_dataset_size = training_dataset.size - neural_network_validation_dataset_size
    selection_indices = [i for i in range(0, training_dataset.size)]
    neural_network_datasets = (training_dataset.subset(selection_indices[:neural_network_training_dataset_size]), testing_dataset, training_dataset.subset(selection_indices[neural_network_training_dataset_size:]))

    print('Experiment 3: varying no of nodes per hidden layer: [',no_of_nodes_per_hidden_layer_range[0],',',no_of_nodes_per_hidden_layer_range[-1],']')
    plt.title('Number of nodes per hidden layer testing..')
    plt.xlabel('index')
    plt.ylabel('training_error')
    errors = []
    x_values = []
    for i in range(0, len(no_of_nodes_per_hidden_layer_range)):
        print('Current setting: no_of_nodes_per_hidden_layer: ', no_of_nodes_per_hidden_layer_range[i])
        x_values.append(i)
        error = experiment(dataset_description, neural_network_datasets, eta, no_of_hidden_layers, no_of_nodes_per_hidden_layer_range[i], training_stop_criteria)
        errors.append(error)
        
    plt.title('Dataset: '+dataset_description['datasetname']+'\nTraining Size: '+str(training_dataset.size)+'\nNeuralNetwork Architecture: '+'Learning rate: '+str(eta)+', Number of hidden layers: '+str(no_of_hidden_layers)+'\nVarying number of hidden nodes per layer: '+str(no_of_nodes_per_hidden_layer_range[0])+' to '+str(no_of_nodes_per_hidden_layer_range[-1]))
    plt.plot(x_values, errors,label='neural_network', marker='p')
    plt.xlabel('learning_rate(eta)')
    plt.ylabel('training_error')    
    plt.plot(x_values, errors)
    plt.show()

def compare_neural_network_and_decison_tree(dataset_description, dataset,  eta, no_of_hidden_layers, no_of_nodes_per_hidden_layer, training_stop_criteria, ratios):
    print('Comparison between neural network and decison tree over 100 iterations on the input dataset:')
    plt.title('Artificical Neural Network and ID3 comparison\n'+'Dataset: '+dataset_description['datasetname']+'\nTraining Size: '+str(dataset_description['dataset_sizes']['training_size'])+'\nTesting Size: '+str(dataset_description['dataset_sizes']['testing_size'])+'\nNeuralNetwork Architecture: '+'Learning rate: '+str(eta)+', Number of hidden layers: '+str(no_of_hidden_layers)+' Number of hidden nodes per layer: '+str(no_of_nodes_per_hidden_layer))
    plt.xlabel('iteration number')
    plt.ylabel('testing_error')
    ann_errors = []
    id3_errors = []
    x_values = []
    for i in range(0, 50):
        training_dataset, testing_dataset, validation_dataset = split_dataset(dataset, ratios)
        datasets = (training_dataset, testing_dataset, validation_dataset)
        x_values.append(i)
        ann_error = (experiment(dataset_description, datasets, eta, no_of_hidden_layers, no_of_nodes_per_hidden_layer, training_stop_criteria))
        ann_errors.append(ann_error)
        discretized_training_dataset = discretize_the_dataset(datasets[0])
        discretized_testing_dataset = discretize_the_dataset(datasets[1])
        id3 = ID3(discretized_training_dataset, discretized_testing_dataset, 0.2)
        id3.train()
        id3_error = (id3.test(discretized_testing_dataset, id3.decison_tree))
        id3_errors.append(id3_error)
        print('ANN error: ', ann_error, ' ID3 error:', id3_error)
    plt.plot(x_values, ann_errors,label='Artificial Neural Network', marker='o')
    plt.plot(x_values, id3_errors,label='ID3', marker='p')
    plt.legend()
    plt.show()

def compare_neural_network_and_decison_tree_using_cross_validation(dataset_description, dataset,  eta, no_of_hidden_layers, no_of_nodes_per_hidden_layer, training_stop_criteria, no_of_folds):
    print('Comparison between neural network and decison tree using k-fold cross validation on the input dataset k is: '+str(no_of_folds)+' : is ANN better than ID3?')
    ann_errors = []
    id3_errors = []
    x_values = []
    indices = [i for i in range(0, dataset.size)]
    shuffle(indices)
    split_size = int(len(indices)/no_of_folds)
    selections = []
    
    for i in range(0,no_of_folds):
        selections.append(indices[i*split_size:(i+1)*split_size-1])
        
    for i in range(0, len(selections)):
        testing_dataset = dataset.subset(selections[i])
        training_selection = []
        for j in range(0, len(selections)):
            if(i!=j):
                training_selection += selections[j]
        id3_training_dataset = dataset.subset(training_selection)
        ann_training_dataset = dataset.subset(training_selection[30:])
        ann_validation_dataset = dataset.subset(training_selection[:30])
        ann_datasets = (ann_training_dataset, testing_dataset, ann_validation_dataset)
        x_values.append(i)
        ann_error = (experiment(dataset_description, ann_datasets, eta, no_of_hidden_layers, no_of_nodes_per_hidden_layer, training_stop_criteria))
        ann_errors.append(ann_error)
        discretized_training_dataset = discretize_the_dataset(id3_training_dataset)
        discretized_testing_dataset = discretize_the_dataset(testing_dataset)
        id3 = ID3(discretized_training_dataset, discretized_testing_dataset, 0.2)
        id3.train()
        id3_error = (id3.test(discretized_testing_dataset, id3.decison_tree))
        id3_errors.append(id3_error)
        print('ANN error: ', ann_error, ' ID3 error:', id3_error)
    
    neural_network_error_differences = [ann_error - id3_error for ann_error, id3_error in zip(ann_errors, id3_errors)]
    id3_error_differences = [id3_error - ann_error for ann_error, id3_error in zip(ann_errors, id3_errors)]
    
    neural_network_mean_error_difference = mean(neural_network_error_differences)
    id3_mean_error_difference = mean(id3_error_differences)
    
    constant = 1/(no_of_folds*(no_of_folds-1))
    
    neural_network_sp_value = (constant*sum([((neural_network_mean_error_difference-error_difference)**2) for error_difference in neural_network_error_differences]))**0.5
    id3_sp_value = (constant*sum([((id3_mean_error_difference-error_difference)**2) for error_difference in id3_error_differences]))**0.5
    
    neural_network_expected_error_difference = neural_network_mean_error_difference + 1.812*neural_network_sp_value
    id3_expected_error_difference = id3_mean_error_difference + 1.812*id3_sp_value
    
    plt.title('Artificial Neural Network and ID3 K-fold cross validation comparison k value is '+str(no_of_folds)+' \nwith approximately 95% probability, true difference of expected error between ANN and ID3 is atmost '+str(neural_network_expected_error_difference)+'\n'+'Dataset: '+dataset_description['datasetname']+'\nTotal Dataset Size: '+str(dataset.size)+'\n One fold size: '+str(split_size)+'\nNeuralNetwork Architecture: '+'Learning rate: '+str(eta)+', Number of hidden layers: '+str(no_of_hidden_layers)+', Number of hidden nodes per layer: '+str(no_of_nodes_per_hidden_layer))
    plt.plot(x_values, ann_errors, label='Artificial Neural Network', marker='o')
    plt.plot(x_values, id3_errors ,label='ID3', marker='p')
    plt.xlabel('Iteration number')
    plt.ylabel('testing_error')
    plt.legend()
    plt.show()
    
    plt.title('ID3 and Artificial Neural Network K-fold cross validation comparison k value is '+str(no_of_folds)+' \nwith approximately 95% probability, true difference of expected error between ID3 and ANN is atmost '+str(id3_expected_error_difference)+'\n'+'Dataset: '+dataset_description['datasetname']+'\nTotal Dataset Size: '+str(dataset.size)+'\n One fold size: '+str(split_size)+'\nNeuralNetwork Architecture: '+'Learning rate: '+str(eta)+', Number of hidden layers: '+str(no_of_hidden_layers)+', Number of hidden nodes per layer: '+str(no_of_nodes_per_hidden_layer))
    plt.plot(x_values, ann_errors, label='Artificial Neural Network', marker='o')
    plt.plot(x_values, id3_errors ,label='ID3', marker='p')
    plt.xlabel('Iteration number')
    plt.ylabel('testing_error')
    plt.legend()
    plt.show()
    

def experiment(dataset_description, datasets,  eta, no_of_hidden_layers, no_of_nodes_per_hidden_layer, training_stop_criteria):
    network_config = NetworkConfig()
    network_config.training_stop_criteria = training_stop_criteria
    input_layer_config = InputLayerConfig(datasets)
    network_config.set_layer_config(input_layer_config)
    for index in range(0, no_of_hidden_layers):
        hidden_layer_config = HiddenLayerConfig(no_of_nodes_per_hidden_layer)
        network_config.set_layer_config(hidden_layer_config)
    output_layer_config = OutputLayerConfig(datasets)
    output_layer_config.threshold = 0.5
    network_config.set_layer_config(output_layer_config)
    
    neural_network = NeuralNetwork(network_config)
    neural_network.set_print_mode(False)
    neural_network.train(eta)
    neural_network.test()
    return neural_network.get_error()

def main_experiment(dataset_description, dataset,  eta, no_of_hidden_layers, no_of_nodes_per_hidden_layer, training_stop_criteria, ratios, plot_mode):
    training_dataset, testing_dataset, validation_dataset = split_dataset(dataset, ratios)
    id3_datasets = (training_dataset, testing_dataset, validation_dataset)
    datasets = id3_datasets
    
    neural_network_validation_dataset_size = 30
    neural_network_training_dataset_size = training_dataset.size - neural_network_validation_dataset_size
    selection_indices = [i for i in range(0, training_dataset.size)]
    neural_network_datasets = (training_dataset.subset(selection_indices[:neural_network_training_dataset_size]), testing_dataset, training_dataset.subset(selection_indices[neural_network_training_dataset_size:]))
    
    if(plot_mode):
        print('{:-<200}'.format('-'))
        print('Printing Training Dataset:')
        neural_network_datasets[0].print_dataset()
        print('{:-<200}'.format('-'))
        print('Printing Validation Dataset:')
        neural_network_datasets[2].print_dataset()
        print('{:-<200}'.format('-'))
        print('Printing Testing Dataset: ')
        datasets[1].print_dataset()
    dataset_description['dataset_sizes'] = {'training_size':training_dataset.size, 'validation_size':validation_dataset.size, 'testing_size':testing_dataset.size}
    print('Size of training dataset: ', int(training_dataset.size), '\nSize of testing dataset: ', int(testing_dataset.size))
    
    print('{:-<200}'.format('-'))
    
    network_config = NetworkConfig()
    network_config.training_stop_criteria = training_stop_criteria
    input_layer_config = InputLayerConfig(neural_network_datasets)
    network_config.set_layer_config(input_layer_config)
    for index in range(0, no_of_hidden_layers):
        hidden_layer_config = HiddenLayerConfig(no_of_nodes_per_hidden_layer)
        network_config.set_layer_config(hidden_layer_config)
    output_layer_config = OutputLayerConfig(neural_network_datasets)
    output_layer_config.threshold = 0.5
    network_config.set_layer_config(output_layer_config)
    
    neural_network = NeuralNetwork(network_config)
    neural_network.set_print_mode(False)
    annbp_training_start_time = time.process_time()
    print('Training the Neural Network....please wait as it may take a long time to complete...')
    neural_network.train(eta)
    annbp_training_end_time = time.process_time()
    annbp_training_time = annbp_training_end_time - annbp_training_start_time
    print('ANN\'s Training time: ', annbp_training_time)
    #print(neural_network.training_iterations_results)
    no_of_high_level_iterations = len(neural_network.training_iterations_results['training_iterations'])
    no_of_returned_validation_errors = len(neural_network.training_iterations_results['validation_errors'])
    valid_plot= True
    if(no_of_returned_validation_errors == no_of_high_level_iterations):
        skip_last_iteration = False
    else:
        skip_last_iteration = True
        
    if(no_of_returned_validation_errors == 0):
        valid_plot = False
    
    if(valid_plot):
        subplots = 3
    else:
        subplots = 2
    
    if(plot_mode):
        plt.subplot('1'+str(subplots)+'1')
        plt.title('Training error of each training iteration\n'+' Dataset: '+dataset_description['datasetname']+'\n Training size: '+str(neural_network_datasets[0].size))
        plt.xlabel('iteration number')
        plt.ylabel('error')
        plt.plot([i for i in range(1, no_of_high_level_iterations+1)],neural_network.training_iterations_results['training_errors'],label='training error', marker='o')
        plt.legend()
        plt.subplot('1'+str(subplots)+'2')
        plt.title('Training time of each training iteration\n'+' Dataset: '+dataset_description['datasetname']+'\n Training size: '+str(neural_network_datasets[0].size))
        plt.xlabel('iteration number')
        plt.ylabel('seconds')
        plt.plot([i for i in range(1, no_of_high_level_iterations+1)], neural_network.training_iterations_results['training_times'],label='training times', marker='^')
        plt.legend()
        if(valid_plot):
            plt.subplot('1'+str(subplots)+'3')
            plt.title('Validation error of each iteration\n'+' Dataset: '+dataset_description['datasetname']+'\n Training size: '+str(neural_network_datasets[0].size)+'\n Validation size: '+str(neural_network_datasets[2].size))
            plt.xlabel('iteration number')
            plt.ylabel('error')
            if(skip_last_iteration):
                x_values = [i for i in range(1, no_of_high_level_iterations)]
            else:
                x_values = [i for i in range(1, no_of_high_level_iterations+1)]
            
            plt.plot(x_values,neural_network.training_iterations_results['validation_errors'],label='validation error', marker='p')
            plt.legend()
        plt.show()
    print('{:-<200}'.format('-'))
    print('Neural network\'s training error on the training dataset at the end of training:',neural_network.training_iterations_results['training_errors'][-1])
    print('')
    print('Neural network\'s Confusion matrix on the training dataset at the end of training')
    neural_network.display_confusion_matrix(neural_network.confusion_matrix)
    #print('tp and fp rates for each individual class on the training dataset')
    #neural_network.print_tp_fp_rates()
    #neural_network.validate()
    print('{:-<200}'.format('-'))
    annbp_testing_start_time = time.process_time()
    neural_network.test()
    annbp_testing_end_time = time.process_time()
    annbp_testing_time = annbp_testing_end_time - annbp_testing_start_time
    print('ANN\'s Testing time: ', annbp_testing_time)
    print('Neural network\'s testing error on the testing dataset is:',neural_network.get_error())
    print('')
    print('Neural network\'s Confusion matrix on the testing dataset...')
    neural_network.display_confusion_matrix()
    print('tp and fp rates for each individual class on testing dataset')
    neural_network.print_tp_fp_rates()
    neural_network.print_95_percent_confidence_interval()
    print('{:-<200}'.format('-'))
    print('')
    neural_network.print_network_information()
    no_of_classes =len([key for key, value in datasets[0].label.indexed_unique_nominal_values.items()])
    label_indexes_to_names_mapping = {value:key for key, value in datasets[0].label.indexed_unique_nominal_values.items()}
    label_names = [label_indexes_to_names_mapping[key] for key in sorted(label_indexes_to_names_mapping.keys())]
    classifiers_results = {}
    tp_fp_rates = []
            
    if(no_of_classes == 2):
        print('')
        print('{:-<200}'.format('-'))
        if(plot_mode):
            print('computing ROC curve for Neural Network and Decison Tree....')
        for i in range(0, 101):
            neural_network.set_threshold(i/100)
            neural_network.test()
            #neural_network.display_confusion_matrix()
            tp_fp_rates.append(neural_network.get_tp_fp_rates())
        classifiers_results['neural_network'] = {'tp_fp_rates':tp_fp_rates}
        classifiers_results['neural_network']['marker'] = 'o'
        classifiers_results['neural_network']['datasetname'] = dataset_description['datasetname']
        discretized_training_dataset = discretize_the_dataset(datasets[0])
        discretized_testing_dataset = discretize_the_dataset(datasets[1])
        id3 = ID3(discretized_training_dataset, discretized_testing_dataset, 0.2)
        print('')
        print('Training ID3...')
        id3_training_start_time = time.process_time()
        id3.train()
        id3_trainng_end_time = time.process_time() 
        id3_time = id3_trainng_end_time - id3_training_start_time
        print('ID3\'s Training time: ', id3_time)
        id3_testing_start_time = time.process_time()
        nu_error = id3.test(discretized_testing_dataset, id3.decison_tree)
        id3_testing_end_time = time.process_time() 
        id3_testing_time = id3_testing_end_time - id3_testing_start_time
        print('ID3\'s Testing time: ', id3_testing_time)
        print('ID3\'s Classification error: ', nu_error)
        id3_tp_fp_rates = id3.decison_tree.print_tp_fp_rates(discretized_testing_dataset)
        classifiers_results['decison_tree'] = {'tp_fp_rates':[id3_tp_fp_rates]}
        print('ID3\'s tp and fp rates: ', classifiers_results['decison_tree']['tp_fp_rates'])
        classifiers_results['decison_tree']['marker'] = '^'
        classifiers_results['decison_tree']['datasetname'] = dataset_description['datasetname']
        print()
        if(plot_mode):
            print_roc_curve(classifiers_results, label_names)
    
def print_roc_curve(classifiers_results, label_names):
    
    no_of_classes = len(label_names)
    for i in range(0, no_of_classes):
        for classifier in classifiers_results.keys():
            tp_fp_rates = classifiers_results[classifier]['tp_fp_rates']
            #print(tp_fp_rates)
            x_values = [tp_fp_rate[1][i] for tp_fp_rate in tp_fp_rates]
            y_values = [tp_fp_rate[0][i] for tp_fp_rate in tp_fp_rates]
            r  = '1' +str(no_of_classes)+str(i+1)
            plt.subplot(int(r))
            plt.plot(x_values, y_values, label=classifier, marker = classifiers_results[classifier]['marker'], s='40')
        plt.xlabel('fp rate')
        plt.ylabel('tp rate')
        plt.title('ROC curve \n'+' Dataset name: '+classifiers_results[classifier]['datasetname']+'\n Class Name: '+label_names[i])
        plt.legend()
    plt.show()
    
        
if __name__ == "__main__":
    print("This is neural network experiments code. Run hw2.py..")
