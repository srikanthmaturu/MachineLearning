# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
from DataSetFilters.datasetfilters import *
from Algorithms.naivebayes import NaiveBayes
from Algorithms.neuralnetwork import *
from Algorithms.decisontree import *
import matplotlib.pyplot as plt
import time

def nayes_bayes_experiment(dataset_description, dataset, ratios, m):
    training_dataset, testing_dataset, validation_dataset = split_dataset(dataset, ratios)
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
    
def multi_classifier_roc_experiment(dataset_description, dataset, ratios):
    
    training_dataset, testing_dataset, validation_dataset = split_dataset(dataset, ratios)
    plot_mode = True
    
    no_of_classes =len([key for key, value in training_dataset.label.indexed_unique_nominal_values.items()])
    label_indexes_to_names_mapping = {value:key for key, value in training_dataset.label.indexed_unique_nominal_values.items()}
    label_names = [label_indexes_to_names_mapping[key] for key in sorted(label_indexes_to_names_mapping.keys())]
    
    print('')
    neural_network_validation_dataset_size = 30
    neural_network_training_dataset_size = training_dataset.size - neural_network_validation_dataset_size
    selection_indices = [i for i in range(0, training_dataset.size)]
    neural_network_datasets = (training_dataset.subset(selection_indices[:neural_network_training_dataset_size]), testing_dataset, training_dataset.subset(selection_indices[neural_network_training_dataset_size:]))

    if(plot_mode):
        print('{:-<200}'.format('-'))
        print('\nPrinting training, validation and testing datasets: \n')
        print('Printing Training Dataset:')
        neural_network_datasets[0].print_dataset()
        print('{:-<200}'.format('-'))
        print('Printing Validation Dataset:')
        neural_network_datasets[2].print_dataset()
        print('{:-<200}'.format('-'))
        print('Printing Testing Dataset: ')
        testing_dataset.print_dataset()
        
    tp_fp_rates = []
    print('{:-<200}'.format('-'))
    
    print('')
    print('{:*<200}'.format('*'))
    print('{:*<200}'.format('*'))
    print('')
    eta = 0.116
    no_of_hidden_layers = 1
    no_of_nodes_per_hidden_layer = 5
    network_config = NetworkConfig()
    training_stop_criteria = {'type':'weights_change', 'value':0.00005}
    training_stop_criteria['max_iterations'] = 10
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
    neural_network.set_print_mode(True)
    annbp_training_start_time = time.process_time()
    print('Training the Neural Network....please wait as it may take a long time to complete...\n')
    neural_network.train(eta)
    annbp_training_end_time = time.process_time()
    annbp_training_time = annbp_training_end_time - annbp_training_start_time
    print('ANN\'s Training time: ', annbp_training_time)
    #print(neural_network.training_iterations_results)
    
    
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
    #neural_network.print_network_information()
    
    classifiers_results = {}
    tp_fp_rates = []
    
    for i in range(0, 101):
        neural_network.set_threshold(i/100)
        neural_network.test()
            #neural_network.display_confusion_matrix()
        tp_fp_rates.append(neural_network.get_tp_fp_rates())
    classifiers_results['neural_network'] = {'tp_fp_rates':tp_fp_rates}
    classifiers_results['neural_network']['marker'] = 'o'
    classifiers_results['neural_network']['datasetname'] = dataset_description['datasetname']
    classifiers_results['neural_network']['markersize'] = 5
    discretized_training_dataset = discretize_the_dataset(training_dataset)
    discretized_testing_dataset = discretize_the_dataset(testing_dataset)
    
    print('')
    print('{:*<200}'.format('*'))
    print('{:*<200}'.format('*'))
    print('')
    
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
    classifiers_results['decison_tree'] = {'tp_fp_rates':[id3_tp_fp_rates]}
    classifiers_results['decison_tree']['marker'] = '^'
    classifiers_results['decison_tree']['datasetname'] = dataset_description['datasetname']
    classifiers_results['decison_tree']['markersize'] = 15
    print()
    
    print('')
    print('{:*<200}'.format('*'))
    print('{:*<200}'.format('*'))
    print('')
    
    naive_bayes = NaiveBayes(discretized_training_dataset, discretized_testing_dataset, 10)
    print('Training NaiveBayes....')
    print('\nm value 10\n')
    naive_bayes_training_start_time = time.process_time()
    naive_bayes.train()
    naive_bayes_training_end_time = time.process_time()
    naive_bayes_training_time = naive_bayes_training_end_time - naive_bayes_training_start_time
    print('Training complete....\n')
    print('NaiveBayes\'s Training time: ', naive_bayes_training_time)
    
    naive_bayes.test(discretized_training_dataset)
    naive_bayes.print_error()
    print('\nConfusion matrix: \n')
    naive_bayes.display_confusion_matrix()    
    print('\nPrinting tp and fp rates for each class:')
    naive_bayes.print_tp_fp_rates()
    
    print('{:-<200}'.format('-'))
    print('Testing using testing dataset...')
    naive_bayes_testing_start_time = time.process_time()
    naive_bayes.test(discretized_testing_dataset)
    naive_bayes_testing_end_time = time.process_time()
    naive_bayes_testing_time = naive_bayes_testing_end_time - naive_bayes_testing_start_time
    print('\nNaiveBayes\'s Testing time: ', naive_bayes_testing_time)
    naive_bayes.print_error()
    naive_bayes.print_95_percent_confidence_interval()
    print('\nConfusion matrix: \n')
    naive_bayes.display_confusion_matrix()
    print('\nPrinting tp and fp rates for each class:')
    naive_bayes.print_tp_fp_rates()
    
    print('\nPrinting prior probabilities: ')
    naive_bayes.print_prior_probabilities()
    
    naive_bayes_tp_fp_rates = naive_bayes.get_tp_fp_rates()
    naive_bayes_tp_fp_rates = [naive_bayes_tp_fp_rates[label_name] for label_name in label_names]
    naive_bayes_tp_fp_rates = [[naive_bayes_tp_fp_rates[j][i] for j in range(0,2) ] for i in range(0,2)]
    classifiers_results['naive_bayes'] = {'tp_fp_rates':[naive_bayes_tp_fp_rates]}
    classifiers_results['naive_bayes']['marker'] = 'p'
    classifiers_results['naive_bayes']['datasetname'] = dataset_description['datasetname']
    classifiers_results['naive_bayes']['markersize'] = 15
    print('\ncomputing ROC curve for NaiveBayes, Neural Network and Decison Tree....')
    if(plot_mode):
        #print_roc_curve(classifiers_results, label_names)
        pass
    

def print_roc_curve(classifiers_results, label_names):
    no_of_classes = len(label_names)
    for i in range(0, no_of_classes):
        for classifier in sorted(classifiers_results.keys()):
            tp_fp_rates = classifiers_results[classifier]['tp_fp_rates']
            #print(tp_fp_rates)
            x_values = [tp_fp_rate[1][i] for tp_fp_rate in tp_fp_rates]
            y_values = [tp_fp_rate[0][i] for tp_fp_rate in tp_fp_rates]
            r  = '1' +str(no_of_classes)+str(i+1)
            plt.subplot(int(r))
            plt.xlim([-0.1, 1.05])
            plt.ylim([0, 1.1])
            plt.plot(x_values, y_values, label=classifier, marker = classifiers_results[classifier]['marker'], markersize=classifiers_results[classifier]['markersize'])
        plt.xlabel('fp rate')
        plt.ylabel('tp rate')
        plt.title('ROC curve \n'+' Dataset name: '+classifiers_results[classifier]['datasetname']+'\n Class Name: '+label_names[i])
        plt.legend()
    plt.show()

def cross_validation_experiment(dataset_description, dataset, no_of_folds):
    neural_network_training_stop_criteria = {'type':'weights_change', 'value':0.00005, 'max_iterations': 10}
    print('\nDataset: '+dataset_description['datasetname']+'\n---Dataset size: '+str(dataset.size))
    models = []
    
    models.append({'name': 'naive_bayes', 'm':10, 'expected_error_differences':[], 'marker':'p'})
    models.append({'name': 'neural_network', 'eta': 0.116 ,'no_of_hidden_layers':1, 'no_of_nodes_per_hidden_layer':5, 'training_stop_criteria':neural_network_training_stop_criteria, 'expected_error_differences':[], 'marker':'o' })
    models.append({'name': 'decison_tree', 'expected_error_differences':[] , 'marker':'*'})
    
    models_errors = cross_validation(dataset, no_of_folds, models)
    
    
    for i in range(0, len(models)):
        models[i]['model_errors'] = models_errors[i]
        for j in range(i+1, len(models)):
            expected_error_differences = compute_expected_error_difference([models_errors[i], models_errors[j]], no_of_folds)
            models[i]['expected_error_differences'].append({'model_name':models[j]['name'], 'expected_error_difference':expected_error_differences[0]})
            models[j]['expected_error_differences'].append({'model_name':models[i]['name'], 'expected_error_difference':expected_error_differences[1]})                
    
    print_expected_error_differences(models, dataset_description)
    plot_error_differences(models, dataset_description, dataset, no_of_folds)

def plot_error_differences(models, dataset_description,  dataset, no_of_folds):
    split_size = int(dataset.size/no_of_folds)
    title = 'K-fold cross validation error differences. No of folds '+str(no_of_folds)+'\nDataset: '+dataset_description['datasetname']+' Dataset size: '+str(dataset.size)+'\nIn each iteration training size or 9-folds: '+str(int(9*split_size))+' testing size or one-fold: '+str(split_size)
    plt.title(title)
    plt.xlabel('Iteration number')
    plt.ylabel('testing_error')
    for model in models:
        plt.plot(model['model_errors'] ,label=model['name'], marker=model['marker'])
    plt.legend()
    plt.show()
    
def print_expected_error_differences(models, dataset_description):
    
    print('\nwith approximately 90% probability, true difference of expected error between any two models is atmost as shown in the following: \n')
    max_model_name_length = max([len(model['name']) for model in models]) + 8
    
    header_string = ('{:'+str(max_model_name_length)+'}').format(dataset_description['datasetname']+'_dataset')
    for model in models:
        header_string += ('{:'+str(max_model_name_length)+'}').format(model['name'])
    
    print(header_string)
    print('{:-<75}'.format(''))
    #print(models)
    for model in models:
        line = ('{:'+str(max_model_name_length)+'}').format(model['name'])
        for model2 in models:
            expected_error_difference = '----'
            for i in range(0, len(model['expected_error_differences'])):
                if model2['name'] == model['expected_error_differences'][i]['model_name']:
                    expected_error_difference = '{:.5f}'.format(model['expected_error_differences'][i]['expected_error_difference'])
            line = line + ('{:'+str(max_model_name_length)+'}').format(str(expected_error_difference))
        print(line)
    print('')
    print('Neural Network information: \n---No of hidden layers:1\n---No of hidden nodes per layer:5\n---learning_rate:0.116')
    print('')
    print('Naivebayes information: \n---m value:10')
    
        
def cross_validation(dataset, no_of_folds, models):
    cross_validation_datasets = get_cross_validation_datasets(dataset, no_of_folds)
    models_errors = []
    for model in models:
        models_errors.append(model_experiments_for_cross_validation(cross_validation_datasets, model))
    return models_errors
    
def compute_expected_error_difference(models_errors, no_of_folds):
    constant = 1/(no_of_folds*(no_of_folds-1))
    
    models_forward_error_differences = [models_errors[0][i] - models_errors[1][i] for i in range(0, len(models_errors[0]))]
    models_forward_error_differences_mean = mean(models_forward_error_differences)
    model1_model2_sp_value = (constant*sum([((models_forward_error_differences_mean-error_difference)**2) for error_difference in models_forward_error_differences]))**0.5
    model1_model2_expected_error_difference = models_forward_error_differences_mean + 1.383*model1_model2_sp_value
    
    models_backward_error_differences = [models_errors[1][i] - models_errors[0][i] for i in range(0, len(models_errors[0]))]
    models_backward_error_differences_mean = mean(models_backward_error_differences)
    model2_model1_sp_value = (constant*sum([((models_forward_error_differences_mean-error_difference)**2) for error_difference in models_backward_error_differences]))**0.5
    model2_model1_expected_error_difference = models_backward_error_differences_mean + 1.383*model2_model1_sp_value
    
    return(model1_model2_expected_error_difference, model2_model1_expected_error_difference)

def model_experiments_for_cross_validation(cross_validation_datasets, model):
    if(model['name'] == 'neural_network'):
        return neural_network_n_experiments(cross_validation_datasets, model)
    elif(model['name'] == 'decison_tree'):
        return decison_tree_n_experiments(cross_validation_datasets, model)
    elif(model['name'] == 'naive_bayes'):
        return naive_bayes_n_experiments(cross_validation_datasets, model)
    
def get_cross_validation_datasets(dataset, no_of_folds):
    indices = [i for i in range(0, dataset.size)]
    shuffle(indices)
    split_size = int(len(indices)/no_of_folds)
    selections = []
    cross_validation_datasets = []
    for i in range(0,no_of_folds):
        selections.append(indices[i*split_size:(i+1)*split_size-1])
        
    for i in range(0, len(selections)):
        testing_dataset = dataset.subset(selections[i])
        training_selection = []
        for j in range(0, len(selections)):
            if(i!=j):
                training_selection += selections[j]
        training_dataset = dataset.subset(training_selection)
        cross_validation_datasets.append((training_dataset, testing_dataset))
    return cross_validation_datasets
def neural_network_n_experiments(datasets_collection, model_configuration):
    errors = []
    for datasets in datasets_collection:
        errors.append(neural_network_experiment(datasets, model_configuration))
    
    return errors

def neural_network_experiment(datasets,model_configuration):
    selection_indices = [i for i in range(0, datasets[0].size)]
    ann_training_dataset = datasets[0].subset(selection_indices[30:])
    ann_validation_dataset = datasets[0].subset(selection_indices[:30])
    ann_testing_dataset = datasets[1]
    neural_network_datasets = (ann_training_dataset, ann_validation_dataset, ann_testing_dataset)
    no_of_hidden_layers = model_configuration['no_of_hidden_layers']
    no_of_nodes_per_hidden_layer = model_configuration['no_of_nodes_per_hidden_layer']
    network_config = NetworkConfig()
    
    network_config.training_stop_criteria = model_configuration['training_stop_criteria']
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
    neural_network.train(model_configuration['eta'])
    neural_network.test()
    return neural_network.get_error()

def naive_bayes_n_experiments(datasets_collection, model_configuration):
    errors = []
    for datasets in datasets_collection:
        errors.append(naive_bayes_experiment(datasets, model_configuration))
    
    return errors

def naive_bayes_experiment(datasets, model_configuration):
    discretized_training_dataset = discretize_the_dataset(datasets[0])
    discretized_testing_dataset = discretize_the_dataset(datasets[1])
    naive_bayes = NaiveBayes(discretized_training_dataset, discretized_testing_dataset, model_configuration['m'])
    naive_bayes.train()
    naive_bayes.test(discretized_testing_dataset)
    return naive_bayes.get_error()

def decison_tree_n_experiments(datasets_collection, model_configuration):
    errors = []
    for datasets in datasets_collection:
        errors.append(decison_tree_experiment(datasets, model_configuration))
    
    return errors

def decison_tree_experiment(datasets, model_configuration):
    discretized_training_dataset = discretize_the_dataset(datasets[0])
    discretized_testing_dataset = discretize_the_dataset(datasets[1])
    id3 = ID3(discretized_training_dataset, discretized_testing_dataset, 0.2)
    id3.train()
    return id3.test()

if __name__ == "__main__":
    print("This is naivebayes experiments code.. run hw3.py")
