# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
from DataFileLoaders.dataset import load_dataset
from DataFileLoaders.dataset import DataSet
from Algorithms.decisontree import ID3
from DataSetFilters import datasetfilters
from Algorithms.decisontree import Prune
from plotter import plot_decison_tree_iterations_results

def experiment1(datafilepath, columns_info_file_path, no_of_iterations):
    dataset = DataSet(*load_dataset(datafilepath, columns_info_file_path))
    ratios = [0.6, 0.35, 0.05]
    iterations_results = []
    for iteration in range(1, no_of_iterations+1):
        print("Iteration no:", iteration)
        training_dataset, testing_dataset, validation_dataset = datasetfilters.split_dataset(dataset, ratios)
        results = experiment_iteration(training_dataset, testing_dataset, validation_dataset)
        results['iteration_index'] = iteration
        iterations_results.append(results)
    plot_decison_tree_iterations_results(iterations_results)
    print_iterations_results_table(iterations_results)
    
def experiment2(datafilepath, columns_info_file_path):
    dataset = DataSet(*load_dataset(datafilepath, columns_info_file_path))
    dataset_size = dataset.size
    split_sizes = {}
    split_sizes['testing_dataset_size'] = 75
    split_sizes['validation_dataset_size'] = 30
    remaining_data_size = dataset_size - split_sizes['testing_dataset_size'] - split_sizes['validation_dataset_size']
    split_sizes['training_dataset_size'] = remaining_data_size
    complete_training_dataset, testing_dataset, validation_dataset = datasetfilters.split_dataset_by_sizes(dataset, split_sizes)
    initial_ratio = 0.3
    current_ratio = initial_ratio
    iterations_results = []
    iteration = 1
    while(current_ratio < 1):
        print("Iteration no:", iteration)
        split_sizes['training_dataset_size'] = int(current_ratio * remaining_data_size)
        #training_dataset, testing_dataset, validation_dataset = datasetfilters.split_dataset_by_sizes(dataset, split_sizes)
        training_dataset = complete_training_dataset.subset([i for i in range(0,split_sizes['training_dataset_size'])])
        results = experiment_iteration(training_dataset, testing_dataset, validation_dataset)
        results['iteration_results']['training_dataset_size'] = split_sizes['training_dataset_size']
        iterations_results.append(results)
        iteration = iteration + 1
        current_ratio = current_ratio + 0.05
    print_iterations_results_table2(iterations_results)
    
def experiment_iteration(training_dataset, testing_dataset, validation_dataset):
    id3 = ID3(training_dataset, testing_dataset, 0.2)
    id3.train()
    results = {}
    results['description'] = 'Training_size: ' +str(training_dataset.size) + ' Testing_size: ' +str(testing_dataset.size) +' Validation_size: ' +str(validation_dataset.size)
    results['iteration_results'] = {}
    details_of_tree = id3.decison_tree.get_details_of_tree()
    results['iteration_results']['max_depth_of_the_tree'] = details_of_tree['max_depth']
    results['iteration_results']['avg_depth'] = details_of_tree['avg_depth']
    results['iteration_results']['no_of_nodes'] = details_of_tree['size_of_tree']
    results['iteration_results']['no_of_leaves'] = details_of_tree['no_of_leaves']
    #accuracies of each training, testing, validation datasets on id3 decison tree built 
    results['iteration_results']['prediction_accuarcy_on_training_dataset'] = id3.test(training_dataset, id3.decison_tree)
    results['iteration_results']['Prediction_accuracy_on_testing_dataset'] = id3.test(id3.testing_set, id3.decison_tree)
    results['iteration_results']['prediction_accuracy_on_valdiation_dataset'] = id3.test(validation_dataset, id3.decison_tree)
    #start pruning
    prune = Prune(id3.decison_tree)
    prune.generate_rules()
    results['iteration_results']['no_of_rules'] = prune.get_size_of_rules()
    results['iteration_results']['no_of_preconditions_before_pruning'] = prune.get_current_size_of_preconditions_from_all_rules()
    prune.prune(validation_dataset)
    results['iteration_results']['pruned_tree_prediction_accuracy_on_validation_datset'] = prune.test(validation_dataset)
    results['iteration_results']['pruned_tree_prediction_accuracy_on_testing_dataset'] = prune.test(testing_dataset)
    results['iteration_results']['no_of_preconditions_after_pruning'] = prune.get_current_size_of_preconditions_from_all_rules()
    #print(results)
    #print_results(results['iteration_results'], results['description'])
    return results

def print_results(iteration_result, description):
    print('Data sets sizes:', description)
    print('Results:--')
    print('---Prediction Accuracies:')
    print('   prediction_accuarcy_on_training_dataset:', iteration_result['prediction_accuarcy_on_training_dataset'])
    print('   Prediction_accuracy_on_testing_dataset:', iteration_result['Prediction_accuracy_on_testing_dataset'])
    print('   prediction_accuracy_on_valdiation_dataset:', iteration_result['prediction_accuracy_on_valdiation_dataset'])
    print('   pruned_tree_prediction_accuracy_on_validation_datset:', iteration_result['pruned_tree_prediction_accuracy_on_validation_datset'])
    print('   pruned_tree_prediction_accuracy_on_testing_dataset:', iteration_result['pruned_tree_prediction_accuracy_on_testing_dataset'])
    print('---Decison Tree details')
    print('   max_depth_of_the_tree:', iteration_result['max_depth_of_the_tree'])
    print('   avg_depth:', iteration_result['avg_depth'])
    print('   no_of_nodes:', iteration_result['no_of_nodes'])
    print('   no_of_leaves:', iteration_result['no_of_leaves'])
    print('---Pruning details:')
    print('   no_of_rules:', iteration_result['no_of_rules'])
    print('   no_of_preconditions_before_pruning:', iteration_result['no_of_preconditions_before_pruning'])
    print('   no_of_preconditions_after_pruning:', iteration_result['no_of_preconditions_after_pruning'])

def print_iterations_results_table2(iterations_results):
    if(len(iterations_results) > 0):
        initial_line = get_initial_line(iterations_results[0]['iteration_results'])
        print(initial_line)
        for iteration_results in iterations_results:
            print(get_line2(iteration_results['iteration_results']))
        
def get_initial_line(iteration_results):
    initial_line = ''
    for key in iteration_results.keys():
        initial_line = initial_line + ',' + str(key)
    initial_line = initial_line[1:]
    return initial_line

def get_line2(iteration_results):
    line = ''
    for key in iteration_results.keys():
        line = line + ',' + str(iteration_results[key])
    line = line[1:]
    return line
        
def print_iterations_results_table(iterations_results):
    initial_line = 'prediction_accuarcy_on_training_dataset' + ',' + 'Prediction_accuracy_on_testing_dataset' + ',' 
    initial_line = initial_line + 'prediction_accuracy_on_valdiation_dataset' + ',' + 'pruned_tree_prediction_accuracy_on_validation_datset' + ',' 
    initial_line = initial_line + 'pruned_tree_prediction_accuracy_on_testing_dataset' + ',' + 'max_depth_of_the_tree' + ',' 
    initial_line = initial_line + 'avg_depth' + ',' + 'no_of_nodes' + ','
    initial_line = initial_line + 'no_of_leaves' + ',' + 'no_of_rules' + ','
    initial_line = initial_line + 'no_of_preconditions_before_pruning' + ',' + 'no_of_preconditions_after_pruning'
    print(initial_line)
    for iteration_results in iterations_results:
        print(get_line(iteration_results['iteration_results']))

def get_line(iteration_results):
    #print(iteration_results)
    line = str(iteration_results['prediction_accuarcy_on_training_dataset']) + ',' + str(iteration_results['Prediction_accuracy_on_testing_dataset']) + ',' 
    line = line + str(iteration_results['prediction_accuracy_on_valdiation_dataset']) + ',' + str(iteration_results['pruned_tree_prediction_accuracy_on_validation_datset']) + ',' 
    line = line + str(iteration_results['pruned_tree_prediction_accuracy_on_testing_dataset']) + ',' + str(iteration_results['max_depth_of_the_tree']) + ',' 
    line = line + str(iteration_results['avg_depth']) + ',' + str(iteration_results['no_of_nodes']) + ','
    line = line + str(iteration_results['no_of_leaves']) + ',' + str(iteration_results['no_of_rules']) + ','
    line = line + str(iteration_results['no_of_preconditions_before_pruning']) + ',' + str(iteration_results['no_of_preconditions_after_pruning'])
    return line

if __name__ == "__main__":
    print("This is experiment1. run hw1.py")