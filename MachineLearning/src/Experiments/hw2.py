# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

from Experiments import neuralnetwork_experiments 
import sys
from DataFileLoaders.dataset import load_dataset
from DataFileLoaders.dataset import DataSet
from Algorithms import neuralnetwork
from DataSetFilters import datasetfilters
from plotter import plot_decison_tree_iterations_results

congress_datafilepath = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments\\Assignment\\HW01\\Srikanth\\Dataset Files\\HouseVotes\\HouseVotes.data.txt'
congress_columns_info_file_path = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments\\Assignment\\HW01\\Srikanth\\Dataset Files\\HouseVotes\\HouseVotes.info.txt'
monks_datafilepath = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments\\Assignment\\HW01\\Srikanth\\Dataset Files\\Monks\\monks.data.txt'
monks_columns_info_file_path = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments\\Assignment\\HW01\\Srikanth\\Dataset Files\\Monks\\monks.info.txt'
balance_scale_datafilepath =  'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments\\Assignment\\HW01\\Srikanth\\Dataset Files\\BalanceScale\\balancescale.data.txt'
balance_scale_columns_info_file_path =  'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments\\Assignment\\HW01\\Srikanth\\Dataset Files\\BalanceScale\\balancescale.info.txt'
srinidhi_datafilepath = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Mini Homeworks\\srinidhi_data.data'
srinidhi_columns_info_file_path = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Mini Homeworks\\srinidhi_data.info'
credit_approval_filepath = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments\\Assignment\\HW01\\Srikanth\\Dataset Files\\CreditApproval\\crx.data.txt'
credit_approval_columns_info_file_path = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments\\Assignment\\HW01\\Srikanth\\Dataset Files\\CreditApproval\\crx.info'

if __name__ == "__main__":
    datasetname = 'balance_scale'
    
    if datasetname == 'congress':
        datafilepath = congress_datafilepath
        columns_info_file_path = congress_columns_info_file_path
    elif datasetname == 'monks':
        datafilepath = monks_datafilepath
        columns_info_file_path = monks_columns_info_file_path
    elif datasetname == 'balance_scale':
        datafilepath = balance_scale_datafilepath
        columns_info_file_path = balance_scale_columns_info_file_path
    elif datasetname == 'srinidhi_data':
        datafilepath = srinidhi_datafilepath
        columns_info_file_path = srinidhi_columns_info_file_path
    elif(datasetname == 'credit_approval'):
        datafilepath = credit_approval_filepath
        columns_info_file_path = credit_approval_columns_info_file_path
        
    arguments_size = len(sys.argv)
    indexes = [index for index in range(0, arguments_size)]
    if 1 not in indexes:
        print('data file (.data) path is not present in arguments list... ending.. program.. bye..bye...')
        #sys.exit(0)
    elif 2 not in indexes:
        print('data info file (.info) path is not present in arguments list... ending program... bye.. bye... ')
        #sys.exit(0)
    else:
        datafilepath = sys.argv[1]
        columns_info_file_path = sys.argv[2]
    
    print('Input data set is ', datasetname)
    

    mode = 'run'
    if(True):
        dataset = DataSet(*load_dataset(datafilepath, columns_info_file_path))
        dataset_size = dataset.size
        testing_size = 50
        training_size = dataset_size - testing_size
        ratios = [training_size/dataset_size, testing_size/dataset_size, 0]
        
        print('{:-<200}'.format('-'))
        print('Dataset loaded successfully, datafile:', datafilepath, ' datainfofilepath:', columns_info_file_path)
        training_stop_criteria = {'type':'weights_change', 'value':0.00005}
        training_stop_criteria['max_iterations'] = 10
        dataset_description = {'datasetname':datasetname}
        if(mode == 'run'):
            neuralnetwork_experiments.main_experiment(dataset_description, dataset, 0.116, 1, 5, training_stop_criteria, ratios, False)
            #print('{:-<200}'.format('-'))
            #neuralnetwork_experiments.compare_neural_network_and_decison_tree_using_cross_validation(dataset_description, dataset, 0.116, 1, 5, training_stop_criteria, 10)
        elif(mode == 'experiment1'):
            init_eta = 0.0001
            step = 0.0005
            etas = [init_eta+step*i for i in range(0, 5)]
            neuralnetwork_experiments.experiment1(dataset_description, dataset, etas, 2, 5, training_stop_criteria, ratios)
        elif(mode == 'experiment2'):
            no_of_layers = [i for i in range(1, 2)]
            neuralnetwork_experiments.experiment2(dataset_description, dataset, 0.1, no_of_layers, 2, training_stop_criteria, ratios)
        elif(mode == 'experiment3'):
            no_of_nodes = [i for i in range(1, 2)]
            neuralnetwork_experiments.experiment3(dataset_description, dataset, 0.1, 1, no_of_nodes, training_stop_criteria, ratios)