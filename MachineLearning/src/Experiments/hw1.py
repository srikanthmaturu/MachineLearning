# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import decisontree_experiments
import sys
from DataFileLoaders.dataset import load_dataset
from DataFileLoaders.dataset import DataSet
from Algorithms.decisontree import ID3
from DataSetFilters import datasetfilters
from Algorithms.decisontree import Prune
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
    datasetname = 'congress'
    
    if datasetname == 'congress':
        datafilepath = congress_datafilepath
        columns_info_file_path = congress_columns_info_file_path
    elif datasetname == 'monks':
        datafilepath = monks_datafilepath
        columns_info_file_path = monks_columns_info_file_path
    elif datasetname == 'balance_scale':
        datafilepath = balance_scale_datafilepath
        columns_info_file_path = balance_scale_columns_info_file_path
        
    arguments_size = len(sys.argv)
    indexes = [index for index in range(0, arguments_size)]
    if 1 not in indexes:
        print('data file (.data) path is not present in arguments list... ending.. program.. bye..bye...')
    elif 2 not in indexes:
        print('data info file (.info) path is not present in arguments list... ending program... bye.. bye... ')
    else:
        datafilepath = sys.argv[1]
        columns_info_file_path = sys.argv[2]
    
    #decisontree_experiments.experiment1(datafilepath, columns_info_file_path, 1)
    #decisontree_experiments.experiment2(datafilepath, columns_info_file_path)
    
    if(True):
        dataset = DataSet(*load_dataset(datafilepath, columns_info_file_path))
        ratios = [0.6, 0.35, 0.05]
        training_dataset, testing_dataset, validation_dataset = datasetfilters.split_dataset(dataset, ratios)
        results = decisontree_experiments.experiment_iteration(training_dataset, testing_dataset, validation_dataset)
        #print(results)
        decisontree_experiments.print_results(results['iteration_results'], results['description'])