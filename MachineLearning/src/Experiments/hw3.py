# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

from Experiments.naivebayes_experiments import *  
import sys
from DataFileLoaders.dataset import load_dataset
from DataFileLoaders.dataset import DataSet

congress_datafilepath = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments&Project\\Assignment\\HW01\\Srikanth\\Dataset Files\\HouseVotes\\HouseVotes.data.txt'
congress_columns_info_file_path = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments&Project\\Assignment\\HW01\\Srikanth\\Dataset Files\\HouseVotes\\HouseVotes.info.txt'
monks_datafilepath = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments&Project\\Assignment\\HW01\\Srikanth\\Dataset Files\\Monks\\monks.data.txt'
monks_columns_info_file_path = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments&Project\\Assignment\\HW01\\Srikanth\\Dataset Files\\Monks\\monks.info.txt'
balance_scale_datafilepath =  'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments&Project\\Assignment\\HW01\\Srikanth\\Dataset Files\\BalanceScale\\balancescale.data.txt'
balance_scale_columns_info_file_path =  'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments&Project\\Assignment\\HW01\\Srikanth\\Dataset Files\\BalanceScale\\balancescale.info.txt'
srinidhi_datafilepath = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Mini Homeworks\\srinidhi_data.data'
srinidhi_columns_info_file_path = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Mini Homeworks\\srinidhi_data.info'
credit_approval_filepath = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments&Project\\Assignment\\HW01\\Srikanth\\Dataset Files\\CreditApproval\\crx.data.txt'
credit_approval_columns_info_file_path = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments&Project\\Assignment\\HW01\\Srikanth\\Dataset Files\\CreditApproval\\crx.info'

if __name__ == "__main__":
    datasetname = 'monks'
    
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
        training_stop_criteria['max_iterations'] = 1000
        dataset_description = {'datasetname':datasetname}
        m = 10
        if(mode == 'run'):
            #nayes_bayes_experiment(dataset_description, dataset, ratios, m)
            multi_classifier_roc_experiment(dataset_description, dataset, ratios)
            #cross_validation_experiment(dataset_description, dataset, 10)
            