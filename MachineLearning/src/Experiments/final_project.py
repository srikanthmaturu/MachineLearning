# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

from DataFileLoaders import mnist_dataset
from final_project_experiments import *
from Experiments.final_project_experiments import *
from DataSetFilters.datasetfilters import *
import time
import sys

training_images_filename = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments&Project\\Project\\Data\\MNIST\\train-images.idx3-ubyte'
training_labels_file_name = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments&Project\\Project\\Data\\MNIST\\train-labels.idx1-ubyte'
testing_images_filename = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments&Project\\Project\\Data\\MNIST\\t10k-images.idx3-ubyte'
testing_labels_file_name = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments&Project\\Project\\Data\\MNIST\\t10k-labels.idx1-ubyte'

letter_dataset_data_filename = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments&Project\\Project\\Data\\MSPLSG\\letter.data'
letter_dataset_info_filename = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments&Project\\Project\\Data\\MSPLSG\\letter.info'

arguments_size = len(sys.argv)
indexes = [index for index in range(0, arguments_size)]
if 1 not in indexes:
    print('data file (.data) path is not present in arguments list... ending.. program.. bye..bye...')
    #sys.exit(0)
elif 2 not in indexes:
    print('data info file (.info) path is not present in arguments list... ending program... bye.. bye... ')
    #sys.exit(0)
else:
    letter_dataset_data_filename = sys.argv[1]
    letter_dataset_info_filename = sys.argv[2]
    training_images_filename = sys.argv[1]
    training_labels_file_name = sys.argv[2]
    testing_images_filename = sys.argv[3]
    testing_labels_file_name = sys.argv[4]

def load_mnist_dataset(discretize):
    print('Loading the MNIST dataset: ')
    #start_time = time.process_time()
    mnist_training_dataset = mnist_dataset.load_mnist_dataset(training_images_filename, training_labels_file_name)

    mnist_testing_dataset = mnist_dataset.load_mnist_dataset(testing_images_filename, testing_labels_file_name)

    mnist_training_dataset.append_dataset(mnist_testing_dataset)
    dataset = mnist_training_dataset
    print('Converting dataset to standard dataset...')
    dataset = dataset.convert_to_standard_dataset()
    if(discretize):
        dataset = discretize_the_dataset(dataset)
    #end_time = time.process_time()
    #print(' Total time taken : ', end_time - start_time)
    return dataset

def load_and_convert_mnist_to_lmdb_dataset():
    print('Loading the MNIST dataset: ')
    #start_time = time.process_time()
    mnist_training_dataset = mnist_dataset.load_mnist_dataset(training_images_filename, training_labels_file_name)
    print('Training dataset conversion to lmdb started.')
    mnist_training_dataset.convert_to_lmdb('mnist_train')
    print('Training dataset conversion to lmdb complete.')
    mnist_testing_dataset = mnist_dataset.load_mnist_dataset(testing_images_filename, testing_labels_file_name)
    print('Testing dataset conversion to lmdb started.')
    mnist_testing_dataset.convert_to_lmdb('mnist_test')
    print('Testing dataset conversion to lmdb complete.')
    return

def load_msplsg_dataset():
    print('Loading the MSPLSG dataset....')
    #start_time = time.process_time()
    msplsg_dataset = DataSet(*load_dataset(letter_dataset_data_filename, letter_dataset_info_filename))
    discrete_dataset = discretize_the_dataset(msplsg_dataset)
    #end_time = time.process_time()
    #print(' Total time taken: ', end_time - start_time)
    #discrete_dataset.print_dataset()
    discrete_dataset.print_image_dataset()
    return discrete_dataset

def load_and_convert_msplsg_to_lmdb_dataset():
    print('Loading the MSPLSG dataset....')
    msplsg_dataset = DataSet(*load_dataset(letter_dataset_data_filename, letter_dataset_info_filename))
    discrete_dataset = discretize_the_dataset(msplsg_dataset)
    training_dataset, testing_dataset, validation_dataset = split_dataset(discrete_dataset, [0.7, 0.3, 0])
    print('Training dataset conversion to lmdb started.')
    training_dataset.convert_to_lmdb('msplsg_train')
    print('Training dataset conversion to lmdb complete.')
    print('Testing dataset conversion to lmdb started.')
    testing_dataset.convert_to_lmdb('msplsg_test')
    print('Testing dataset conversion to lmdb complete.')
    discrete_dataset.print_image_dataset()
    
def start_ann_experiments(datasetname, datasets):
    ann_architecture = {'no_of_hidden_layers': 2, 'no_of_nodes_per_hidden_layer':800 }
    #training_stop_criteria = {'type':'weights_change', 'value':0.00005, 'max_iterations':1}
    training_stop_criteria = {'type':'max_iterations', 'value':1000, 'max_iterations':1}
    dataset_description = {'datasetname':datasetname}
    training_parameters = {'eta':0.1666}
    neural_network_experiment(dataset_description, datasets, ann_architecture, training_parameters,  training_stop_criteria)

def start_naive_bayes_experiments(datasetname, datasets):
    dataset_description = {'datasetname':datasetname}
    naive_bayes_experiment(dataset_description, datasets, 50)
    
def start_decison_tree_experiments(datasetname, datasets):
    dataset_description = {'datasetname':datasetname}
    decison_tree_experiment(dataset_description, datasets)

if __name__ == "__main__":
    datasetname = 'msplsg'
    
    if(datasetname == 'mnist'):
        load_and_convert_mnist_to_lmdb_dataset()
        quit()
        
        dataset = load_mnist_dataset(False)
        quit()
        training_dataset, testing_dataset, validation_dataset = split_dataset(dataset, [0.8, 0.15, 0.05])
        training_size = training_dataset.size
        selection_indices = [i for i in range(0, training_size)]
        validation_size = int(training_size * 0.1)
        training_size -= validation_size
        datasets = (training_dataset, testing_dataset, validation_dataset)
        start_ann_experiments(datasetname, datasets)
        #start_naive_bayes_experiments(datasetname, (training_dataset, testing_dataset, 0))
    elif(datasetname == 'msplsg'):
        load_and_convert_msplsg_to_lmdb_dataset()
        quit()
        dataset = load_msplsg_dataset()
        training_dataset, testing_dataset, validation_dataset = split_dataset(dataset, [0.7, 0.3, 0])
        datasets = (training_dataset, testing_dataset, validation_dataset)
        #start_naive_bayes_experiments(datasetname, datasets)
        decison_tree_experiment(datasetname, datasets)

