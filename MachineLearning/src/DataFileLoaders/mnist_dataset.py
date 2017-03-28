# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

from DataFileLoaders import dataset

from struct import *
import copy
import numpy as np
import lmdb
import caffe
import sys


float_format_string = '>f'
unsighned_int_format_string = '>I'


type_code_to_type = {8:'unsighned_byte', 9:'signed_byte', 11:'unsighned_short', 12:'unsighned_int', 13:'float', 14:'double'}

type_to_format_code = {'unsighned_byte':'B', 'sighned_byte':'b', 'unsighned_short':'H', 'unsighned_int':'I', 'float':'f', 'double':'d'}

type_size = {'unsighned_byte':1, 'sighned_byte':1, 'unsighned_short':2, 'unsighned_int':4, 'float':4, 'double':8}


class Image:
    def __init__(self, image_type='', matrix=[], dimensions = []):
        self.image_type = image_type
        self.matrix = matrix
        self.dimensions = dimensions
        
    def get_dimensions(self):
        return self.dimensions
    
    def set_matrix(self, matrix, dimensions):
        self.dimensions = dimensions
        self.matrix = matrix
        
    def get_matrix(self):
        return self.matrix
    
    def get_duplicate(self):
        image_type = self.image_type
        matrix = copy.deepcopy(self.matrix)
        dimensions = copy.deepcopy(self.dimensions)
        image = Image(image_type, matrix, dimensions)
        return image

class MNIST_dataset:
    def __init__(self, images=[], labels=[]):
        self.images = images
        self.labels = labels
        self.number_of_images = len(self.images)
        
    def set_images(self, images):
        self.images = images
        self.number_of_images = len(self.images)
        
    def set_labels(self, labels):
        self.labels = labels
        
    def get_images(self):
        return self.images
    
    def get_labels(self):
        return self.labels
    
    def print_2d_dataset(self):
            for image_id in range(0, self.number_of_images):
                print('\nImage '+str(image_id)+': ')
                for row_id in range(0, self.images[image_id].dimensions[0]):
                    row_ = ''
                    for col_id in range(0, self.images[image_id].dimensions[1]):
                        row_ += '{:>3} '.format(str(self.images[image_id].get_matrix()[row_id][col_id]))
                    print(row_)
                print('\n Label '+str(image_id)+': '+str(self.labels[image_id]))
    
    def subset(self, selection_indices):
        images = [self.images[index].get_duplicate() for index in selection_indices]
        labels = [self.labels[index] for index in selection_indices]
        dataset = MNIST_dataset(images, labels)
        return dataset
    
    def convert_to_standard_dataset(self):
        dimensions = self.images[0].get_dimensions()
        columns = []
        columns_infos = []
        for r in range(0, dimensions[0]):
            for c in range(0, dimensions[1]):
                column = []
                for ii in range(0, self.number_of_images):
                    column.append(self.images[ii].get_matrix()[r][c])
                column_info = {'column_name':'pixel_'+str(r)+'_'+str(c), 'column_type':'numeric','numeric_data':1,'class_labels_column_name':0 }
                columns.append(column)
                columns_infos.append(column_info)
        attributes = [dataset.get_attribute(column, column_info) for column, column_info in zip(columns, columns_infos)]
        label_column_info = {'column_name':'number','column_type':'nominal', 'class_labels_column_name':1}
        label_column_info['unique_nominal_values_set_size'] = 10
        label_column_info['unique_nominal_values'] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        label_column = [str(label) for label in self.labels]
        label = dataset.get_attribute(label_column, label_column_info)
        dataset_obj = dataset.DataSet(attributes, label)
        self.standard_dataset = dataset_obj
        return dataset_obj
    
    def append_dataset(self, dataset):
        self.images += dataset.images
        self.labels += dataset.labels
        self.number_of_images = len(self.images)
    
    def convert_to_lmdb(self, name):
        image_dimensions = self.images[0].get_dimensions()
        map_size = len(self.images)*image_dimensions[0]*image_dimensions[1]*10
        env = lmdb.open(name+'_lmdb', map_size=map_size)
        with env.begin(write=True) as txn:
            for i in range(len(self.images)):
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = 1
                datum.height = image_dimensions[0]
                datum.width = image_dimensions[1]
                datum.data = np.array(self.images[i].get_matrix(), dtype=np.uint8).tobytes()
                #datum.label = pack('B', int(self.labels[i]))
                datum.label = int(self.labels[i])
                #print(sys.getsizeof(self.labels[i]))
                str_id = '{:08}'.format(i)
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
        
def load_mnist_image_file(filename):
    f = open(filename, 'rb');
    zero_value = unpack('>H', f.read(2))
    print('Reading image file.. start byte = ', zero_value)
    mat_val_type_code = unpack('>B', f.read(1))[0]
    f.seek(1, 1)
    number_of_images = unpack(unsighned_int_format_string, f.read(4))[0]
    number_of_rows = unpack(unsighned_int_format_string, f.read(4))[0]
    number_of_columns = unpack(unsighned_int_format_string, f.read(4))[0]
    images = []
    mat_value_format_string = '>'+type_to_format_code[type_code_to_type[mat_val_type_code]]
    
    for image_id in range(0, number_of_images):
        image = []
        for row_id in range(0, number_of_rows):
            row = []
            for col_id in range(0, number_of_columns):
                value = unpack(mat_value_format_string, f.read(type_size[type_code_to_type[mat_val_type_code]]))[0]
                row.append(value)
            image.append(row)
        image_ob = Image('gray_scale', image, [number_of_rows, number_of_columns])
        images.append(image_ob)
    return images
    
def load_mnist_label_file(filename):
    f = open(filename, 'rb');
    zero_value = unpack('>H', f.read(2))
    print('Reading label file.. start byte = ', zero_value)
    label_val_type_code = unpack('>B', f.read(1))[0]
    f.seek(1, 1)
    number_of_labels = unpack(unsighned_int_format_string, f.read(4))[0]
    labels = []
    label_value_format_string = '>'+type_to_format_code[type_code_to_type[label_val_type_code]]
    
    for label_id in range(0, number_of_labels):
        value = unpack(label_value_format_string, f.read(type_size[type_code_to_type[label_val_type_code]]))[0]
        labels.append(value)
    
    return labels

def load_mnist_dataset(training_images_filename, training_labels_file_name):
    images = load_mnist_image_file(training_images_filename)
    labels = load_mnist_label_file(training_labels_file_name)
    dataset = MNIST_dataset(images, labels)
    return dataset


if __name__ == "__main__":
    training_images_filename = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments&Project\\Project\\Data\\MNIST\\train-images.idx3-ubyte'
    training_labels_file_name = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments&Project\\Project\\Data\\MNIST\\train-labels.idx1-ubyte'
    testing_images_filename = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments&Project\\Project\\Data\\MNIST\\t10k-images.idx3-ubyte'
    testing_labels_file_name = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Assignments&Project\\Project\\Data\\MNIST\\t10k-labels.idx1-ubyte'

    load_mnist_dataset(testing_images_filename, testing_labels_file_name).subset([i for i in range(0, 100)]).print_2d_dataset()
    
    print("This MNIST dataset handling code.")


