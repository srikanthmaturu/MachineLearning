# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

from Algorithms.kmeans import *
from DataFileLoaders.dataset import load_dataset
from DataFileLoaders.dataset import DataSet

srinidhi_datafilepath = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Mini Homeworks\\srinidhi_data.data'
srinidhi_columns_info_file_path = 'D:\\Cloud\\Dropbox\\MS\\CSCE 878\\Mini Homeworks\\srinidhi_data.info'
datafilepath = srinidhi_datafilepath
columns_info_file_path = srinidhi_columns_info_file_path

if __name__ == "__main__":
    dataset = DataSet(*load_dataset(datafilepath, columns_info_file_path))
    print('Printing dataset...')
    dataset.print_dataset()
    print('\n \nNormalizing...')
    dataset.normalize_dataset()
    dataset.print_normalized_dataset()
    print('\n\n')
    kms = KMeans(3)
    kms.load_dataset(dataset)
    kms.run()