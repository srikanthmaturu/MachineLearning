# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

from DataFileLoaders.dataset import *
from random import shuffle

def split_dataset(dataset, ratios):
    trsr, tesr, vasr = ratios
    trs_offset = 0
    tes_offset = int(dataset.size*trsr)
    vas_offset = tes_offset + int(dataset.size*tesr)
    indices = [i for i in range(0, dataset.size)]
    shuffle(indices)
    training_dataset = dataset.subset(indices[:tes_offset])
    testing_dataset = dataset.subset(indices[tes_offset:vas_offset])
    validation_dataset = dataset.subset(indices[vas_offset:])
    #print(training_dataset.size, testing_dataset.size, validation_dataset.size)
    return (training_dataset, testing_dataset, validation_dataset)

def split_dataset_by_sizes(dataset, split_sizes):
    dataset_size = dataset.size
    trs_offset = 0
    tes_offset = dataset_size - split_sizes['testing_dataset_size'] - split_sizes['validation_dataset_size']
    vas_offset = tes_offset + split_sizes['validation_dataset_size']
    indices = [i for i in range(0, dataset.size)]
    shuffle(indices)
    print(split_sizes)
    training_dataset = dataset.subset(indices[:split_sizes['training_dataset_size']])
    testing_dataset = dataset.subset(indices[tes_offset:vas_offset])
    validation_dataset = dataset.subset(indices[vas_offset:])
    #print(training_dataset.size, testing_dataset.size, validation_dataset.size)
    return (training_dataset, testing_dataset, validation_dataset)

def discretize_the_dataset(dataset):
    for index in range(0, len(dataset.attributes)):
        if(isinstance(dataset.attributes[index], NumericAttribute)):
            numeric_attribute = dataset.attributes[index]
            dataset.attributes[index] = dataset.attributes[index].discrete_version
            dataset.attributes[index].numeric_version = numeric_attribute
        elif(isinstance(dataset.label, NumericLabel)):
            numeric_label = dataset.label
            dataset.label = dataset.label.discretize()
            dataset.label.numeric_version = numeric_label
    return dataset
    
if __name__ == "__main__":
    print("Experiments....")

