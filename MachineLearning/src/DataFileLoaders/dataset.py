# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

from DataFileLoaders import loaddatafiles
from statistics import mean
import numpy as np
import lmdb
import caffe

unknowns = ['', '?']

class Attribute(object):
    def __init__(self, column='', column_info=''):
        if(column=='' or column_info==''):
            return
        
        self.attrname = column_info['column_name']
        self.size = len(column)
        self.normalized = False
        self.numeric_version = ''
        self.discrete_version = ''
        
    def subset(self, selection_indices):
        subset = self.__class__()
        subset.attrname = self.attrname
        subset.attrtype = self.attrtype
        #print(selection_indices)
        #print(self.column)
        subset.numeric_version = ''
        subset.discrete_version = ''
        subset.column = [self.column[index] for index in selection_indices]
        subset.attribute = [self.attribute[index] for index in selection_indices]
        if(subset.attribute):
            subset.minimum_value = min(subset.attribute)
            subset.maximum_value = max(subset.attribute)
        subset.selectionindices = selection_indices
        subset.size = len(subset.attribute)
        subset.prev_subset = self
        subset.normalized = self.normalized
        if(self.normalized):
            subset.norm_minimum_value = self.norm_minimum_value
            subset.norm_maximum_value = self.norm_maximum_value
            
        return subset
    
    def compute_max_min(self):
        self.maximum_value = max(self.attribute)
        self.minimum_value = min(self.attribute)
    
    def normalize(self):
        self.compute_max_min()
        self.unnormaized_attribute = self.attribute
        self.normailized = True
        new_attribute = []
        self.norm_maximum_value = self.maximum_value
        self.norm_minimum_value = self.minimum_value
        for index in range(0, len(self.attribute)):
            new_attribute.append((self.attribute[index] - self.minimum_value)/(self.maximum_value - self.minimum_value))

        self.attribute = new_attribute
        self.normalized = True
    
class NumericAttribute(Attribute):
    def __init__(self, column='', column_info=''):
        if(column=='' or column_info==''):
            return
        
        super(NumericAttribute, self).__init__(column, column_info)
        self.attrtype = 'numeric'
        self.column = column
        if(column_info['numeric_data'] == 1):
            self.attribute = column
        else:
            self.attribute = get_numeric_column(column)
            
        self.minimum_value = min(self.attribute)
        self.maximum_value = max(self.attribute)
        self.mean = mean(self.attribute)
        
        unique_values = []
        for value in self.attribute:
            if value not in unique_values:
                unique_values.append(value)
        number_of_intervals = len(unique_values)
        
        if(number_of_intervals > 10):
            number_of_intervals = 10
            
        self.numeric_version = self
        self.discrete_version = self.discretize(number_of_intervals)
        
    def subset(self, selection_indices, categorical_version=''):
        subset = super(self.__class__, self).subset(selection_indices)
        if(len(selection_indices) > 0):
            subset.minimum_value = min(subset.attribute)
            subset.maximum_value = max(subset.attribute)
            subset.mean = mean(subset.attribute)
        else:
            subset.minimum_value = ''
            subset.maximum_value = 'max(subset.attribute)'
            subset.mean = ''
        
        if(categorical_version == ''):
            subset.discrete_version = self.discrete_version.subset(selection_indices, subset)
        else:
            subset.discrete_version = categorical_version
            
        return subset
    
    def discretize(self, number_of_intervals):
        discrete_column, discretization_information = get_discrete_column(self.attribute, number_of_intervals)
        discrete_column_info = {'column_name':'discreteized_'+self.attrname, 'unique_nominal_values':discretization_information['labels']}
        discrete_attribute = CategoricalAttribute(discrete_column, discrete_column_info)
        #print(str(len(discrete_column)))
        #print(discrete_column_info)
        return discrete_attribute
    
    
class CategoricalAttribute(Attribute):
    def __init__(self, column='', column_info=''):
        if(column=='' or column_info==''):
            return
        
        super(CategoricalAttribute, self).__init__(column, column_info)
        self.attrtype = 'nominal'
        self.unique_nominal_values = column_info['unique_nominal_values']
        self.indexed_unique_nominal_values = get_indexed_unique_nominal_values(column_info['unique_nominal_values'])
        self.indexed_column = index_column_values(column, self.indexed_unique_nominal_values)
        self.attribute = self.indexed_column
        self.minimum_value = min(self.attribute)
        self.maximum_value = max(self.attribute)
        self.column = column
        self.numeric_version = ''
        
        self.discrete_version = self
        
    def subset(self, selection_indices, numeric_version = ''):
        subset = super(self.__class__, self).subset(selection_indices)

        if(self.numeric_version != ''):
            if(numeric_version != ''):
                subset.numeric_version = numeric_version
            else:
                subset.numeric_version = self.numeric_version.subset(selection_indices, subset)
        subset.unique_nominal_values = self.unique_nominal_values
        subset.indexed_unique_nominal_values = self.indexed_unique_nominal_values
        subset.indexed_column = subset.attribute
        return subset
    
class Label(object):
    def __init__(self, column='', column_info=''):
        if(column=='' or column_info==''):
            return
        self.labelname = column_info['column_name']
        
    def subset(self, selection_indices):
        subset = self.__class__()
        subset.labelname = self.labelname
        subset.labeltype = self.labeltype
        subset.column = [self.column[index] for index in selection_indices]
        subset.label = [self.label[index] for index in selection_indices]
        subset.selection_indices = selection_indices
        subset.prev_subset = self
        return subset
    
class NumericLabel(Label):
    def __init__(self, column='', column_info=''):
        if(column=='' or column_info==''):
            return
        super(NumericLabel, self).__init__(column, column_info)
        self.labeltype = 'numeric'
        self.column = column
        self.label = get_numeric_column(column)
        self.minimum_value = min(self.attribute)
        self.maximum_value = max(self.attribute)
        self.mean = mean(self.attribute)
    
    def subset(self, selection_indices):
        subset = super().subset(selection_indices)
        subset.maximum_value = max(subset.attribute)
        subset.minimum_value = min(subset.attribute)
        subset.mean = mean(subset.attribute)
        return subset
    
class CategoricalLabel(Label):
    def __init__(self, column='', column_info=''):
        if(column=='' or column_info==''):
            return
        super(CategoricalLabel, self).__init__(column, column_info)
        self.labeltype = 'nominal'
        self.unique_nominal_values = column_info['unique_nominal_values']
        self.indexed_unique_nominal_values = get_indexed_unique_nominal_values(column_info['unique_nominal_values'])
        self.indexed_column = index_column_values(column, self.indexed_unique_nominal_values)
        self.label = self.indexed_column
        self.column = column
        if(len(column_info['unique_nominal_values']) > 2):
            self.label_vector_size = len(self.unique_nominal_values)
            self.multi_label = True
            self.encode_multi_label_as_label_vector()
        else:
            self.multi_label = False
        
    def subset(self, selection_indices):
        subset = super(self.__class__, self).subset(selection_indices)
        subset.unique_nominal_values = self.unique_nominal_values
        subset.indexed_unique_nominal_values = self.indexed_unique_nominal_values
        subset.indexed_column = self.label
        if(self.multi_label):
            subset.label_vector_size = self.label_vector_size
            subset.multi_label = True
            subset.label_vector = [[self.label_vector[index][p] for p in range(0, self.label_vector_size)] for index in selection_indices]
        else:
            subset.multi_label = False
            
        return subset
        
    def encode_multi_label_as_label_vector(self):
        self.label_vector = []
        for label in self.label:
            vector = []
            for l in range(0, len(self.indexed_unique_nominal_values)):
                if label == l:
                    vector.append(1)
                else:
                    vector.append(0)
            self.label_vector.append(vector)
    
    def get_single_label_from_label_vector(self, index):
        label_vector = []
        for vector in self.label_vector:
            label_vector.append(vector[index])
    
        return label_vector
        
    def decode_label_vector_as_multi_label(self, label_vector):
        for i, l in zip(range(0, len(self.indexed_unique_nominal_values), label_vector)):
            if l == 1:
                label = i
                break
        return label
    
    def get_unique_nominal_value_from_index(self, index):
        indexes_to_unique_nominal_values = { value:key for key, value in self.indexed_unique_nominal_values.items()}
        return indexes_to_unique_nominal_values[index]

def get_discrete_column(column, number_of_intervals):
    min_value = min(column)
    max_value = max(column)
    #print(str(min_value), str(max_value), str(number_of_intervals))
    ran = max_value - min_value
    step = float(ran)/float(number_of_intervals)
    if(step > 1):
        step = int(step)
    labels = []
    last_value = min_value
    intervals = {}
    discretization_information = {'min_value':min_value,'max_value':max_value,'range':ran,'step':step}
    for i in range(0, number_of_intervals):
        next_value = last_value + step
        label = 'interval-['+str(last_value)+'-to-'+str(next_value)
        if((i + 1) == number_of_intervals):
            label += ']'
            last_interval = True
            next_value = max_value
        else:
            label += ')'
            last_interval = False
        #print(label)
        labels.append(label)
        intervals[i] = {'min':last_value, 'max':next_value, 'label':label, 'last_interval':last_interval}
        last_value = next_value
    
    discrete_column = []
    for value in column:
        st = 'original: ' + str(value)
        for key, interval in intervals.items():
            if(interval['last_interval']):
                if((value > interval['min'] and value < interval['max']) or (value == interval['min']) or (value == interval['max'])):
                    discrete_column.append(interval['label'])
                    st += 'coverted_to: '+ interval['label']
            else:
                if((value > interval['min'] and value < interval['max']) or (value == interval['min'])):
                    discrete_column.append(interval['label'])
                    st += 'coverted_to: '+ interval['label']   
    
    discretization_information['intervals'] = intervals
    discretization_information['labels'] = labels
    #print(' Column size: ', len(column), ' Discrete column size: ', len(discrete_column),' No of intervals: ', number_of_intervals, ' Column: ', column)
    #print(' Column size: ', len(column), ' Discrete column size: ', len(discrete_column),' No of intervals: ', number_of_intervals, discretization_information)
    return (discrete_column, discretization_information)
    
def is_string_column(column):
    return not is_numeric_column(column)

def is_numeric_column(column):
    values = [value for value in column if(not is_unknown(value))]
    size = len(values)
    values = [float(value) for value in values if(is_float(value))]
    return (size == len(values))
    
def is_float_column(column):
    filtered_column = [value for value in column if(not is_unknown(value))]
    float_values = [value for value in filtered_column if( '.' in value)]
    return len(float_values) > 0

def is_int_column(column):
    return not is_float_column(column)

def get_float_column(column):
    values = []
    filtered_column = [float(value) for value in column if(not is_unknown(value))]
    mean_value = sum(filtered_column)
 
    for value in column:
        if(is_unknown(value)):
            values.append(mean_value)
        else:
            values.append(float(value))
    
    return values

def get_int_column(column):
    values = []
    mean_value = mean([int(value) for value in column if(not is_unknown(value))])
    for value in column:
        if(is_unknown(value)):
            values.append(mean_value)
        else:
            values.append(int(value))
    return values

def get_numeric_column(column):
    if(is_numeric_column(column)):
        if(is_int_column(column)):
            return get_int_column(column)
        else:
            return get_float_column(column)
        
def normalize_numeric_column(column):
    pass

def get_unique_nominal_values(column):
    unique_nominal_values = set(column) - unknowns
    return list(unique_nominal_values)

def get_indexed_unique_nominal_values(uniquenominalvalues):
    uniquenominalvalues = sorted(uniquenominalvalues)
    #print(uniquenominalvalues)
    unique_column_values_indexes = { uniquenominalvalues[i] : i for i in range(0, len(uniquenominalvalues)) }
    return unique_column_values_indexes

def index_column_values(column, unique_column_values_indexes):
    indexed_column = []
    unique_column_values_frequencies = {key:0 for key in unique_column_values_indexes.keys()}
    for value in column:
        if value in unique_column_values_frequencies.keys():
            unique_column_values_frequencies[value] += 1
    max_value = -1
    max_key = ''
    for key,value in unique_column_values_frequencies.items():
        if(value > max_value):
            max_value = value
            max_key = key
    
    for value in column:
        if(value in unique_column_values_indexes.keys()):
            indexed_column.append(unique_column_values_indexes[value])
        else:
            indexed_column.append(unique_column_values_indexes[max_key])
    return indexed_column

def is_unknown(value):
    value = value.replace(' ', '')
    return (value in unknowns)

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def get_attribute(column, column_info):
    if(column_info['class_labels_column_name']==0):
        #print(column_info)
        if(column_info['column_type'] == 'nominal'):
            return CategoricalAttribute(column, column_info)
        else:
            return NumericAttribute(column, column_info)
    elif(column_info['class_labels_column_name']==1):
        if(column_info['column_type'] == 'nominal'):
            return CategoricalLabel(column, column_info)
        else:
            return NumericLabel(column, column_info)

class DataSet:
    def __init__(self, attributes, label):
        self.size = attributes[0].size
        self.attributes = attributes
        self.label = label
        
    def subset(self, selectionindexes):
        subset_attributes = [attribute.subset(selectionindexes) for attribute in self.attributes]
        subset_label = self.label.subset(selectionindexes)
        subset = self.__class__(subset_attributes, subset_label)
        subset.prev_subset = self
        return subset
    
    def split(self, attr_index):
        indexed_unique_nominal_values = self.attributes[attr_index].indexed_unique_nominal_values
        attr_values = [indexed_unique_nominal_values[unique_nominal_value] for unique_nominal_value in indexed_unique_nominal_values.keys()]
        subsets = {attr_value:self.get_subset(attr_index, attr_value) for attr_value in attr_values}
        return subsets
    
    def get_subset(self, attr_index, attr_value):
        return self.subset(self.get_attr_value_indices(attr_index, attr_value))
    
    def get_attr_value_indices(self, attr_index, attr_value):
        attr= self.attributes[attr_index]
        attr_value_indices = [i for i in range(0, len(attr.attribute)) if(attr.attribute[i] == attr_value)]
        return attr_value_indices
    
    def get_dataset(self):
        return (self.attributes, self.label, self.size)
    
    def print_dataset(self):
        for i in range(0, self.size):
            row = ''
            for column_id in range(0, len(self.attributes)):
                row = row + str(self.attributes[column_id].column[i]) +','
            row = row + str(self.label.column[i])
            print(row)
    
    def normalize_dataset(self):
        for index in range(0, len(self.attributes)):
            self.attributes[index].normalize()
    
    def print_normalized_dataset(self):
        for i in range(0, self.size):
            row = ''
            for column_id in range(0, len(self.attributes)):
                row = row + str(self.attributes[column_id].attribute[i]) +','
            row = row + str(self.label.column[i])
            print(row)  
    
    def print_image_dataset(self):
        x_dim, y_dim  = 0, 0
        for attribute in self.attributes:
            name = attribute.attrname.split('_')
            if(name[0] == 'discreteized'):
                name.pop(0)
            if(x_dim < int(name[1])):
                x_dim = int(name[1])
            
            if(y_dim < int(name[2])):
                y_dim = int(name[2])
        
        x_dim += 1
        y_dim += 1
        
        number_of_images = len(self.attributes[0].attribute)
        images = [[[0 for y in range(0, y_dim)] for r in range(0, x_dim)] for ii in range(0, number_of_images)]
        #print('No of images: ' + str(len(images)) + ' X_dim: ' + str(len(images[0])) + ' y_dim: ' + str(len(images[0][0])))
        for attribute in self.attributes:
            name = attribute.attrname.split('_')
            if(name[0] == 'discreteized'):
                name.pop(0)
            row, column = name[1], name[2]    
            #print(' Attr_size: '+str(len(attribute.column))+' Numeric column size: ', str(len(attribute.numeric_version.column)))
            for ii in range(0, number_of_images):
                #print(attribute.numeric_version.column[ii])
                images[ii][int(row)][int(column)] = attribute.numeric_version.column[ii]
            
        for image_id in range(0, number_of_images):
            print('\nImage '+str(image_id)+': ')
            for row_id in range(0, x_dim):
                row_ = ''
                for col_id in range(0, y_dim):
                    row_ += '{:>3} '.format(str(images[image_id][row_id][col_id]))
                print(row_)
            print('\n Label '+str(image_id)+': '+str(self.label.column[image_id]))
    
    def convert_to_lmdb(self, fname):
        x_dim, y_dim  = 0, 0
        for attribute in self.attributes:
            name = attribute.attrname.split('_')
            if(name[0] == 'discreteized'):
                name.pop(0)
            if(x_dim < int(name[1])):
                x_dim = int(name[1])
            
            if(y_dim < int(name[2])):
                y_dim = int(name[2])
        
        x_dim += 1
        y_dim += 1
        image_dimensions = [x_dim, y_dim]
        number_of_images = len(self.attributes[0].attribute)
        
        images = [[[0 for y in range(0, y_dim)] for r in range(0, x_dim)] for ii in range(0, number_of_images)]

        for attribute in self.attributes:
            name = attribute.attrname.split('_')
            if(name[0] == 'discreteized'):
                name.pop(0)
            row, column = name[1], name[2]    
            #print(' Attr_size: '+str(len(attribute.column)))
            for ii in range(0, number_of_images):
                images[ii][int(row)][int(column)] = attribute.numeric_version.column[ii]
        
        map_size = number_of_images*image_dimensions[0]*image_dimensions[1]*10
        env = lmdb.open(fname+'_lmdb', map_size=map_size)
        with env.begin(write=True) as txn:
            for i in range(len(images)):
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = 1
                datum.height = image_dimensions[0]
                datum.width = image_dimensions[1]
                datum.data = np.array(images[i], dtype=np.uint8).tobytes()
                datum.label = self.label.label[i]
                str_id = '{:08}'.format(i)
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
        
    def print_wiki_markup(self):
        for i in range(0, self.size):
            row = '|-\n'
            for column_id in range(0, len(self.attributes)):
                row = row + '||' +str(self.attributes[column_id].column[i]) +''
            row = row +'||'+ str(self.label.column[i])
            print(row)
    
    def get_wiki_markup(self):
        markup = ''
        for i in range(0, self.size):
            row = ''
            for column_id in range(0, len(self.attributes)):
                row = row + '||' +str(self.attributes[column_id].column[i]) +''
            row = row +'||'+ str(self.label.column[i])
            markup = markup + row + '\n'
        return markup
        
    def print_header_wiki_markup(self):
        print('{| class="wikitable sortable" align="center" ')
        header = '|-'
        for attribute in self.attributes:
            #header = header + '\n!'+attribute.attrname+' style="width:150px;'
            header = header + '\n!'+attribute.attrname
        header = header + '\n!'+self.label.labelname
        print(header)
    
    def get_header_wiki_markup(self):
        markup = ''
        header = ''
        for attribute in self.attributes:
            #header = header + '\n!'+attribute.attrname+' style="width:150px;'
            header = header + '\n!'+attribute.attrname
        header = header + '\n!'+self.label.labelname
        markup = markup + header + '\n'
        return markup
    
class TrainingDataSet(DataSet):
    def __init__(self):
        super(TrainingDataSet, self).__init__(attributes, label)

class TestingDataSet(DataSet):
    def __init__(self):
        super(TestingDataSet, self).__init__(attributes, label)

class ValidationDataSet(DataSet):
    def __init__(self):
        super(ValidationDataSet, self).__init__(attributes, label)
    
def load_dataset(datafile, datainfofile):
    fields, columns_info = loaddatafiles.read_data(datafile, datainfofile)
    attributes = [get_attribute(field, column_info) for field, column_info in zip(fields, columns_info) if(column_info['class_labels_column_name'] == 0)]
    label = [get_attribute(field, column_info) for field, column_info in zip(fields, columns_info) if (column_info['class_labels_column_name'] == 1)]
    return (attributes, label[0])

if __name__ == "__main__":
    print(len(load_dataset(loaddatafiles.datafilepath, loaddatafiles.columns_info_file_path)[0]))