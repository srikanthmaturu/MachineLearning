# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.    

import os

delimiters = { 'comma':',', 'space':' ', 'tab':'\t'}

class Column:
    def __init__(self, column_info):
        self.columnname = column_info['columnname']
        self.columntype = column_info['columntype']
        if('columntype' in column_info.keys()):
            pass

datafilepath = 'C:\\Users\\SrikanthPhaniKrishna\\Dropbox\\MS\\CSCE 878\\Assignments\\Assignment\\HW01\\Srikanth\\Dataset Files\\HouseVotes\\HouseVotes.data.txt'
columns_info_file_path = 'C:\\Users\\SrikanthPhaniKrishna\\Dropbox\\MS\\CSCE 878\\Assignments\\Assignment\\HW01\\Srikanth\\Dataset Files\\HouseVotes\\HouseVotes.info.txt'
rows_size = 0
columns_size = 0
labels = {}
attributenames = {}
attributes_info = {}
vectors = []
vectors_labels = []


def extract_fields(filepath, field_delimiter):
    f = open(filepath)
    rawdata = [[value for value in line.strip().split(field_delimiter)] for line in f]
    f.close()
    global rows_size, columns_size
    rows_size = len(rawdata)
    columns_size = len(rawdata[0])
    fields = [[rawdata[j][i] for j in range(0 , rows_size)] for i in range(0, columns_size)]
    #print(fields)
    return fields
    

def read_columns_info(filepath):
    contents = open(filepath).readlines()
    field_delimiter = delimiters[contents[0].rstrip().split(' ')[1]]
    class_labels_column_name = contents[1].rstrip().split(' ')[1]
    #print(class_labels_column_name)
    contents.pop(0)
    contents.pop(0)
    columns_info = [read_column_info(line.rstrip().split(' '), class_labels_column_name) for line in contents]
    return (columns_info, field_delimiter)
    
def read_column_info(info, class_labels_column_name):
    column_info = {}
    column_info['column_name'] = info[0][:-1]
    
    if(info[0][:-1] == class_labels_column_name):
        column_info['class_labels_column_name'] = 1
    else:
        column_info['class_labels_column_name'] = 0
    
    if(info[1] == 'irrelavant'):
        column_info['hidden'] = True
        return column_info
    else:
        column_info['hidden'] = False
    
    if(info[1] != 'numeric'):
        column_info['unique_nominal_values_set_size'] = int(info[1])
        column_info['unique_nominal_values'] = info[2][1:-1].split(',')
        column_info['column_type'] = 'nominal'
    else:
        column_info['column_type'] = 'numeric'
        column_info['numeric_data'] = False
    
    return column_info

def read_data(datafile, datainfofile):
    columns_info, field_delimiter = read_columns_info(datainfofile)
    fields = extract_fields(datafile, field_delimiter)
    indices = [i for i in range(0, len(columns_info)) if(columns_info[i]['hidden'])]
    new_fields, new_columns_info = [], []
    for field, column_info in zip(fields, columns_info):
        if(not column_info['hidden']):
            new_fields.append(field)
            new_columns_info.append(column_info)
            
    #print(index)
    #print(len(fields), len(columns_info))
    
    #print(fields)
    #print(columns_info)
    return (new_fields, new_columns_info)
    
if __name__ == "__main__":
    #filepath = sys.argv[1]
    if(os.path.isfile(datafilepath)):
        read_data(datafilepath, columns_info_file_path)
    else:
        print('file not found')