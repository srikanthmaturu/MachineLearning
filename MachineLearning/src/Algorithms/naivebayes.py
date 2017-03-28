# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

class NaiveBayes:
    def __init__(self, training_dataset, testing_dataset, m):
        self.training_dataset = training_dataset
        self.testing_dataset = testing_dataset
        self.testing_error = ''
        self.confusion_matrix = ''
        self.pseudo_count = m
    
    def set_psuedo_count(self, m):
        self.pseudo_count = m
        
    def set_training_dataset(self, training_dataset):
        self.training_dataset = training_dataset
        
    def set_testing_dataset(self, testing_dataset):
        self.testing_dataset = testing_dataset
    
    def train(self,training_dataset=''):
        if(training_dataset != ''):
            self.training_dataset = training_dataset
        self.nb_prior_probabilities = NBPriorProbabilities(self.training_dataset, self.pseudo_count)
    
    def test(self, testing_dataset=''):
        if(testing_dataset != ''):
            self.testing_dataset = testing_dataset
        selection = [i for i in range(0, len(self.testing_dataset.attributes[0].attribute))]
        #self.testing_dataset.print_dataset()
        #print(' Size: '+str(self.testing_dataset.size)+'\nAttribute size: '+str(self.testing_dataset.attributes[0].attribute)+' \nSelection: \n ', selection)
        self.set_initial_confusion_matrix()
        for selected_test_instance_index in selection:
            selected_test_instance = self.testing_dataset.subset([selected_test_instance_index])
            predicted_label = self.nb_prior_probabilities.predict_label(selected_test_instance)
            #selected_test_instance.print_image_dataset()
            #print('\nPredicted label: ', selected_test_instance.label.get_unique_nominal_value_from_index(predicted_label),'\n')
            self.confusion_matrix[selected_test_instance.label.label[0]][predicted_label] += 1
        
        row_size = len(self.confusion_matrix)
        self.testing_error = sum([sum([self.confusion_matrix[i][j] for j in range(0, row_size) if i!=j ]) for i in range(0, row_size)])/len(self.testing_dataset.attributes[0].attribute)
        self.compute_tp_fp_rates()
        
    def set_initial_confusion_matrix(self):
        no_of_classes = len(self.training_dataset.label.unique_nominal_values)
        self.confusion_matrix = [[0 for i in range(0, no_of_classes)] for j in range(0, no_of_classes)]
        self.unique_labels = self.training_dataset.label.unique_nominal_values
    
    def display_confusion_matrix(self, confusion_matrix=''):
        if(confusion_matrix == ''):
            confusion_matrix = self.get_confusion_matrix()
        label_names = self.unique_labels
        #print('Printing confusion matrix: ')
        label_names_lengths = [len(label_name) for label_name in label_names]
        max_label_name_length = max(label_names_lengths + [16])
        field_size = max_label_name_length + 3
        label_string = '{:' + str(field_size) + '}'
        for label_name in label_names:
            label_string = label_string + '{:' + str(field_size) + '}'
        header = ['Actual\\Predicted'] + label_names
        
        print(label_string.format(*header))
        print(('{:-<'+str(len(label_string.format(*header)))+'}').format('-'))
        
        current_line = ''
        for label_name, label_composition in zip(label_names, confusion_matrix):
            current_line += '{:'+ str(field_size) +'}'
            for label_count in label_composition:
                current_line += '{:<'+ str(field_size) +'}'
            print(current_line.format(label_name, *label_composition))
            current_line = ''
        print('')
        
    def compute_tp_fp_rates(self):
        no_of_classes = len(self.unique_labels)
        tp_fp_rates = {}
        for i in range(0, no_of_classes):
            tp = self.confusion_matrix[i][i]
            p_total = sum(self.confusion_matrix[i][:])
            tp_rate = tp/p_total
            fp = sum([self.confusion_matrix[j][i] for j in range(0, no_of_classes) if(i!=j)])
            f_total = sum([sum(self.confusion_matrix[j][:]) for j in range(0, no_of_classes) if i!=j])
            fp_rate = fp/f_total
            tp_fp_rates[self.unique_labels[i]] = [tp_rate, fp_rate]
        
        self.tp_fp_rates = tp_fp_rates
    
    def get_tp_fp_rates(self):
        return self.tp_fp_rates
    
    def get_error(self):
        return self.testing_error
        
    def get_confusion_matrix(self):
        return self.confusion_matrix  

    def print_tp_fp_rates(self):
        print('')
        #print('{:-<200}'.format('-'))
        
        for label, tp_fp_rates in self.tp_fp_rates.items():
            print('Class ',label,': tp rate = ', tp_fp_rates[0], ' fp rate = ', tp_fp_rates[1])
        print('')
    
    def print_95_percent_confidence_interval(self, error='', dataset_size=''):
        if(error == '' and dataset_size == ''):
            error = self.get_error()
            dataset_size = self.testing_dataset.size
            
        standard_deviation = 1.96 * ((error * (1 - error)) / dataset_size)**0.5
        lower_bound = error - standard_deviation
        upper_bound = error + standard_deviation
        print('')
        print('Approximately with 95% confidence, generalization_error lies in between ({},{})'.format(lower_bound, upper_bound))
        print('')
    
    def print_error(self):
        print('')
        print('Prediction error: ', self.testing_error)
        print('')
    
    def print_prior_probabilities(self):
        print('')
        self.nb_prior_probabilities.print_prior_probabilities()
        print('')
    
class NBPriorProbabilities:
    def __init__(self, training_dataset, m):
        self.training_dataset = training_dataset
        self.pseudo_count = m
        self.compute_prior_probabilites()
        
    
    def set_training_dataset(self, training_dataset):
        self.training_dataset = training_dataset
    
    def compute_prior_probabilites(self):
        self.attributes_prior_probabilities = {}
        #print(self.pseudo_count)
        for attribute in self.training_dataset.attributes:
            self.attributes_prior_probabilities[attribute.attrname] = AttrPriorProbabilities(attribute, self.training_dataset.label, self.pseudo_count)
        
        self.label_prior_probabilities = LabelPriorProbabilities(self.training_dataset.label)
            
    def predict_label(self, dataset):
        label_probabilities = {}
        max_probability = -1
        max_label = ''
        
        for label in self.label_prior_probabilities.unique_label_nominal_values:
            label_probabilities[label] = self.label_prior_probabilities.get_prior_probabilities(label)
            product = 1 * label_probabilities[label]
            for attribute in dataset.attributes:
                product *= self.attributes_prior_probabilities[attribute.attrname].get_prior_probabilities(attribute.attribute[0], label)
            
            if product > max_probability:
                max_probability = product
                max_label = label
        
        return self.label_prior_probabilities.indexed_unique_label_nominal_values[max_label]
    
    def print_prior_probabilities(self):
        print('--Label prior probabilities: ')
        self.label_prior_probabilities.print_prior_probabilities()
        print('')
        print('--Attributes prior probabilities: ')
        for key in self.attributes_prior_probabilities.keys():
            self.attributes_prior_probabilities[key].print_prior_probabilities()
        print('')
            
    
class PriorProbabilities:
    def __init__(self):
        pass
    
    def compute_prior_probabilities(self):
        pass
    
    def get_priori_proabilities(self):
        pass

class AttrPriorProbabilities:
    def __init__(self, attribute, label, m):
        self.attribute = attribute
        self.attrname = attribute.attrname
        self.label = label
        self.unique_label_nominal_values = label.unique_nominal_values
        self.unique_attribute_nominal_values = attribute.unique_nominal_values
        self.pseudo_count = m
        self.prior_estimate = 1/len(self.unique_attribute_nominal_values)
        self.compute_prior_probabilities()
    
    def compute_prior_probabilities(self):
        self.prior_probabilities = {}
        self.label_counter = {}
        #print(self.unique_label_nominal_values)
        for unique_attribute_nominal_value in self.unique_attribute_nominal_values:
            for unique_label_nominal_value in self.unique_label_nominal_values:
                key = unique_attribute_nominal_value+'_'+unique_label_nominal_value
                self.prior_probabilities[key] = 0
        
        for unique_label_nominal_value in self.unique_label_nominal_values:
            self.label_counter[unique_label_nominal_value] = 0
        
        self.indexed_unique_attribute_nominal_values = self.attribute.indexed_unique_nominal_values
        self.indexes_to_unique_attr_values = {value:key for key, value in self.indexed_unique_attribute_nominal_values.items()}
        self.indexed_unique_label_nominal_values = self.label.indexed_unique_nominal_values
        self.indexes_to_unique_label_values = {value:key for key, value in self.indexed_unique_label_nominal_values.items()}
        attribute = self.attribute.attribute
        label = self.label.label
        for attr, lbl in zip(attribute, label):
            label = self.indexes_to_unique_label_values[lbl]
            self.label_counter[label] += 1
            attribute_value = self.indexes_to_unique_attr_values[attr]
            key = attribute_value+'_'+label
            #print(self.prior_probabilities)
            self.prior_probabilities[key] += 1
        
        for key, value in self.prior_probabilities.items():
            splits = key.split('_')
            self.prior_probabilities[key] = (value + self.pseudo_count * self.prior_estimate )/ (self.label_counter[splits[1]] + self.pseudo_count)
        
    def get_prior_probabilities(self, attribute, value):
        return self.prior_probabilities[self.indexes_to_unique_attr_values[attribute]+'_'+value]
    
    def print_prior_probabilities(self):
        for key in sorted(self.prior_probabilities.keys(), key = lambda key: (key.split('_')[1], key.split('_')[0])):
            splits = key.split('_')
            label = splits[1]
            attr_value = splits[0]
            print('---- p('+attr_value+'/'+label+') = ', self.prior_probabilities[key])
    
    def print_prior_probabilities_wiki_markup(self):
        print('{| class="wikitable sortable" align="center"')
        print('|-\n! style="width:150px;" |', self.attrname, '\n! style="width:150px;" | m = 3 p = '+"{:.4f}".format(self.prior_estimate))
        for key in sorted(self.prior_probabilities.keys(), key = lambda key: (key.split('_')[1], key.split('_')[0])):
            splits = key.split('_')
            label = splits[1]
            attr_value = splits[0]
            print('|-')
            print('| p('+attr_value+'/'+label+') || ', "{:.4f}".format(self.prior_probabilities[key]))
        
        print('|}')
            
            
    
class LabelPriorProbabilities:
    def __init__(self, label):
        self.label = label
        self.compute_prior_probabilities()
        
    def compute_prior_probabilities(self):
        self.unique_label_nominal_values = self.label.unique_nominal_values
        self.indexed_unique_label_nominal_values = self.label.indexed_unique_nominal_values
        self.indexes_to_unique_label_values = {value:key for key, value in self.indexed_unique_label_nominal_values.items()}
        self.prior_probabilities = {}
        self.label_counter = {}
        for unique_label_nominal_value in self.unique_label_nominal_values:
            self.label_counter[unique_label_nominal_value] = 0
            self.prior_probabilities[unique_label_nominal_value] = 0
        
        self.datasize = len(self.label.label)
        for lbl in self.label.label:
            label = self.indexes_to_unique_label_values[lbl]
            self.prior_probabilities[label] += 1
            
        for key, value in self.prior_probabilities.items():
            self.prior_probabilities[key] = value /self.datasize
    
    def get_prior_probabilities(self, label):
        #return self.prior_probabilities[self.indexes_to_unique_label_values[label]]
        return self.prior_probabilities[label]
    
    def print_prior_probabilities(self):
        for key, value in self.prior_probabilities.items():
            print('---- p('+key+') = ',value)
    
    def print_prior_probabilities_wiki_markup(self):
        print('{| class="wikitable sortable" align="center" ')
        print('|-\n! style="width:150px;" |Class probabilites \n! style="width:150px;" | ')
        for key, value in self.prior_probabilities.items():
            print('|-')
            print('|p('+key+') || ',value)
            
        print('|}')
        
        
if __name__ == "__main__":
    print("This is naive bayes code.. run hw3.py")
  