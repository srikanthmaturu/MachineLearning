# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

#ID3
from math import log
from statistics import mean

class DecisonTree:
    def __init__(self, dataset, root=''):
        self.dataset = dataset
        self.attributes, self.label, self.training_size = self.dataset.get_dataset()
        self.root = root
        
    def process_query(self, attributes):
        return self.root.process_query(attributes).label
    
    def classification_error(self, dataset):
        predicted_label = [self.process_query(dataset.subset([i]).attributes) for i in range(0, dataset.size)]
        #print(dataset.label.label, dataset.size)
        #print(predicted_label, len(predicted_label))
        accurate_predictions = [self.is_equal(s) for s in zip(dataset.label.label, predicted_label) ]
        return (1-sum(accurate_predictions)/len(accurate_predictions))
        
    def is_equal(self, s):
        al, pl = s
        if(al == pl):
            return 1
        else:
            return 0
    
    def get_details_of_tree(self):
        leaves = self.get_leaves()
        no_of_leaves = len(leaves)
        depths = [len(self.get_path_from_root(leaf)) for leaf in leaves]
        details = {}
        details['no_of_leaves'] = no_of_leaves
        details['max_depth'] = max(depths)
        details['avg_depth'] = mean(depths)
        details['size_of_tree'] = self.get_size_of_tree(self.root)
        return details
    
    def get_max_depth_of_the_tree(self):
        leaves = self.get_leaves()
        depths = [len(self.get_path_from_root(leaf)) for leaf in leaves]
        return (max(depths), avg(depths))
        
    def get_leaves(self):
        return self.depth_first_search(self.root)
    
    def depth_first_search(self, node):
        if(isinstance(node, DecisonNode)):
            leaves = []
            for child in node.get_children():
                leaves = leaves + self.depth_first_search(child)
            return leaves
        elif(isinstance(node, Leaf)):
            #print(node)
            return [node]
    
    def get_size_of_tree(self, node):
        no_of_nodes = 0
        if(isinstance(node, DecisonNode)):
            sizes_of_children = [self.get_size_of_tree(child) for child in node.get_children()]
            no_of_nodes = 1 + sum(sizes_of_children)
        elif(isinstance(node, Leaf)):
            no_of_nodes = 1
        
        return no_of_nodes
    
    def get_path_from_root(self, id3tree_current_node):
        if(id3tree_current_node == ''):
            return []
        
        path = []
        parent_path = []
        
        if(id3tree_current_node.parent != ''):
            parent_path = self.get_path_from_root(id3tree_current_node.parent)
        
        path = parent_path + [id3tree_current_node]
        #print(path)
        return path    
    
class Node:
    def __init__(self):
        pass
    
    def set_parent(self, parent):
        self.parent = parent
        
    def get_parent(self):
        return self.parent
    
    def set_parent_attr_value_to_child(self, attr_value):
        self.parent_attr_value_to_child = attr_value
        
    def get_parent_attr_value_to_child(self):
        if parent != '':
            return self.parent_attr_value_to_child
    
    def process_query(self, attributes):
        if(isinstance(self, Leaf)):
            return self
        elif(isinstance(self, DecisonNode)):
            decison_attribute = self.get_decison_attribute(attributes)
            children = self.get_children()
            #print('choosing_next_child:', self.attr_value_to_child_map)
            nextchild = self.attr_value_to_child_map[decison_attribute.attribute[0]]
            return nextchild.process_query(attributes)
    
class Leaf(Node):
    def __init__(self, dataset, label, parent='', parent_attr_value_to_child=''):
        super().__init__()
        self.type = 'leaf'
        if(parent == ''):
            self.parent == ''
            self.isroot == True
        else:
            self.parent = parent
            self.isroot = False
            
        self.label = label
        self.parent = parent
        self.children = ''
        self.dataset = dataset
        self.parent_attr_value_to_child = parent_attr_value_to_child
        
class DecisonNode(Node):
    def __init__(self, dataset, attr_index, parent='', parent_attr_value_to_child=''):
        super().__init__()
        self.type = 'decison_node'
        if(parent == ''):
            self.parent = ''
            self.isroot = True
        else:
            self.parent = parent
            self.isroot = False
        
        self.dataset = dataset
        self.attribute = dataset.attributes[attr_index]
        self.decison_attribute_index = attr_index
        self.decison_attribute = self.attribute.attrname
        self.decison_attribute_unique_nominal_values = self.attribute.unique_nominal_values
        self.decison_attribute_indexed_unique_nominal_values = self.attribute.indexed_unique_nominal_values
        self.decison_values = [self.decison_attribute_indexed_unique_nominal_values[key] for key in self.decison_attribute_indexed_unique_nominal_values]
        self.max_children_size = len(self.decison_values)
        self.children = []
        self.attr_value_to_child_map = {}
        self.parent_attr_value_to_child = parent_attr_value_to_child
        
    def set_children(self, children, attr_value_to_child_map):
         self.children = children
         self.attr_value_to_child_map = attr_value_to_child_map
    
    def get_children(self):
        return self.children
         
    def get_decison_attribute(self, attributes):
        attribute = [attr for attr in attributes if(attr.attrname==self.decison_attribute)]
        return attribute[0]
    
    def get_child(self, attrvalue):
        return self.attr_value_to_child_map(attrvalue)
    
    
        
class ID3:
    def __init__(self, training_set, testing_set, theta):
        self.training_set = training_set
        self.testing_set = testing_set
        self.theta = theta
        self.training_attributes, self.training_label, self.training_size = training_set.attributes, training_set.label, training_set.size
        self.attributes_size = len(self.training_attributes)
        self.attributes_names = {index:self.training_attributes[index].attrname for index in range(0, self.attributes_size)}
        self.testing_attributes, self.testing_label, self.testing_size = testing_set.attributes, testing_set.label, testing_set.size
        self.labels_names = self.training_label.unique_nominal_values
        self.labels_size = len(self.labels_names)
        self.labels = [i for i in range(0, self.labels_size)]
    
    def set_testing_dataset(self, testing_dataset):
        self.testing_set = testing_dataset
        
    def train(self):
        #print('Training Started...')
        self.decison_tree = DecisonTree(self.training_set, self.build_tree(self.training_set, self.get_available_attributes([])))
        #print('Decison Tree is constructed... Successful')
    
    def test(self):
        self.error = self.decison_tree.classification_error(self.testing_set)
        return self.error
    
    def test_with_dtree_dataset(self, dataset, decison_tree):
        return decison_tree.classification_error(dataset)
    
    def build_tree(self, dataset, remaining_attributes, parent = '', decison_label = ''):
        
        if(dataset.size == 0):
            return ''
        
        if((self.entropy(dataset) <= self.theta) or len(remaining_attributes) == 0):
            if(parent==''):
                leaf_node = self.define_leaf_node(dataset)
            else:
                leaf_node = self.define_leaf_node(dataset, parent)
            #print('Path from root:', self.get_path_from_root(leaf_node))
            return leaf_node
        
        if(dataset.size > 0):
            
            if(parent == ''):
                new_decison_node = self.define_new_decison_node(dataset, remaining_attributes)
            else:
                new_decison_node = self.define_new_decison_node(dataset, remaining_attributes, parent)
            path_from_root = self.get_path_from_root(new_decison_node)
            #print(len(path_from_root), path_from_root)
            available_attributes = self.get_available_attributes(path_from_root)
            
            #print('new_decison_node_attribute', new_decison_node.decison_attribute )
            #print('dataset_size_at_current_node:', dataset.size, 'Entropy:', self.entropy(dataset))
            #print('available_attributes:',len(available_attributes), available_attributes)
            #print('current_attribute_index:', new_decison_node.decison_attribute_index)
            subsets = dataset.split(new_decison_node.decison_attribute_index)
            #print('distribution:',[subset.size for attr_value, subset in subsets.items()])
            new_decison_node.attr_value_to_child_map = {attr_value:self.build_tree(subsets[attr_value],available_attributes,new_decison_node) for attr_value in new_decison_node.decison_values}
            #print('Sub tree ended')
            high_freq_label = self.get_high_freq_label(dataset)
            for attr_value, child in new_decison_node.attr_value_to_child_map.items():
                if(child == ''):
                   new_decison_node.attr_value_to_child_map[attr_value] = Leaf(subsets[attr_value], high_freq_label, new_decison_node)
                   new_decison_node.attr_value_to_child_map[attr_value].parent_attr_value_to_child = attr_value
                else:
                    child.parent_attr_value_to_child = attr_value
                #print(new_decison_node.attr_value_to_child_map[attr_value])
                new_decison_node.children.append(new_decison_node.attr_value_to_child_map[attr_value])
            
            return new_decison_node
    
    def define_leaf_node(self, dataset, parent=''):
        high_freq_label = self.get_high_freq_label(dataset)
        if(parent == ''):
            leaf = Leaf(dataset, high_freq_label)
        else:
            leaf = Leaf(dataset, high_freq_label, parent)
        return leaf
        
    def define_new_decison_node(self, dataset, remaining_attributes, parent=''):
        #print(remaining_attributes)
        choices = [self.calculate_impurity(dataset, attr_index) for attr_index, name in remaining_attributes.items()]
        best_attribute_index = ''
        best_attribute_subsets = ''
        best_attribute_impurity = 1000
        #print(remaining_attributes)
        for index in range(0, len(choices)):
            choice_subsets , choice_impurity, attr_index = choices[index]
            #print(index, attr_index, choice_impurity)
            if(choice_impurity < best_attribute_impurity):
                best_attribute_subsets = choice_subsets
                best_attribute_impurity = choice_impurity
                best_attribute_index = attr_index
        
        if(parent==''):
            new_decison_node = DecisonNode(dataset, best_attribute_index)
        else:
            new_decison_node = DecisonNode(dataset, best_attribute_index, parent)
        #print('best_attribute_index', best_attribute_index, 'is_there?', remaining_attributes[best_attribute_index])
        return new_decison_node
        
    def get_path_from_root(self, id3tree_current_node):
        if(id3tree_current_node == ''):
            return []
        
        path = []
        parent_path = []
        
        if(id3tree_current_node.parent != ''):
            parent_path = self.get_path_from_root(id3tree_current_node.parent)
        
        path = parent_path + [id3tree_current_node]
        #print(path)
        return path
    
    def get_available_attributes(self, path_from_root):
        used_attributes = {node.decison_attribute_index: node.decison_attribute for node in path_from_root}
        #print(len(used_attributes), used_attributes)
        remaining_attributes = {index:self.attributes_names[index] for index in range(0, self.attributes_size) if index not in used_attributes.keys() }
        return remaining_attributes
        
    def pick_attribute(self, id3tree_current_node):
        pass
    
    def calculate_impurity(self, dataset, attr_index):
        subsets = dataset.split(attr_index)
        dataset_size = dataset.size
        impurity = sum([(subset.size/dataset_size)*self.entropy(subset) for attr_value, subset in subsets.items()])
        #print([(subset.size/dataset_size)*self.entropy(subset) for attr_value, subset in subsets.items()])
        #print([subset.size for attr_value, subset in subsets.items()])
        #print([subset.size for attr_value, subset in subsets.items()])
        #print([subset.size for attr_value, subset in subsets.items()])
        #print('Impurity:', impurity)
        return (subsets, impurity, attr_index)
        
    def entropy(self, dataset):
        if(dataset.size == 0):
            return 0
        
        labels = dataset.label.label
        unique_labels = set(labels)
        labels_histogram = {label:0 for label in unique_labels}
        
        for label in labels:
            if label in labels_histogram.keys():
                labels_histogram[label] = labels_histogram[label]+1
        length = len(labels)
        entropy = sum([(labels_histogram[label]/length)*(self.log_value(labels_histogram[label]/length)) for label in labels_histogram])*-1
        #print('Entropy: ', entropy)
        return entropy
    
    def get_high_freq_label(self, dataset):
        labels = dataset.label.label
        unique_labels = set(labels)
        labels_histogram = {label:0 for label in unique_labels}
        for label in labels:
            if label in labels_histogram.keys():
                labels_histogram[label] = labels_histogram[label]+1
        length = len(labels)
        max_label_freq = max([label_freq for label, label_freq in labels_histogram.items()])
        high_freq_labels = [label for label, label_freq in labels_histogram.items() if(max_label_freq == label_freq)]
        return high_freq_labels[0]
    
    def log_value(self, value):
        if(value==0):
            return 0
        else:
            return log(value,2)
    
    def compute_confusion_matrix(self):
        indexed_class_names = {value:key for key,value in self.testing_set.label.indexed_unique_nominal_values.items()}
        indexed_class_names = {key:indexed_class_names[key] for key in sorted(indexed_class_names.keys())}
        class_names = [value for key, value in indexed_class_names.items()]
        self.class_names = class_names
        predicted_label = [self.decison_tree.process_query(self.testing_set.subset([i]).attributes) for i in range(0, self.testing_set.size)]
        self.confusion_matrix = [[0 for i in range(0, len(class_names))] for j in range(0, len(class_names))]
        label = self.testing_set.label.label
        for index in range(0, len(label)):
            self.confusion_matrix[label[index]][predicted_label[index]] += 1
    
    def compute_tp_fp_rates(self):
        self.compute_true_positive_rates()
        self.compute_false_positive_rates()
        
    def get_confusion_matrix(self):
        return self.confusion_matrix
    
    def compute_true_positive_rates(self):
        instance_sizes = [sum(row) for row in self.confusion_matrix]
        self.tp_rates = [self.confusion_matrix[i][i]/instance_sizes[i] for i in range(0, len(self.confusion_matrix[0]))]
        
    def compute_false_positive_rates(self):
        no_of_classes = len(self.confusion_matrix[0])
        instance_sizes = [sum(row) for row in self.confusion_matrix]
        false_positive_errors = []
        for current_class in range(0, no_of_classes):
            other_classes_instances_size = sum([instance_sizes[class_index] for class_index in range(0, no_of_classes) if(class_index != current_class)])
            false_positive_count = 0
            for j in range(0, no_of_classes):
                if(current_class!=j):
                    false_positive_count += self.confusion_matrix[j][current_class]
            false_positive_errors.append(false_positive_count/other_classes_instances_size)
        self.fp_rates = false_positive_errors
    
    def get_tp_fp_rates(self):
        return (self.tp_rates, self.fp_rates)
    
    def print_tp_fp_rates(self):
        (tp_rates, fp_rates) = self.get_tp_fp_rates()
        
        label_indexes_to_names_mapping = {value:key for key, value in self.testing_set.label.indexed_unique_nominal_values.items()}
        label_names = [label_indexes_to_names_mapping[key] for key in sorted(label_indexes_to_names_mapping.keys())]
        
        for i in range(0, len(tp_rates)):
            print('Class ',label_names[i],': tp rate = ', tp_rates[i], ' fp rate = ', fp_rates[i])
        print('')
    
    def display_confusion_matrix(self, confusion_matrix=''):
        if(confusion_matrix == ''):
            confusion_matrix = self.get_confusion_matrix()
        label_names = self.class_names
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
    
    def print_95_percent_confidence_interval(self):
        error = self.error
        dataset_size = self.testing_set.size
        if(error == '' and dataset_size == ''):
            error = self.get_error()
            dataset_size = self.input_layer.get_current_dataset().size
            
        standard_deviation = 1.96 * ((error * (1 - error)) / dataset_size)**0.5
        lower_bound = error - standard_deviation
        upper_bound = error + standard_deviation
        print('Approximately with 95% confidence, generalization_error lies in between ({},{})'.format(lower_bound, upper_bound))
   
            
class Prune:
    def __init__(self, decison_tree):
        self.decison_tree = decison_tree
        self.past_flip = ''
        
    def generate_rules(self):
        leaves = self.decison_tree.get_leaves()
        #print('No_of_leaves:', len(leaves))
        paths = [self.decison_tree.get_path_from_root(leaf) for leaf in leaves]
        self.rules = [self.get_rule(path) for path in paths]
        #print('No_of_rules:', len(self.rules))
    
    def get_rule(self, path):
        if(len(path) > 0):
            preconditions = [Precondition(path[index].decison_attribute, path[index+1].parent_attr_value_to_child) for index in range(0, len(path)) if not isinstance(path[index], Leaf)]
            leaf = [node for node in path if isinstance(node, Leaf)]
            if(len(leaf) > 0):
                return Rule(leaf[0].label, preconditions)
    
    def prune(self, validation_dataset):
        if(self.test(validation_dataset) < 1):
            self.validation_dataset = validation_dataset
            #print('Current_size_of_preconditions_before_pruning: ', sum([rule.get_current_size_of_active_preconditions() for rule in self.rules]))
            #initial_error = self.test(validation_dataset)
            self.filter_preconditions()
            #final_error = self.test(validation_dataset)
            #print('Initial_error_before_prunining_on_validation_dataset', initial_error, 'Final_error_after_pruining_on_validation_dataset', final_error)
            #print('Final_size_of_preconditions_after_pruning: ', sum([rule.get_current_size_of_active_preconditions() for rule in self.rules]))
        else:
            #print("Can't do pruning. Accuracy on validation set is already 100%")
            pass
   
    def get_current_size_of_preconditions_from_all_rules(self):
        return sum([rule.get_current_size_of_active_preconditions() for rule in self.rules])
    
    def filter_preconditions(self):
        complete = False
        restart = False
        while(not complete):
            for rule_index in range(0, len(self.rules)):
                for precondition_index in range(0, len(self.rules[rule_index].get_preconditions())):
                    result = self.test_precondition(rule_index, precondition_index)
                    if(result == 'not_available' or result == 'no_improvement'):
                        continue
                    elif(result == 'complete'):
                        return
                    else:
                        restart = True
                        break
                if(restart):
                    break
            if(restart):
                restart = False
                continue
            else:
                complete = True
        
    def test_precondition(self, rule_index, precondition_index):
        if(self.rules[rule_index].get_state_of_precondition(precondition_index)):
            initial_error = 1 - self.test(self.validation_dataset)
            self.rules[rule_index].modify_precondition_by_precondition_index(precondition_index, False)
            after_error = 1 - self.test(self.validation_dataset)
            if(after_error < initial_error):
                return after_error
            elif(after_error == 1):
                return 'complete'
            else:
                self.rules[rule_index].modify_precondition_by_precondition_index(precondition_index, True)
            return 'no_improvement'
        else:
            return 'not_available'
    
    def test(self, dataset):
        predicted_label = [self.process_query(dataset.subset([i]).attributes) for i in range(0, dataset.size)]
        #print(dataset.label.label, dataset.size)
        #print(predicted_label, len(predicted_label))
        accurate_predictions = [self.is_equal(s) for s in zip(dataset.label.label, predicted_label) ]
        return sum(accurate_predictions)/len(accurate_predictions)
    
    def process_query(self, attributes):
        #print('Attributes: ', {attribute.attrname:attribute.attribute for attribute in attributes})
        choices = [rule.predict_label(attributes) for rule in self.rules]
        #print('Labels: ', choices)
        choice = [c for c in choices if(c != 'No_prediction_available')]
        if(len(choice) > 0):
            #print(choice, 'Atleast one choice is available.. picking the first one...')
            return choice[0]
        else:
            return False
    
    def is_equal(self, s):
        al, pl = s
        if(al == pl):
            return 1
        else:
            return 0
    
    def get_size_of_rules(self):
        return len(self.rules)
        
class Rule:
    def __init__(self, label, preconditions):
        self.label = label
        self.preconditions = preconditions
        #print({precondition.attr_name: precondition.attr_value for precondition in preconditions})
        self.attr_names_to_preconditions_map = {precondition.get_attr_name():precondition for precondition in preconditions}
    
    def modify_precondition_by_attr_name(self, attr_name, precondition_state):
        self.attr_names_to_preconditions_map[attr_name].modify_active_state(precondition_state)
        
    def modify_precondition_by_precondition_index(self, precondition_index, precondition_state):
        self.preconditions[precondition_index].modify_active_state(precondition_state)
    
    def predict_label(self, attributes):
        if(self.process_query(attributes)):
            return self.label
        else:
            return 'No_prediction_available'
    
    def process_query(self, attributes):
        #print(self.attr_names_to_preconditions_map)
        result = [self.attr_names_to_preconditions_map[attribute.attrname].process_query(attribute.attrname, attribute.attribute[0]) for attribute in attributes if attribute.attrname in self.attr_names_to_preconditions_map]
        #print('---', result)
        return (len(result) == sum(result))
    
    def get_preconditions(self):
        return self.preconditions
    
    def get_size_of_preconditions(self):
        return len(self.preconditions)
        
    def get_current_size_of_active_preconditions(self):
        return sum([precondition.get_active_state() for precondition in self.preconditions])
    
    def get_state_of_precondition(self, precondition_index):
        return self.preconditions[precondition_index].get_active_state()
    
class Precondition:
    def __init__(self, attr_name, attr_value, active=True):
        self.active = active
        self.attr_name = attr_name
        self.attr_value = attr_value
    
    def get_attr_name(self):
        return self.attr_name
    
    def process_query(self, attr_name, attr_value):
        if(self.active):
            #print(self.attr_name, attr_name, self.attr_value, attr_value, ((self.attr_name == attr_name) and (self.attr_value == attr_value)))
            return (self.attr_name == attr_name) and (self.attr_value == attr_value)
        else:
            return True
        
    def modify_active_state(self, state):
        self.active = state
    
    def get_active_state(self):
        return self.active
    
if __name__ == "__main__":
    print("This is decison tree code..")
