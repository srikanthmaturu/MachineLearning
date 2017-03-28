# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

from statistics import mean

class KMeans():
    def __init__(self, k):
        self.k = k
        self.generations = []
        
    def load_dataset(self, dataset):
        self.empty_datapoint = dataset.subset([])
        self.dataset = [DataPoint(dataset.subset([i])) for i in range(0,dataset.size)]
        
    def compute_initial_clusters(self):
        total_size = len(self.dataset)
        cluster_size = int(total_size/self.k)
        self.initial_clusters = Clusters(self.empty_datapoint)
        for i in range(0, self.k):
            cluster = Cluster(self.empty_datapoint, self.dataset[i*cluster_size:i*cluster_size + cluster_size])
            self.initial_clusters.add_cluster(cluster) 
        
    def run(self):
        self.compute_initial_clusters()
        self.initial_clusters.compute_within_cluster_average_dissimilarity()
        self.initial_clusters.compute_total_within_cluster_average_dissimilarity()
        print("\nIteration 0:")
        print('\nInitial Clusters: \n')
        self.initial_clusters.print_clusters()
        current_dissimilarity = self.initial_clusters.get_total_within_cluster_average_dissimilarity()
        terminate = False
        current_clusters = self.initial_clusters
        while(not terminate):
            next_gen_clusters = current_clusters.get_next_generation_clusters()
            self.generations.append(next_gen_clusters)
            dissimilarity = next_gen_clusters.get_total_within_cluster_average_dissimilarity()
            
            print('\n\nIteration no: ', len(self.generations))
            
            if(current_dissimilarity - dissimilarity < 0.04):
                terminate = True
                print("\n\n\nClustering completed..")
                print("Total clusters dissimilairity ", current_dissimilarity,'\n')
            else:
                print( ' Dissimilarity: ', dissimilarity, ' \n')
            
            #next_gen_clusters.print_clusters()
            next_gen_clusters.print_wiki_clusters()
            current_dissimilarity = dissimilarity
            current_clusters = next_gen_clusters
            
class Clusters():
    def __init__(self, empty_datapoint):
        self.k = 0
        self.clusters = []
        self.empty_datapoint = empty_datapoint
    
    def add_cluster(self, cluster):
        self.clusters.append(cluster)
        self.k += 1
    
    def compute_within_cluster_average_dissimilarity(self):
        for cluster in self.clusters:
            cluster.compute_within_cluster_average_dissimilarity()
            
    def compute_total_within_cluster_average_dissimilarity(self):
        self.total_within_cluster_average_dissimilarity = sum([cluster.get_within_cluster_average_dissimilarity() for cluster in self.clusters])
    
    def get_total_within_cluster_average_dissimilarity(self):
        return self.total_within_cluster_average_dissimilarity
    
    def get_next_generation_clusters(self):
        new_clusters = Clusters(self.empty_datapoint)
        for index in range(0, len(self.clusters)):
            new_cluster = Cluster(self.empty_datapoint)
            new_cluster.set_initial_mean_vector(self.clusters[index].get_mean_vector())
            new_clusters.add_cluster(new_cluster)
        
        for cluster in self.clusters:
            for datapoint in cluster.datapoints:
                new_clusters.add_datapoint_to_cluster(new_clusters.get_nearest_cluster_index(datapoint), datapoint)
        
        
        new_clusters.compute_mean_vectors()
        new_clusters.compute_within_cluster_average_dissimilarity()
        new_clusters.compute_total_within_cluster_average_dissimilarity()
        return new_clusters
        
    def get_nearest_cluster_index(self, datapoint):
        clusters_distances = [cluster.get_distance_to_initial_mean_vector(datapoint) for cluster in self.clusters]
        #print(clusters_distances)
        nearest_cluster_index = clusters_distances.index(min(clusters_distances))
        return nearest_cluster_index
    
    def compute_mean_vectors(self):
        for cluster in self.clusters:
            cluster.compute_mean_vector()
    
    def add_datapoint_to_cluster(self, cluster_index, datapoint):
        self.clusters[cluster_index].insert_point(datapoint)
    
    def print_clusters(self):
        print('No of clusters k: ', self.k)
        for index in range(0, len(self.clusters)):
            print("---Cluster : ", index)
            print("Mean vector: ")
            self.clusters[index].get_mean_vector().print_as_mean_vector()
            print("Cluster datapoints: ")
            self.clusters[index].print_cluster()
    
    def print_clusters_mean_vectors(self):
        for index in range(0, len(self.clusters)):
            self.clusters[index].get_mean_vector().print_as_mean_vector()
    
    def print_wiki_clusters(self):
        for index in range(0, len(self.clusters)):
            self.clusters[index].print_cluster_wiki_markup(index)
    
class Cluster():
    def __init__(self, empty_dataset, datapoints=''):
        self.size = 0
        self.initial_mean_vector = ''
        self.mean_vector = ''
        self.empty_datapoint = empty_dataset
        self.datapoints = datapoints
        if(datapoints!=''):
            self.compute_mean_vector()
        else:
            self.datapoints = []

    def set_initial_mean_vector(self, initial_mean_vector):
        self.initial_mean_vector = initial_mean_vector
    
    def insert_point(self, datapoint):
        self.datapoints.append(datapoint)
    
    def compute_mean_vector(self):
        vector = DataPoint(self.empty_datapoint.subset([]))
        for datapoint in self.datapoints:
            for index in range(0, len(datapoint.point.attributes)):
                if(not vector.point.attributes[index].attribute):
                    vector.point.attributes[index].attribute.append(datapoint.point.attributes[index].attribute[0])
                else:
                    vector.point.attributes[index].attribute[0] += datapoint.point.attributes[index].attribute[0]
        
        for index in range(0, len(vector.point.attributes)):
            vector.point.attributes[index].attribute[0] /= len(self.datapoints)
        
        self.mean_vector = vector
    
    def get_mean_vector(self):
        return self.mean_vector
    
    def get_initial_mean_vector(self):
        return self.initial_mean_vector
        
    def compute_within_cluster_average_dissimilarity(self):
        self.within_cluster_average_dissimilarity = mean([self.mean_vector.get_euclidean_distance(datapoint) for datapoint in self.datapoints])
    
    def get_within_cluster_average_dissimilarity(self):
        return self.within_cluster_average_dissimilarity
    
    def get_distance_to_cluster_center(self, datapoint):
        return self.mean_vector.get_euclidean_distance(datapoint)
    
    def get_distance_to_initial_mean_vector(self, datapoint):
        return self.initial_mean_vector.get_euclidean_distance(datapoint)
    
    def print_cluster(self):
        for datapoint in self.datapoints:
            datapoint.print_datapoint()
            #datapoint.print_as_mean_vector()
    
    def print_cluster_wiki_markup(self, cluster_number):
        print('\nAverage within cluster euclidean distance from cluster center: '+str(self.get_within_cluster_average_dissimilarity())+'\n')
        #self.datapoints[0].print_dataset_header_as_wiki_markup()
        header_markup = '{| class="wikitable sortable" align="center" style="margin: 0px" \n'
        header_markup = header_markup +  self.datapoints[0].get_dataset_header_as_wiki_markup()
        datapoints_markup = ''
        for datapoint in self.datapoints:
            datapoints_markup += '|-\n'+datapoint.get_wiki_markup()
        #print(datapoints_markup)
        #print('|}')
        main_table_markup = header_markup +'\n'+datapoints_markup+'\n'+'|}'
        final_table_markup = '{| class="wikitable" align="center" \n ! '+'Cluster:- '+str(cluster_number)+'\n|- \n||\n'+main_table_markup+'\n|}'
        print(final_table_markup)
        
        
class DataPoint():
    def __init__(self, raw_datapoint):
        self.point = raw_datapoint
    
    def get_euclidean_distance(self, datapoint):
        sum = 0
        for value1, value2 in zip(self.point.attributes, datapoint.point.attributes):
            sum += (value1.attribute[0] - value2.attribute[0])**2
        return sum**0.5
    
    def print_datapoint(self):
        self.point.print_dataset()
    
    def print_as_mean_vector(self):
        output = ''
        for attribute in self.point.attributes:
            output += " " + str(attribute.attribute[0])
        
        print(output)
    
    def print_as_wiki_markup(self):
        self.point.print_wiki_markup()
    
    def get_wiki_markup(self):
        return self.point.get_wiki_markup()
    
    def print_dataset_header_as_wiki_markup(self):
        self.point.print_header_wiki_markup()
    
    def get_dataset_header_as_wiki_markup(self):
        return self.point.get_header_wiki_markup()
    
if __name__ == "__main__":
    print("k-means module loaded..")