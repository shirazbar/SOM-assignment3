"""
Created on June 2022

@author: Shiraz Nave, Eden Meidan
"""

import numpy as np
import pandas as pd
import random
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import matplotlib.cm as cm
from tabulate import tabulate

# Hyperparameters
alpha = 0.3
neighbors_impact = {"first": 0.7, "second": 0.3}

# Globals
df = None
dict_city_vec = None
hexes = None
indices = None
clusters = defaultdict(list) # cluster id : all samples belong to that cluster
prev_clusters = None
cluster_weights = {} # cluster id : vector of cell (center)

# normalize vectors of the train and the test examples
def normalization_min_max(input_elections):
    norm_elections = [[]] * len(input_elections)
    min_elect = np.min(input_elections, axis=0)
    max_elect = np.max(input_elections, axis=0)
    for i in range(len(input_elections)):
        norm_elections[i] = (input_elections[i] - min_elect) / (max_elect - min_elect)

    return norm_elections

# creates hexagon of 61 cells around x,y,x axes 61 cells one per cluster - cluster_id 0:60
def set_representation():
    global hexes, indices
    radius = 5
    deltas = [[1,0,-1],[0,1,-1],[-1,1,0],[-1,0,1],[0,-1,1],[1,-1,0]]
    hexes = []
    indices = {}
    index = 0
    
    for r in range(radius):
        x = 0
        y = -r
        z = +r
        hexes.append((x,y,z))
        indices[(x,y,z)] = index   # Or store objects here
        index += 1
        # go through all neighbors
        for j in range(6):
            if j==5: # out of board range
                num_of_hexes_in_edge = r-1
            else:    # in board range
                num_of_hexes_in_edge = r
            for i in range(num_of_hexes_in_edge):
                x = x+deltas[j][0]
                y = y+deltas[j][1]
                z = z+deltas[j][2]
                hexes.append((x,y,z))
                indices[(x,y,z)] = index   # Or store objects here
                index += 1
    hexes = tuple(hexes)

# initializes data - inputs csv into df and dicts accordingly for easy acsess
def initialize(filename):
    global df, dict_city_vec, cluster_weights
    df = pd.read_csv(filename) # import csv file in df
    df.iloc[:,1:] = normalization_min_max(df.iloc[:,1:].to_numpy()) # normilize df
    dict_city_vec = {}
    # vectors dict city : vector nums
    for index, example in df.iterrows():
        city = example['Municipality']
        sample = example['Economic Cluster':'Tikva Hadasha']
        dict_city_vec[city] = sample.to_numpy()
    set_representation()
    chosen_cities = random.sample(dict_city_vec.keys(), len(indices))
    random.shuffle(chosen_cities)
    for index in indices.values():
        cluster_weights[index] = dict_city_vec[chosen_cities[index]].copy()
        
# returns the cluster_id of the closest cluster to the vector of a city being checked
def find_best_cell(vector, k=1):
    distances = [] 
    for cell in cluster_weights:
        distances.append((np.linalg.norm(vector - cluster_weights[cell]),cell))
    distances.sort()
    best_cell = distances[0][1] # dist,cell index
    if k==1: 
        return best_cell
    elif k==2: 
        second_best = distances[1][1]
        return best_cell, second_best

# updates the cluster's vectors
# wi = wi + alpha *  (xi - wi)
def update_weights(weights, sample, alpha, strength = 1):
    for i in range(len(weights)):
            weights[i] = weights[i] + (alpha * (sample[i] - weights[i]) * strength)
    return weights

# gets all possible neighbors around a certain location - max 6
def get_neighbors(location):
    x, y, z = location
    return {(x, y-1, z+1),
            (x+1, y-1, z),
            (x+1, y, z-1),
            (x, y+1, z-1),
            (x-1, y+1, z),
            (x-1, y, z+1)}

# updates cluster_weights dict around a certain cell
# doesn't undate the same neighbor more then once.
def update_neighbors_weight(location, sample, alpha, checked = []):
    checked = set()
    checked.add(location)
    ## first_circle of neighbors
    first_circle = get_neighbors(location)
    # for each possible neighbor update the cluster's vectors
    for neighbor in first_circle:
        if neighbor in indices:
            index = indices[neighbor]
            cluster_weights[index] = update_weights(cluster_weights[index], sample, alpha, strength=neighbors_impact["first"])
            checked.add(neighbor)
    ## second_circle of neighbors
    second_circle = [get_neighbors(location) for location in first_circle]
    for neighbors_set in second_circle:
        for neighbor in neighbors_set.difference(checked):
            if neighbor in indices:
                index = indices[neighbor]
                cluster_weights[index] = update_weights(cluster_weights[index], sample, alpha, strength=neighbors_impact["second"])
                checked.add(neighbor)

def get_border(c, clusters): # returns whether or not to border the cell based on its vector's length 
    if len(clusters[indices[c]]) == 0:
        return 'k'
    return "None"

# plots board
def plot(cluster_weights, clusters): 
    lst = {c:cluster_weights[c][0] for c in cluster_weights.keys()}
    
    # the values of the economy are between 0-1 since we performed normalization 
    # therefore, the avg of cities' economy in a certain cluster is also limited between 0-1
    norm = matplotlib.colors.Normalize(vmin=0., vmax=1., clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Reds)
    
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    
    for c, cluster_id in indices.items():
        hexagon = RegularPolygon((c[0], 2. * np.sin(np.radians(60)) * (c[1] - c[2]) /3.), numVertices=6, radius=np.sqrt(1/3), label=1, facecolor=mapper.to_rgba(lst[indices[c]]) ,orientation=np.radians(30), alpha=1, linewidth=1., edgecolor=get_border(c, clusters))
        ax.add_patch(hexagon)
        ax.text(c[0], 2. * np.sin(np.radians(60)) * (c[1] - c[2]) /3., cluster_id, ha='center', va='center', size=7)
    plt.autoscale(enable = True)
    plt.show()
    
    
# calacuates the normilazed distance between each city (city_vector) and the 
# cluster_vector it was belonged to. Normilaze by the amount of cities in the cluster.
def distance_error():
    avg_distance = 0
    for city, cluster_id in dict_city_cluster_id.items():
        city_vector = dict_city_vec[city]
        cluster_vector = cluster_weights[cluster_id]
        avg_distance += np.linalg.norm(city_vector - cluster_vector)
    avg_distance /= len(dict_city_cluster_id)
    return avg_distance

# topology provides quality about the density of the map.
def topological_error():
    avg = 0    
    for city, city_vec in dict_city_vec.items():
        best, second_best = find_best_cell(city_vec, k=2)
        avg += np.linalg.norm(np.array(best) - np.array(second_best))
    avg /= len(dict_city_vec)
    return avg

def total_error():
    return 0.2*topological_error() + 0.8*distance_error()
     
# the program ########################################################################################## 

dict_city_cluster_id = {}
solutions = []    
print("Enter the full path and filename for Self Organizing Map (e.g 'Elec_24.csv'): ")
filename = input()
try:
    initialize(filename)
    print(" ")
    print("Running 10 different attempts to find best solution...")
    for i in range(10):
        step = 0
        #initialize()
        initialize(filename)
        clusters = defaultdict(list) # cluster id : all samples belong to that cluster
        prev_clusters = None
        while clusters != prev_clusters and step < 70: # limit to 30 improvments
            prev_clusters = clusters
            clusters = defaultdict(list) # cluster id : all samples belong to that cluster
            cities_keys_different_order = list(dict_city_vec.keys())
            random.shuffle(cities_keys_different_order) # in order to avoid overfitting to the order of the data
            for city in cities_keys_different_order:
                sample = dict_city_vec[city] # vec (15,)
                best_cell = find_best_cell(dict_city_vec[city])
                clusters[best_cell].append(city)
                # update cell vector
                cluster_weights[best_cell] = update_weights(cluster_weights[best_cell], dict_city_vec[city], alpha)
                dict_city_cluster_id[city] = best_cell
                # update neighbors
                update_neighbors_weight(hexes[best_cell], dict_city_vec[city], alpha)
            step += 1
        solutions.append((total_error(), cluster_weights, clusters))
        print("Execution number:", i+1, " Error Evaluation:", round(total_error(),4))
    best_error, best_clusters_weights, best_clusters = min(solutions)
    print("Best solution found with Error Evaluation:", round(best_error,4))
    print("Plot and clusters info below ...")
    print(" ")
    # table output of bet solotion clusters
    table_data = [['Cluster ID', 'Location', 'Cities']]
    for cluster_id in best_clusters_weights.keys():
        location = hexes[cluster_id]
        cities = '\n'.join(best_clusters[cluster_id])
        table_data.append([cluster_id, location, cities])
    print(tabulate(table_data, tablefmt='fancy_grid'))
    plot(best_clusters_weights, best_clusters)
    
except:
    print("No such file. Please try again")

#########################################################################################################








