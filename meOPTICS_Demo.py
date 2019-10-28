## General imports
import numpy as np
import pandas as pd
import os,inspect

# Get this current script file's directory:
loc = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# Set working directory
os.chdir(loc)
# from myFunctions import gen_FTN_data
# from meSAX import *

# from dtw_featurespace import *
# from dtw import dtw
# from fastdtw import fastdtw

# to avoid tk crash
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## my colors

color_list = ['gold', 'darkcyan','slateblue', 'hotpink', 'indigo', 'firebrick', 'skyblue', 'coral', 'sandybrown', 'mediumpurple',  'forestgreen', 'magenta', 'seagreen', 'greenyellow', 'roaylblue', 'gray', 'lightseagreen']

# Matplotlib default color cycler
# matplotlib.rcParams['axes.prop_cycle']
default_color_list = []
for obj in matplotlib.rcParams['axes.prop_cycle']:
    default_color_list.append(obj['color'])
# combine the two color lists
my_colors =  default_color_list
[my_colors.append(c) for c in color_list]

# my_colors = np.array(['#1f77b4','#2ca02c','#ff7f0e'])


my_colors = np.array(my_colors)

## generate data

# set random seed
np.random.seed(0)

# my sample data
x_mean1 = 0
y_mean1 = 0

x_mean2 = 45
y_mean2 = 13

x_mean3 = 7
y_mean3 = 40

N1 = 100
N2 = 100
N3 = 100

# coords1 = np.random.uniform(0,12,(N1,2))
# coords2 = np.random.uniform(0,5,(N2,2))
coords1 = np.random.randn(N1,2) * 16
coords2 = np.random.randn(N2,2) * 4
coords3 = np.random.randn(N3,2) * 1
outliers = np.array([15,15,23,12]).reshape(2,2)
coords = np.empty((N1+N2+N3+outliers.shape[0],2))


coords[:N1] =  coords1 + (x_mean1,y_mean1)
coords[N1:(N1+N2)] =  coords2 + (x_mean2,y_mean2)
coords[(N1+N2):-2] =  coords3 + (x_mean3,y_mean3)
coords[-2:] = outliers

# # sklearn example data
# n_points_per_cluster = 250
# 
# C1 = [-5, -2] + .8 * np.random.randn(n_points_per_cluster, 2)
# C2 = [4, -1] + .1 * np.random.randn(n_points_per_cluster, 2)
# C3 = [1, -2] + .2 * np.random.randn(n_points_per_cluster, 2)
# C4 = [-2, 3] + .3 * np.random.randn(n_points_per_cluster, 2)
# C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
# C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
# coords = np.vstack((C1, C2, C3, C4, C5, C6))




## Use in detail

from meOPTICS import DataPoint
from meOPTICS import *


from scipy.spatial import distance
dist = distance.minkowski
D = gen_dist_mat(coords,dist)

eps = 24
eps2 = 5
MinPts = 15



# initialize data points
DPs = [] # list of data points
for i,datapoint in enumerate(coords):
    p = DataPoint(i)
    DPs.append(p)
# compute all nearest neighbors
NN,NN_dists = nearest_neighbors(None,MinPts,D)



order_list = get_order(DPs,D,eps,MinPts,NN_dists)
cluster_list = cluster(order_list,eps2,MinPts)

cluster_labels = np.array([o.clusterID for o in cluster_list])
cluster_index = np.array([o.index for o in cluster_list])


# plot
plt.figure()
plt.scatter(coords[cluster_index,0],coords[cluster_index,1],color = my_colors[cluster_labels],alpha = 0.8)
# for i in range(coords.shape[0]):
#     plt.scatter(coords[:,0],coords[:,1],color = my_colors[lof_labels[i]])

# ax.set_xlim((0,20))
# ax.set_ylim((0,20))
plt.gca().set_aspect('equal', adjustable='box') # making the x and y scale the same
plt.title('My OPTICS')
plt.show()

## simple use: class object


from meOPTICS import DataPoint
from meOPTICS import meOPTICS
from meOPTICS import gen_dist_mat
from scipy.spatial import distance
dist = distance.minkowski

# parameters
D = gen_dist_mat(coords,dist)
eps = 24
eps2 = 11
MinPts = 15

# Run OPTICS
optics = meOPTICS(coords,eps,eps2,MinPts,D)
optics_list = optics.fit() # the order_list with cluster ID/label for the DataPoints(class)

cluster_labels = np.array([o.clusterID for o in optics_list])
cluster_index = np.array([o.index for o in optics_list])
cluster_r_dists = np.array([o.r_dist for o in optics_list])
cluster_markers = np.array(['o' if o.clusterID!=-1 else 'x' for o in optics_list])


# scatter plot
plt.figure()
# plt.scatter(coords[cluster_index,0],coords[cluster_index,1],color = my_colors[cluster_labels],alpha = 0.8)

for i,index in enumerate(cluster_index):
    plt.scatter(coords[index,0],coords[index,1],color = my_colors[cluster_labels[i]],
                alpha = 0.8,marker=cluster_markers[i])
    # plt.text(coords[index,0],coords[index,1],str(i)) # plot text labels


# # add in links
# N = cluster_index.shape[0]
# for i in range(N-1):
#     if cluster_labels[i] == cluster_labels[i+1]:
#         plt.plot(coords[cluster_index[[i,i+1]],0],coords[cluster_index[[i,i+1]],1],
#                 color = my_colors[cluster_labels[i]],alpha = 0.8)



# ax.set_xlim((0,20))
# ax.set_ylim((0,20))
plt.gca().set_aspect('equal', adjustable='box') # making the x and y scale the same
plt.title('My OPTICS')
plt.show()

# reachability-distance plot
plt.figure()
# plt.scatter(range(cluster_r_dists.shape[0]),cluster_r_dists,color = my_colors[cluster_labels],alpha = 0.8)

for i,r_dist in enumerate(cluster_r_dists):
    plt.scatter(i,r_dist,color = my_colors[cluster_labels[i]],alpha = 0.8,marker=cluster_markers[i])
                
plt.title('reachability-distance plot')
plt.show()



## Compare with DBSCAN
# k-dist plot
# from sklearn.neighbors import kneighbors_graph
k = MinPts # k-th neighbor
'''
inputs:
    D: distance matrix(N by N)
    k: k-th neighbor distance
'''
def k_dist(D,k = 4):
    import numpy as np
    D = np.array(D)
    N = D.shape[0]
    # initialize k_dist vector
    k_dist = np.zeros((N,1))
    for i in range(N):
        row = list(D[i,:])
        for j in range(k): # remove min(row) k times, not k-1 times, because closest is always itself!
            row.remove(min(row))
        k_dist[i] = min(row)
    return(k_dist)

k_distances = k_dist(D,k=k)
k_distances = np.sort(k_distances,axis = 0)

plt.figure()
plt.plot(k_distances)
plt.show()


# DBSCAN
from sklearn.cluster import DBSCAN

# If metric is “precomputed”, X is assumed to be a distance matrix and must be square
# Parameters: eps and min_samples are set by inspecting the k-dist plot!

dbscan = DBSCAN(eps=15.4, min_samples=k,metric = 'precomputed').fit(D)
dbscan.labels_
clusters = list(set(dbscan.labels_))
print(clusters)
for cluster in clusters:
    cluster_size = len(dbscan.labels_[dbscan.labels_== cluster])
    print('cluster{} \t has a size of {}'.format(cluster,cluster_size))

plt.figure()
plt.hist(dbscan.labels_)
plt.show()


# scatter plot
plt.figure()
# plt.scatter(coords[:,0],coords[:,1],color = my_colors[dbscan.labels_],alpha = 0.8)

marker_list = ['o' if x != -1 else 'x' for x in dbscan.labels_]
for i,coord in enumerate(coords):
    plt.scatter(coord[0],coord[1],color = my_colors[dbscan.labels_[i]],alpha = 0.8,marker=marker_list[i])
# legend labels
for i in range(len(set(dbscan.labels_))-1):
    plt.scatter([],[],color = my_colors[i],alpha = 0.5,label = 'cluster{}'.format(i))
plt.scatter([],[],color = my_colors[-1],alpha = 0.5,marker='x',label = 'noise')

plt.gca().set_aspect('equal', adjustable='box') # making the x and y scale the same
plt.title('DBSCAN')
plt.legend()
plt.show()


## Auto cluster with xi-steep area method


from meOPTICS import DataPoint
from meOPTICS import meOPTICS
from meOPTICS import gen_dist_mat
from scipy.spatial import distance
dist = distance.minkowski

# parameters
D = gen_dist_mat(coords,dist)
eps = 24
eps2 = 5
MinPts = 15
xi = 0.01

# Run OPTICS
optics = meOPTICS(coords,eps,eps2,MinPts,D = None,xi = xi)
order_list = optics.get_order()
cluster_list = optics.auto_cluster(order_list)

# optics_list = optics.fit() # the order_list with cluster ID/label for the DataPoints(class)
# cluster_labels = np.array([o.clusterID for o in optics_list])
# cluster_index = np.array([o.index for o in optics_list])
# cluster_r_dists = np.array([o.r_dist for o in optics_list])



# scatter plot
plt.figure()
# generate clusters and legend labels
clusters = []
for i,cluster in enumerate(cluster_list):
    clusters.append(range(cluster[0],cluster[1]))
    plt.scatter([],[],color = my_colors[i],alpha = 0.5,label = 'cluster{}:{}'.format(i,clusters[i]))
plt.scatter([],[],color = my_colors[-1],alpha = 0.5,marker='x',label = 'noise')
# plot
for i,p in enumerate(order_list):
    for j,cluster in enumerate(clusters):
        if i in cluster:
            p.clusterID = j
            coord = coords[p.index]
            plt.scatter(coord[0],coord[1],color = my_colors[j],alpha = 0.5)
            
    if p.clusterID is None or p.clusterID==-1: # noise
        p.clusterID = -1
        coord = coords[p.index]
        plt.scatter(coord[0],coord[1],color = my_colors[-1],alpha = 0.5,marker='x')


plt.legend(bbox_to_anchor=(1.05, 1))
plt.gca().set_aspect('equal', adjustable='box') # making the x and y scale the same
'''
To adjust legend box location and settings:
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
'''
# plt.gca().set_xlim((-45,105))
# plt.gca().set_ylim((-60,50))

plt.title('My OPTICS\nAutoExtractClusters')
plt.show()        


# reachability-distance plot
plt.figure()
# reachability-distances
cluster_r_dists = np.array([o.r_dist for o in order_list])

# cluster labels
cluster_labels = np.ones((D.shape[0],),dtype=int) * -1 # if unset, default is -1(noise)
for i,o in enumerate(order_list):
    for j,cluster in enumerate(clusters):
        if i in cluster: cluster_labels[i] = j
# legend labels
for i,cluster in enumerate(cluster_list):
    plt.scatter([],[],color = my_colors[i],alpha = 0.5,label = 'cluster{}:{}'.format(i,clusters[i]))
plt.scatter([],[],color = my_colors[-1],alpha = 0.5,marker='x',label = 'noise')
# plot

plt.scatter(range(cluster_r_dists.shape[0]),cluster_r_dists,color = my_colors[cluster_labels],alpha = 0.4)
plt.legend(bbox_to_anchor=(1.01, 1))
plt.title('reachability-distance plot\nauto extract')
plt.show()



## get only the main clusters(non-hierarchical cluster structure, only the highest clusters)

max_clusters = cluster_list

i = 0
while i < len(max_clusters):
    flag = False
    ci = max_clusters[i]
    for j,cj in enumerate(max_clusters):
        if ((cj[0] < ci[0] and cj[1] >= ci[1]) or (cj[0] <= ci[0] and cj[1] > ci[1])):
            max_clusters.pop(i)
            flag = True
            break
    if not flag: i += 1



# scatter plot
plt.figure()
# generate clusters and legend labels
clusters = []
for i,cluster in enumerate(max_clusters):
    clusters.append(range(cluster[0],cluster[1]))
    plt.scatter([],[],color = my_colors[i],alpha = 0.5,label = 'cluster{}:{}'.format(i,clusters[i]))
plt.scatter([],[],color = my_colors[-1],alpha = 0.5,marker='x',label = 'noise')
# plot
for i,p in enumerate(order_list):
    for j,cluster in enumerate(clusters):
        if i in cluster:
            p.clusterID = j
            coord = coords[p.index]
            plt.scatter(coord[0],coord[1],color = my_colors[j],alpha = 0.5)
            
    if p.clusterID is None or p.clusterID==-1: # noise
        p.clusterID = -1
        coord = coords[p.index]
        plt.scatter(coord[0],coord[1],color = my_colors[-1],alpha = 0.5,marker='x')


plt.legend()
plt.gca().set_aspect('equal', adjustable='box') # making the x and y scale the same
plt.title('My OPTICS\nAutoExtractClusters(max clusters)')
plt.show()        


# reachability-distance plot
plt.figure()
# reachability-distances
cluster_r_dists = np.array([o.r_dist for o in order_list])

# cluster labels
cluster_labels = np.ones((D.shape[0],),dtype=int) * -1 # if unset, default is -1(noise)
for i,o in enumerate(order_list):
    for j,cluster in enumerate(clusters):
        if i in cluster: cluster_labels[i] = j
# legend labels
for i,cluster in enumerate(cluster_list):
    plt.scatter([],[],color = my_colors[i],alpha = 0.5,label = 'cluster{}:{}'.format(i,clusters[i]))
plt.scatter([],[],color = my_colors[-1],alpha = 0.5,marker='x',label = 'noise')
# plot

plt.scatter(range(cluster_r_dists.shape[0]),cluster_r_dists,color = my_colors[cluster_labels],alpha = 0.4)
plt.legend()
plt.title('reachability-distance plot\nauto extract(max clusters)')
plt.show()




##

SUAset,SDAset = optics.max_steep_area(order_list)















