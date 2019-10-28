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

# sklearn example data
n_points_per_cluster = 50

C1 = [-5, -2] + .8 * np.random.randn(n_points_per_cluster, 2)
C2 = [4, -1] + .1 * np.random.randn(n_points_per_cluster, 2)
C3 = [1, -2] + .2 * np.random.randn(n_points_per_cluster, 2)
C4 = [-2, 3] + .3 * np.random.randn(n_points_per_cluster, 2)
C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
coords = np.vstack((C1, C2, C3, C4, C5, C6))


# scatter plot
plt.figure()
plt.scatter(coords[:,0],coords[:,1])
plt.xlim((-8,10))
plt.ylim((-8,10))
plt.show()





## OPTICS
# Get this current script file's directory:
loc = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# Set working directory
os.chdir(loc)

from meOPTICS import *
from meOPTICS import DataPoint
from meOPTICS import meOPTICS
from meOPTICS import gen_dist_mat
from scipy.spatial import distance
dist = distance.minkowski

# parameters
D = gen_dist_mat(coords,dist)
eps = 21096
eps2 = 10000
MinPts = 15
xi = 0.01

# Run OPTICS
optics = meOPTICS(coords,eps,eps2,MinPts,D = None,xi = xi)
order_list = optics.get_order()
cluster_list = optics.auto_cluster(order_list)


clusters = []
for i,cluster in enumerate(cluster_list):
    clusters.append(range(cluster[0],cluster[1]))


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

# plt.scatter(range(cluster_r_dists.shape[0]),cluster_r_dists,color = my_colors[cluster_labels],alpha = 0.4)
# plt.legend(bbox_to_anchor=(1.01, 1))

cluster_markers = np.array(['o' if o != -1 else 'x' for o in cluster_labels])
for i,r_dist in enumerate(cluster_r_dists):
    plt.scatter(i,r_dist,color = my_colors[cluster_labels[i]],alpha = 0.5,marker=cluster_markers[i])
    
# plt.legend(bbox_to_anchor=(1.01, 1))
plt.legend()
plt.title('reachability-distance plot\nauto extract')
plt.show()



# plot scatter
plt.figure()
for i,o in enumerate(order_list):
    plt.scatter(coords[o.index][0],coords[o.index][1], color = my_colors[cluster_labels[i]], alpha = 0.3,
    marker = cluster_markers[i])


for i in range(-1,len(set(cluster_labels))-1):
    plt.scatter([],[],color = my_colors[i],label = 'noise' if i==-1 else 'cluster{}'.format(i),
    marker = 'x' if i == -1 else 'o')

# plt.xlabel('Time[hour]')
# plt.ylabel('Heat load[W]')
plt.xlim((-8,10))
plt.ylim((-8,10))
plt.legend()
plt.show()


# # plot time-series
# plt.figure()
# for i,o in enumerate(order_list):
#     plt.plot(coords[o.index], color = my_colors[cluster_labels[i]], alpha = 0.3,
#     linestyle= '--' if cluster_labels[i]==-1 else '-')
# 
# for i in range(-1,len(set(cluster_labels))-1):
#     plt.plot([],color = my_colors[i],label = 'noise' if i==-1 else 'cluster{}'.format(i),
#     linestyle= '--' if i == -1 else '-')
# 
# # plt.xlabel('Time[hour]')
# # plt.ylabel('Heat load[W]')
# plt.legend()
# plt.show()




## max clusters in OPTICS
import copy
max_clusters = copy.deepcopy(cluster_list)

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

# generate clusters and legend labels
m_clusters = []
for i,cluster in enumerate(max_clusters):
    m_clusters.append(range(cluster[0],cluster[1]))


# reachability-distance plot
plt.figure()
# reachability-distances
cluster_r_dists = np.array([o.r_dist for o in order_list])

# cluster labels
max_cluster_labels = np.ones((D.shape[0],),dtype=int) * -1 # if unset, default is -1(noise)
for i,o in enumerate(order_list):
    for j,cluster in enumerate(m_clusters):
        if i in cluster: max_cluster_labels[i] = j
# legend labels
for i,cluster in enumerate(max_clusters):
    plt.scatter([],[],color = my_colors[i],alpha = 0.5,label = 'cluster{}:{}'.format(i,m_clusters[i]))
plt.scatter([],[],color = my_colors[-1],alpha = 0.5,marker='x',label = 'noise')
# plot

# plt.scatter(range(cluster_r_dists.shape[0]),cluster_r_dists,color = my_colors[max_cluster_labels],alpha = 0.4)

max_cluster_markers = np.array(['o' if o != -1 else 'x' for o in max_cluster_labels])
for i,r_dist in enumerate(cluster_r_dists):
    plt.scatter(i,r_dist,color = my_colors[max_cluster_labels[i]],alpha = 0.5,marker=max_cluster_markers[i])
    
plt.legend()
plt.xlabel('Data sample index')
plt.ylabel('Distance')
# plt.title('reachability-distance plot\nauto extract(max clusters)')
plt.show()



# plot scatter
plt.figure()
for i,o in enumerate(order_list):
    plt.scatter(coords[o.index][0],coords[o.index][1], color = my_colors[max_cluster_labels[i]], alpha = 0.3,
    marker = cluster_markers[i])


for i in range(-1,len(set(max_cluster_labels))-1):
    plt.scatter([],[],color = my_colors[i],label = 'noise' if i==-1 else 'cluster{}'.format(i),
    marker = 'x' if i == -1 else 'o')

# plt.xlabel('Time[hour]')
# plt.ylabel('Heat load[W]')
plt.xlim((-8,10))
plt.ylim((-8,10))
plt.legend()
plt.show()



# # plot time-series
# plt.figure()
# for i,o in enumerate(order_list):
#     plt.plot(coords[o.index], color = my_colors[max_cluster_labels[i]], alpha = 0.3,
#     linestyle= '--' if max_cluster_labels[i]==-1 else '-')
# 
# for i in range(-1,len(set(max_cluster_labels))-1):
#     plt.plot([],color = my_colors[i],label = 'noise' if i==-1 else 'cluster{}'.format(i),
#     linestyle= '--' if i == -1 else '-')
# 
# plt.xlabel('Time[hour]')
# plt.ylabel('Heat load[W]')
# plt.legend()
# plt.show()
























