      ################################################################## 
      # script for illustrating hierarchical partition from successive   
      # solutions of generalized minimal partition problems solved with  
      # cut-pursuit                                                      
      #                                                                 
"""
Within each level, the components are identified from 0 to V_l - 1, where V_l
is the number of components at level l (V stands for vertices)

Hierarchical partition graph is represented with the following structure:

 - graph[l]: reduced graph at level l (forward-star representation, two arrays
   of indices of length respectively V_l+1 and E_l, where E_l is the number of
   edges at level l)
 - coarse_to_fine[l]: for each component at level l, the list of components at
   level l - 1 constituing it (list of arrays of indices)
 - fine_to_coarse[l]: for each component at level l, its assignment to the
   component at level l + 1 comprising it (array of indices)

Other information that are used for the hierarchical partition which may be
useful in analysing the result:

 - values[l]: values (features) of the components at level l (array of size
   D-by-V_l if the features are D-dimensionnal)
 - comp_weights[l]: importance weights attributed to the components at level i
   (array of length V_l)
 - edge_weights[l]: importance weights on the edges of the reduced graph at
   level l (array of length E_l)

Level 0 is the raw data: graph[0] is the main graph, coarse_to_fine[0] does not
exist, fine_to_coarse[0] are components assignments after first solution of
generalized minimal partition problem with cut-pursuit (with the smallest
regularization);
values[0] are all observations, and comp_weights[0] and edge_weights[0] are
usually all one.

From
    (values[l], graph[l], comp_weights[l], edge_weights[l]),
the solution of the generalized minimal partition problem with cut-pursuit
gives
    (values[l+1], graph[l+1], comp_weights[l+1], edge_weights[l+1],
     fine_to_coarse[l], coarse_to_fine[l+1]);
in the current documentation of the python interface, values[l+1] is the output
`rX`, fine_to_coarse[l] is `Comp`, coarse_to_fine[l+1] is `List`, graph[l+1]
and edge_weights[l+1] are in `Graph`, and comp_weights[l+1] must be recomputed
from comp_weights[l] and coarse_to_fine.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from pycut_pursuit.cp_d0_dist import cp_d0_dist

##  problem parameters
Ds = 2 # dimension of points positions; 2D eases graphical representation
Df = 3 # dimension of points features; unity eases graphical representation
V = 1000 # number of points, V_0
knn = 5 # number of nearest neighbors for main graph connectivity
regularizations = [.005, .05]; # regularization parameter
# spatial_weight = .2; # importance of space vs. features similarity
spatial_weight = .2; # importance of space vs. features similarity
min_comp_weights = [5.0, 50.0]; # force minimum components sizes

##  draw observations
positions = np.random.rand(Ds, V)
features = np.random.rand(Df, V)
features = features/features.sum(axis=0)
# to introduce some meaningful structure, sort the data so that the (first
# coordinates of) values correlates with the (first coordinates of) positions
positions = positions[:, np.argsort(positions[0, :])]
features = features[:, np.argsort(features[0, :])]

##  draw (spatial) nearest neighbors graph
adj_mat = NearestNeighbors(n_neighbors=knn).fit(positions.T).kneighbors_graph()
first_edge = adj_mat.indptr
adj_vertices = adj_mat.indices

##  initialize hierarchical partition
values = [np.asfortranarray(np.concatenate((positions, features)))]
graph = [(first_edge, adj_vertices)]
comp_weights = [None] # could be changed to nonhomogeneous weights
edge_weights = [1]    # could be changed to nonhomogeneous weights
coarse_to_fine = [None] # no finer level than the first
fine_to_coarse = []

##  successive minimal partition problems
# loss = Df + Ds # quadratic fidelity term
# coor_weights = np.concatenate((spatial_weight*np.ones(Ds), np.ones(Df)))
loss = Ds # quadratic on space KL on features
coor_weights = np.concatenate((spatial_weight*np.ones(Ds), [1.0]))
for l in range(len(regularizations)):
    Comp, rX, List, Graph = cp_d0_dist(loss, values[l],
        graph[l][0], graph[l][1],
        edge_weights=edge_weights[l]*regularizations[l],
        vert_weights=comp_weights[l], coor_weights=coor_weights,
        min_comp_weight=min_comp_weights[l], compute_List=True,
        compute_Graph=True)
    values.append(rX)
    graph.append((Graph[0], Graph[1].astype("uint32")))
    if comp_weights[l] is None:
        comp_weights.append(np.array([len(L) for L in List], dtype="float"))
    else:
        comp_weights.append(np.array([sum(comp_weights[l][L]) for L in List]))
    edge_weights.append(Graph[2]/regularizations[l])
    fine_to_coarse.append(Comp)
    coarse_to_fine.append(List)
fine_to_coarse.append(None) # no coarser than the last

##  graphical representation of the result
# representation of partitioning with colors in HSV space:
# level 2 partitioning is encoded with different hues
# level 1 partitioning (hierarchically included in level 2) is further encoded
# with different saturations
colors = [np.zeros((v.shape[1], 3)) for v in values]
hue = 0;
brighness = .8
for comp_num_2 in range(len(coarse_to_fine[2])):
    hue = min(1., hue + 1/len(coarse_to_fine[2]))
    colors[2][comp_num_2, :] = [hue, 1., 1.]
    saturation = .1
    comp_list_2 = coarse_to_fine[2][comp_num_2]
    for comp_num_1 in comp_list_2:
        saturation = min(1., saturation + .9/len(comp_list_2))
        colors[1][comp_num_1, :] = [hue, saturation, 1.]
        colors[0][coarse_to_fine[1][comp_num_1], :] = [hue, saturation, .7]
fig = plt.figure(1)
fig.clear()
# plot the main graph
start_vertex = np.repeat(range(V), first_edge[1:] - first_edge[:-1])
plt.plot(np.vstack((positions[0, start_vertex], positions[0, adj_vertices])),
         np.vstack((positions[1, start_vertex], positions[1, adj_vertices])),
         color='black', linewidth=1, zorder=1)
# plot all points at all levels
# first feature encoded with point size
# level 0 are raw observations, with white contours
# levels 1 and 2 centroids gets gray and black contours
for l in range(3):
    plt.scatter(values[l][0, :], values[l][1, :], s=100*values[l][Ds, :],
                c=hsv_to_rgb(colors[l]), zorder=2+l,
                edgecolors=[1-l/2, 1-l/2, 1-l/2])
plt.axis("off")
