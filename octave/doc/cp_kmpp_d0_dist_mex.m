function [Comp, rX, it, Obj, Time, Dif] = cp_kmpp_d0_dist_mex(loss, Y, ...
    first_edge, adj_vertices, options)
%
%        [Comp, rX, it, Obj, Time, Dif] = cp_kmpp_d0_dist_mex(loss, Y,
%   first_edge, adj_vertices, options)
%
% Cut-pursuit algorithm with d0 (weighted contour length) penalization, with a
% loss akin to a distance:
%
% minimize functional over a graph G = (V, E)
%
%        F(x) = sum_v loss(y_v, x_v) + ||x||_d0
%
% where for each vertex, y_v and x_v are D-dimensional vectors, the loss is
% either the sum of square differences or smoothed Kullback-Leibler divergence
% (equivalent to cross-entropy in this formulation); see the 'loss' attribute,
%   and ||x||_d0 = sum_{uv in E} w_d0_uv ,
%
% using greedy cut-pursuit approach with splitting initialized with k-means++.
%
% Available data-fidelity loss include:
%
% quadratic:
%      f(x) = ||y - x||_{l2,W}^2 ,
% where W is a diagonal metric (separable product along ℝ^V and ℝ^D),
% that is ||y - x||_{l2,W}^2 = sum_{v in V} w_v ||x_v - y_v||_{l2,M}^2
%                            = sum_{v in V} w_v sum_d m_d (x_vd - y_vd)^2;
%
% (smoothed, weighted) Kullback-Leibler divergence (equivalent to
% cross-entropy) on the probability simplex:
%     f(x) = sum_v w_v KLs_m(x_v, y_v),
% with KLs(y_v, x_v) = KL_m(s u + (1 - s) y_v ,  s u + (1 - s) x_v), where
%     KL is the regular Kullback-Leibler divergence,
%     u is the uniform discrete distribution over {1,...,D}, and
%     s = loss is the smoothing parameter
%     m is a diagonal metric weighting the coordinates;
% it yields
%     KLs_m(y_v, x_v) = - H_m(s u + (1 - s) y_v)
%         - sum_d m_d (s/D + (1 - s) y_{v,d}) log(s/D + (1 - s) x_{v,d}) ,
% where H_m is the (weighted) entropy, that is H_m(s u + (1 - s) y_v)
%       = - sum_d m_d (s/D + (1 - s) y_{v,d}) log(s/D + (1 - s) y_{v,d}) ;
% note that the choosen order of the arguments in the Kullback-Leibler
% does not favor the entropy of x (H_m(s u + (1 - s) y_v) is a constant),
% hence this loss is actually equivalent to cross-entropy.
%
% NOTA: by default, components are identified using uint16 identifiers; this
% can be easily changed in the mex source if more than 65535 components are
% expected (recompilation is necessary)
%
% INPUTS: real numeric type is either single or double, not both;
%         indices are C-style (start at 0) of type uint32
%
% loss - 1 for quadratic, 0 < loss < 1 for smoothed Kullback-Leibler
% Y - observations, (real) D-by-V array, column-major format;
%     for Kullback-Leibler loss, the value at each vertex is supposed to lie on
%     the probability simplex 
% first_edge, adj_vertices - forward-star graph representation:
%     edges are numeroted (C-style indexing) so that all vertices originating
%         from a same vertex are consecutive;
%     for each vertex, first_edge indicates the first edge starting from the
%         vertex (or, if there are none, starting from the next vertex);
%         (uint32) array of length V + 1, the first value is always zero and
%         the last value is always the total number of edges E;
%     for each edge, adj_vertices indicates its ending vertex, (uint32) array
%         of length E
% options - structure with any of the following fields [with default values]:
%     edge_weights [1.0], vert_weights [none], coor_weights [none],
%     cp_dif_tol [1e-3], cp_it_max [10], K [2], split_iter_num [2],
%     split_damp_ratio [1.0], kmpp_init_num [3], kmpp_iter_num [3],
%     verbose [true], max_num_threads [none], balance_parallel_split [true]
% edge_weights - (real) array of length E or scalar for homogeneous weights
% vert_weights - weights on vertices (w_v above); (real) array of length V
% coor_weights - weights on coordinates (m_d above); (real) array of length D
% cp_dif_tol - stopping criterion on iterate evolution; algorithm stops if
%     relative changes (that is, relative dissimilarity measures defined by the
%     choosen loss between successive iterates and between current iterate and
%     observation) is less than dif_tol; 1e-3 is a typical value
% cp_it_max  - maximum number of iterations (graph cut, subproblem, merge)
%     10 cuts solve accurately most problems
% K - number of alternative values considered in the split step
% split_iter_num - number of partition-and-update iterations in the split step
% split_damp_ratio - edge weights damping for favoring splitting; edge
%     weights increase in linear progression along partition-and-update
%     iterations, from this ratio up to original value; real scalar between 0
%     and 1, the latter meaning no damping
% kmpp_init_num - number of random k-means initializations in the split step
% kmpp_iter_num - number of k-means iterations in the split step
% verbose - if nonzero, display information on the progress
% max_num_threads - if greater than zero, set the maximum number of threads
%     used for parallelization with OpenMP
% balance_parallel_split - if true, the parallel workload of the split step 
%     is balanced; WARNING: this might trade off speed against optimality
%
% OUTPUTS: indices are C-style (start at 0)
%
% Comp - assignement of each vertex to a component, (uint16) array of length V
% rX - values of each component of the minimizer, (real) array of length rV;
%     the actual minimizer can be reconstructed with X = rX(Comp + 1);
% it - actual number of cut-pursuit iterations performed
% Obj - the values of the objective functional along iterations;
%     array of length it + 1
% Time - if requested, the elapsed time along iterations;
%     array of length cp_it + 1
% Dif  - if requested, the iterate evolution along iterations;
%     array of length cp_it
% 
% Parallel implementation with OpenMP API.
%
% L. Landrieu and G. Obozinski, Cut Pursuit: fast algorithms to learn
% piecewise constant functions on general weighted graphs, SIAM Journal on
% Imaging Science, 10(4):1724-1766, 2017
%
% L. Landrieu et al., A structured regularization framework for spatially
% smoothing semantic labelings of 3D point clouds, ISPRS Journal of
% Photogrammetry and Remote Sensing, 132:102-118, 2017
%
% Hugo Raguet 2019, 2020
