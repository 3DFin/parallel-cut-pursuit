function [Comp, rX, cp_it, Obj, Time, Dif] = cp_pfdr_d1_lsx_mex(loss, Y, ...
    first_edge, adj_vertices, options)
%
%        [Comp, rX, cp_it, Obj, Time, Dif] = cp_pfdr_d1_lsx_mex(loss, Y,
%   first_edge, adj_vertices, options)
%
% Cut-pursuit algorithm with d1 (total variation) penalization, with a
% separable loss term and simplex constraints:
%
% minimize functional over a graph G = (V, E)
%
%        F(x) = f(x) + ||x||_d1 + i_{simplex}(x)
%
% where for each vertex, x_v is a D-dimensional vector,
%       f is a separable data-fidelity loss
%       ||x||_d1 = sum_{uv in E} w_d1_uv (sum_d w_d1_d |x_ud - x_vd|),
% and i_{simplex} is the standard D-simplex constraint over each vertex,
%     i_{simplex} = 0 for all v, (for all d, x_vd >= 0) and sum_d x_vd = 1,
%                 = infinity otherwise;
%
% using cut-pursuit approach with preconditioned forward-Douglas-Rachford 
% splitting algorithm.
%
% Available separable data-fidelity loss include:
%
% linear
%       f(x) = - <x, y> ,  with  <x, y> = sum_{v,d} x_{v,d} y_{v,d} ;
%
% quadratic
%       f(x) = 1/2 ||y - x||_{l2,w}^2 ,
%   with  ||y - x||_{l2,w}^2 = sum_{v,d} w_v (y_{v,d} - x_{v,d})^2 ;
%
% smoothed Kullback-Leibler divergence (cross-entropy)
%     f(x) = sum_v w_v KLs(x_v, y_v),
% with KLs(y_v, x_v) = KL(s u + (1 - s) y_v ,  s u + (1 - s) x_v), where
%     KL is the regular Kullback-Leibler divergence,
%     u is the uniform discrete distribution over {1,...,D}, and
%     s = loss is the smoothing parameter ;
% it yields
%     KLs(y_v, x_v) = - H(s u + (1 - s) y_v)
%         - sum_d (s/D + (1 - s) y_{v,d}) log(s/D + (1 - s) x_{v,d}) ,
% where H is the entropy, that is H(s u + (1 - s) y_v)
%       = - sum_d (s/D + (1 - s) y_{v,d}) log(s/D + (1 - s) y_{v,d}) ;
% note that the choosen order of the arguments in the Kullback-Leibler
% does not favor the entropy of x (H(s u + (1 - s) y_v) is a constant),
% hence this loss is actually equivalent to cross-entropy.
%
% NOTA: by default, components are identified using uint16 identifiers; this
% can be easily changed in the mex source if more than 65535 components are
% expected (recompilation is necessary)
% 
% INPUTS: real numeric type is either single or double, not both;
%         indices start at 0, type uint32
%
% loss - 0 for linear, 1 for quadratic, 0 < loss < 1 for smoothed
%     Kullback-Leibler (see above)
% Y - observations, (real) D-by-V array, column-major format;
%     for Kullback-Leibler loss, the value at each vertex must lie on the
%     probability simplex 
% first_edge, adj_vertices - graph forward-star representation:
%     vertices are numeroted (start at 0) in the order they are given in Y
%         (careful to the internal memory representation of multidimensional
%          arrays, usually Octave and Matlab use column-major format)
%     edges are numeroted (start at 0) so that all vertices originating
%         from a same vertex are consecutive;
%     for each vertex, first_edge indicates the first edge starting from the
%         vertex (or, if there are none, starting from the next vertex);
%         (uint32) array of length V + 1, the first value is always zero and
%         the last value is always the total number of edges E;
%     for each edge, adj_vertices indicates its ending vertex, (uint32) array
%         of length E
% options - structure with any of the following fields [with default values]:
%     edge_weights [1.0], loss_weights [none], d1_coor_weights [none],
%     cp_dif_tol [1e-3], cp_it_max [10], pfdr_rho [1.0], pfdr_cond_min [1e-2],
%     pfdr_dif_rcd [0.0], pfdr_dif_tol [1e-3*cp_dif_tol], pfdr_it_max [1e4],
%     verbose [1e2], max_num_threads [none], balance_parallel_split [true]
% edge_weights - (real) array of length E, or scalar for homogeneous weights
% loss_weights - weights on vertices; (real) array of length V
% d1_coor_weights - for multidimensional data, weights the coordinates in the
%     l1 norms of finite differences; (real) array of length D;
%     all weights must be strictly positive, and it is advised to normalize the
%     weights so that the first value is unity
% cp_dif_tol - stopping criterion on iterate evolution; algorithm stops if
%     relative changes (in Euclidean norm) is less than dif_tol;
%     1e-3 is a typical value; a lower one can give better precision but with
%     longer computational time and more final components
% cp_it_max - maximum number of iterations (graph cut and subproblem);
%     10 cuts solve accurately most problems
% pfdr_rho - relaxation parameter, 0 < rho < 2;
%     1 is a conservative value; 1.5 often speeds up convergence
% pfdr_cond_min - stability of preconditioning; 0 < cond_min < 1;
%     corresponds roughly the minimum ratio to the maximum descent metric;
%     1e-2 is typical; a smaller value might enhance preconditioning
% pfdr_dif_rcd - reconditioning criterion on iterate evolution;
%     a reconditioning is performed if relative changes of the iterate drops
%     below dif_rcd; WARNING: reconditioning might temporarily draw minimizer
%     away from solution, and give bad subproblem solutions
% pfdr_dif_tol - stopping criterion on iterate evolution; algorithm stops if
%     relative changes (in l1 norm) is less than dif_tol;
%     1e-3*cp_dif_tol is a conservative value
% pfdr_it_max - maximum number of iterations;
%     1e4 iterations provides enough precision for most subproblems
% max_num_threads - if greater than zero, set the maximum number of threads
%     used for parallelization with OpenMP
% balance_parallel_split - if true, the parallel workload of the split step 
%     is balanced; WARNING: this might trade off speed against optimality
% verbose - if nonzero, display information on the progress, every 'verbose'
%     PFDR iterations
%
% OUTPUTS: indices start at 0
%
% Comp - assignement of each vertex to a component, (uint16) array of length V
% rX - values of each component of the minimizer, (real) array of length rV;
%     the actual minimizer can be reconstructed with X = rX(Comp + 1)
% cp_it - actual number of cut-pursuit iterations performed
% Obj  - the values of the objective functional along iterations;
%     array of length cp_it + 1
% Time - if requested, the elapsed time along iterations;
%     array of length cp_it + 1
% Dif  - if requested, the iterate evolution along iterations;
%     array of length cp_it
% 
% Parallel implementation with OpenMP API.
%
% H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing Nonsmooth
% Functionals with Graph Total Variation, International Conference on Machine
% Learning, PMLR, 2018, 80, 4244-4253
%
% H. Raguet, A Note on the Forward-Douglas--Rachford Splitting for Monotone 
% Inclusion and Convex Optimization Optimization Letters, 2018, 1-24
%
% Hugo Raguet 2017, 2018, 2019, 2020
