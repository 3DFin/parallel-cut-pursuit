function [Comp, rX, List, Gtv, Obj, Time, Dif] = cp_prox_tv_mex(Y, ...
    first_edge, adj_vertices, options)
%
%       [Comp, rX, List, Gtv, Obj, Time, Dif] = cp_prox_tv_mex(Y, first_edge,
%   adj_vertices, options)
%
% Compute the proximity operator of the d1 (total variation) penalization:
%
% minimize functional over a graph G = (V, E)
%
%        F(x) = 1/2 ||y - x||^2 + ||x||_d1
%
% where y in R^V is given, and for x in R^V,
%      ||x||_d1 = sum_{uv in E} w_d1_uv |x_u - x_v|,
%
% using cut-pursuit approach with preconditioned forward-Douglas-Rachford 
% splitting algorithm.
%
% NOTA: by default, components are identified using uint16 identifiers; this
% can be easily changed in the mex source if more than 65535 components are
% expected (recompilation is necessary)
% 
% INPUTS: real numeric type is either single or double, not both;
%         indices start at 0, type uint32
%
% Y - observations, (real) array of length N (direct matricial case), or
%                          array of length V (left-premult. by A^t), or
%                          empty matrix (for all zeros)
% first_edge, adj_vertices - forward-star graph representation:
%     vertices are numeroted (start at 0) in the order they are given in Y or A
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
%     edge_weights [1.0], cp_dif_tol [1e-4], cp_it_max [10], pfdr_rho [1.0],
%     pfdr_cond_min [1e-2], pfdr_dif_rcd [0.0], pfdr_dif_tol [1e-2*cp_dif_tol],
%     pfdr_it_max [1e4], verbose [1e3], max_num_threads [none],
%     balance_parallel_split [true]
% edge_weights - (real) array of length E, or scalar for homogeneous weights
% cp_dif_tol - stopping criterion on iterate evolution; algorithm stops if
%     relative changes (in Euclidean norm) is less than dif_tol;
%     1e-4 is a typical value; a lower one can give better precision but with
%     longer computational time and more final components
% cp_it_max - maximum number of iterations (graph cut and subproblem);
%     10 cuts solve accurately most problems
% pfdr_rho - relaxation parameter, 0 < rho < 2;
%     1 is a conservative value; 1.5 often speeds up convergence
% pfdr_cond_min - stability of preconditioning; 0 < cond_min < 1;
%     corresponds roughly the minimum ratio to the maximum descent metric;
%     1e-2 is typical, a smaller value might enhance preconditioning
% pfdr_dif_rcd - reconditioning criterion on iterate evolution;
%     a reconditioning is performed if relative changes of the iterate drops
%     below dif_rcd; WARNING: reconditioning might temporarily draw minimizer
%     away from the solution, and give bad subproblem solutions
% pfdr_dif_tol - stopping criterion on iterate evolution; algorithm stops if
%     relative changes (in Euclidean norm) is less than dif_tol;
%     1e-2*cp_dif_tol is a conservative value
% pfdr_it_max - maximum number of iterations;
%     1e4 iterations provides enough precision for most subproblems
% verbose - if nonzero, display information on the progress, every 'verbose'
%     PFDR iterations
% max_num_threads - if greater than zero, set the maximum number of threads
%     used for parallelization with OpenMP
% balance_parallel_split - if true, the parallel workload of the split step 
%     is balanced; WARNING: this might trade off speed against optimality
%
% OUTPUTS: indices start at 0
%
% Comp - assignement of each vertex to a component, (uint16) array of length V
% rX - values of eachcomponents of the minimizer, (real) array of length rV;
%     the actual minimizer can be reconstructed with X = rX(Comp + 1);
% List - if requested, list of vertices constituting each component; cell array
%     of length rV, containing (uint32) arrays of indices
% Gtv - subgradients of the total variation penalization at solution; (real)
%     array of length E; if e is the edge (u, v), the subgradient of the
%     total variation penalization at vertices (u, v) is (-Gd1(e), Gd1(e))
% Obj - the values of the objective functional along iterations, up to the
%     constant 1/2||Y||^2; array of length number of iterations performed + 1;
% Time - if requested, the elapsed time along iterations;
%     array of length number of iterations performed + 1
% Dif  - if requested, the iterate evolution along iterations;
%     array of length number of iterations performed
% 
% Parallel implementation with OpenMP API.
%
% L. Landrieu and G. Obozinski, Cut Pursuit: Fast Algorithms to Learn
% Piecewise Constant Functions on General Weighted Graphs, SIIMS, 10, 4,
% 1724–1766, 2017.
%
% Hugo Raguet 2022
