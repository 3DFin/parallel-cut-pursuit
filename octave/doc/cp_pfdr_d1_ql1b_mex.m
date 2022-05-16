function [Comp, rX, cp_it, Obj, Time, Dif] = cp_pfdr_d1_ql1b_mex(Y, A, ...
    first_edge, adj_vertices, options)
%
%       [Comp, rX, cp_it, Obj, Time, Dif] = cp_pfdr_d1_ql1b_mex(Y | AtY,
%    A | AtA, first_edge, adj_vertices, options)
%
% Cut-pursuit algorithm with d1 (total variation) penalization, with a 
% quadratic functional, l1 penalization and box constraints:
%
% minimize functional over a graph G = (V, E)
%
%        F(x) = 1/2 ||y - A x||^2 + ||x||_d1 + ||yl1 - x||_l1 + i_[m,M](x)
%
% where y in R^N, x in R^V, A in R^{N-by-|V|}
%      ||x||_d1 = sum_{uv in E} w_d1_uv |x_u - x_v|,
%      ||x||_l1 = sum_{v  in V} w_l1_v |x_v|,
% and the convex indicator
%      i_[m,M] = infinity if it exists v in V such that x_v < m_v or x_v > M_v
%              = 0 otherwise;
%
% using cut-pursuit approach with preconditioned forward-Douglas-Rachford 
% splitting algorithm.
%
% It is easy to introduce a SDP metric weighting the squared l2-norm
% between y and A x. Indeed, if M is the matrix of such a SDP metric,
%   ||y - A x||_M^2 = ||Dy - D A x||^2, with D = M^(1/2).
% Thus, it is sufficient to call the method with Y <- Dy, and A <- D A.
% Moreover, when A is the identity and M is diagonal (weighted square l2 
% distance between x and y), one should call on the precomposed version 
% (see below) with Y <- DDy = My and A <- D2 = M.
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
% A - matrix, (real) N-by-V array (direct matricial case), or
%                    V-by-V array (premultiplied to the left by A^t), or
%                    V-by-1 array (_square_ diagonal of A^t A = A^2), or
%                    nonzero scalar (for identity matrix), or
%                    zero scalar (for no quadratic part);
%     for an arbitrary scalar matrix, use identity and scale observations and
%     penalizations accordingly
%     if N = V in a direct matricial case, set the last option 'Gram_if_square'
%     to false
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
%     edge_weights [1.0], Yl1 [none], l1_weights [0.0], low_bnd [-Inf],
%     upp_bnd [Inf], cp_dif_tol [1e-4], cp_it_max [10], pfdr_rho [1.0],
%     pfdr_cond_min [1e-2], pfdr_dif_rcd [0.0], pfdr_dif_tol [1e-2*cp_dif_tol],
%     pfdr_it_max [1e4], verbose [1e3], max_num_threads [none],
%     balance_parallel_split [true], Gram_if_square [true]
% edge_weights - (real) array of length E, or scalar for homogeneous weights
% Yl1 - offset for l1 penalty, (real) array of length V
% l1_weights - (real) array of length V, or scalar for homogeneous weights
% low_bnd - (real) array of length V or scalar;
%     set to negative infinity for no lower bound
% upp_bnd - (real) array of length V or scalar;
%     set to positive infinity for no upper bound
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
% Gram_if_square - if A is square, set this to false for direct matricial case
%
% OUTPUTS: indices start at 0
%
% Comp - assignement of each vertex to a component, (uint16) array of length V
% rX - values of eachcomponents of the minimizer, (real) array of length rV;
%     the actual minimizer can be reconstructed with X = rX(Comp + 1);
% cp_it - actual number of cut-pursuit iterations performed
% Obj - the values of the objective functional along iterations;
%     array of length cp_it + 1;
%     WARNING: in the precomputed A^t A version (including diagonal or identity
%     case), a constant 1/2||Y||^2 in the quadratic part is omited
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
