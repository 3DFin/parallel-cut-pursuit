function [first_edge_reverse, adj_vertices_reverse, reverse_arc] = ...
    forward_star_to_reverse_mex(first_edge, adj_vertices)
%
%        [first_edge_reverse, adj_vertices_reverse, reverse_arc] = ...
%   forward_star_to_reverse_mex(first_edge, adj_vertices)
%
% Build a redundant graph representation from a forward-star representation
% 
% INPUTS: indices are C-style (start at 0) of type uint16 or uint32, not both
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
%         array of length V + 1 (uint32), the first value is always zero and
%         the last value is always the total number of edges E;
%     for each edge, adj_vertices indicates its ending vertex, array of 
%         length E (uint32)
%
%     NOTA: if performing multiple calls to this function on the same graph
%     structure, it might be worth precomputing the "two-ways forward-star
%     graph representation" used by the cut-pursuit algorithm;
%     this consists in specifying the reverse edges in the same fashion,
%     contiguously to the above arrays, and, for each resulting oriented edge,
%     the index of the oriented edge in the reverse direction; in that case,
%     first_edge is of length 2V + 1, its last value is always twice the
%     total number of edges 2E, adj_vertices is of length 2E, and the reverse
%     arc table is an array of length 2E given in the dedicated option below;
%     see also the function forward_star_to_reverse_mex
%
% OUTPUTS: indices are C-style (start at 0)
%
% Hugo Raguet 2020
