/*=============================================================================
 * tools for manipulating graphs
 *
 * Hugo Raguet 2020
 *===========================================================================*/
#pragma once

/* vertex_t is supposed to be an unsigned integer type able to hold the total
 * number of _vertices_;
 * edge_t is supposed to be an unsigned integer type able to hold the total
 * number of _arcs_, that is twice the number of edges */
template <typename vertex_t, typename edge_t>
void forward_star_to_reverse(vertex_t V, edge_t E, const edge_t* first_edge,
    const vertex_t* adj_vertices, edge_t* first_edge_reverse,
    vertex_t* adj_vertices_reverse, edge_t* reverse_arc);
/* Build a redundant graph representation from a forward-star representation
 *
 * Forward-star representation:
 * - edges are numeroted so that all vertices originating from a same vertex
 * are consecutive;
 * - for each vertex, 'first_edge' indicates the first edge starting
 * from the vertex (or, if there are none, starting from the next vertex);
 * array of length V + 1, the first value is always zero and the last
 * value is always the total number of edges E;
 * - for each edge, 'adj_vertices' indicates its ending vertex;
 *
 * Redundant two-ways forward-star representation:
 * the reverse arcs are also kept in a forward-star representation and it is
 * possible to get the reverse arc of any given arc in constant time. 
 * Moreover, it is convenient to keep both forward-star representation
 * available, and contiguously. Thus:
 * - 'first_edge_reverse' is thus of length 2V + 1, the first value is always
 * zero, the (V+1)-th value is always the total numer of edges E, the last
 * value is always the total number of arcs 2E;
 * - 'adj_vertices_reverse' is thus of length 2E.
 * - 'reverse_arc' is thus of length 2E, indicating for each arc, the index of
 * its reverse arc within this concatenation of both forward-star
 * representation
 *
 * Memory should be already allocated; the conversion can be done in-place, by
 * giving same pointer for first_edge and first_edge_reverse, and same pointer
 * for adj_vertices and adj_vertices_reverse. */
