/*=============================================================================
 * tools for manipulating graphs
 *
 * Hugo Raguet 2020
 *===========================================================================*/
#include <cstdint>
#include "../include/graph_tools.hpp"

template <typename vertex_t, typename edge_t>
void forward_star_to_reverse(vertex_t V, edge_t E, const edge_t* first_edge,
    const vertex_t* adj_vertices, edge_t* first_edge_reverse,
    vertex_t* adj_vertices_reverse, edge_t* reverse_arc)
{
    /**  get pointers to the start of reverse information  **/
    vertex_t* first_edge_rev = first_edge_reverse + V;

    /**  construct reverse forward-star representation  **/
    /* count the number of reverse edges for each vertex */
    for (vertex_t v = 0; v <= V; v++){ first_edge_rev[v] = 0; }
    for (edge_t e = 0; e < E; e++){ first_edge_rev[adj_vertices[e] + 1]++; };
    /* cumulative sum, giving actual first reverse edge id for each vertex */
    for (vertex_t v = 2; v <= V; v++){
        first_edge_rev[v] += first_edge_rev[v - 1];
    }
    /* set reverse adjacent vertices, using previous sum as starting indices */
    for (vertex_t v = 0; v < V; v++){
        for (edge_t e = first_edge[v]; e < first_edge[v + 1]; e++){
            edge_t e_rev = first_edge_rev[adj_vertices[e]]++;
            e_rev += E; // actual reverse arc number
            adj_vertices_reverse[e_rev] = v;
            reverse_arc[e] = e_rev;
            reverse_arc[e_rev] = e;
        }
    }
    /* first reverse edges have been shifted in the process
     * shift back and add E to get the actual arc numbers */
    for (vertex_t v = V; v > 0; v--){
        first_edge_rev[v] = E + first_edge_rev[v - 1];
    }
    first_edge_rev[0] = E;

    /**  copy initial forward-star representation if necessary  **/
    if (first_edge_reverse != first_edge){
        for (vertex_t v = 0; v < V; v++){
            first_edge_reverse[v] = first_edge[v];
        }
    }
    if (adj_vertices_reverse != adj_vertices){
        for (edge_t e = 0; e < E; e++){
            adj_vertices_reverse[e] = adj_vertices[e];
        }
    }
}

/* instantiate for compilation */
template void forward_star_to_reverse<uint16_t, uint16_t>(uint16_t V,
    uint16_t E, const uint16_t* first_edge, const uint16_t* adj_vertices,
    uint16_t* first_edge_reverse, uint16_t* adj_vertices_reverse,
    uint16_t* reverse_arc);
template void forward_star_to_reverse<uint32_t, uint32_t>(uint32_t V,
    uint32_t E, const uint32_t* first_edge, const uint32_t* adj_vertices,
    uint32_t* first_edge_reverse, uint32_t* adj_vertices_reverse,
    uint32_t* reverse_arc);
template void forward_star_to_reverse<uint64_t, uint64_t>(uint64_t V,
    uint64_t E, const uint64_t* first_edge, const uint64_t* adj_vertices,
    uint64_t* first_edge_reverse, uint64_t* adj_vertices_reverse,
    uint64_t* reverse_arc);
