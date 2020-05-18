/*=============================================================================
 * Maximum flow augmenting path algorithm, allowing to compute maxflows in
 * parallel along constant connected components delimited by arcs with negative
 * flows, using the same graph structure; this enables the use of the same main
 * graph structure, without having to recompute a sub graph structure for each
 * connected compoent.
 *
 * The graph structure used is a two-ways forward-star representation, choosen
 * for convenience for integration with algorithms making use of simple
 * (unoriented) forward-star representation.
 *
 * Modified from MAXFLOW by Vladimir Kolmogorov and Yuri Boykov under GPLv3 
 *=============================================================================
   MAXFLOW is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   MAXFLOW is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with MAXFLOW.  If not, see <http://www.gnu.org/licenses/>.       
*============================================================================*/
#include "block.hpp" /* Kolmogorov's memory management for chain lists */

template <typename index_t, typename flow_t>
class Maxflow
{
private:
    enum Terminal : char {SOURCE, SINK}; // requires C++11 to ensure 1 byte

public:
    /* made public for user pre-allocation */
    struct Flow_node
	{
		index_t parent; // node's parent arc
        index_t next_in_queue; // next node in waiting queue
        /* timestamp showing when the distance to the terminal was last
         * computed; used to be a long int, overflow check in definition */
        index_t timestamp; 
        index_t term_dist; // distance to the terminal
        Terminal cut_side;
        /* if term_cap > 0, residual capacity of the arc SOURCE->node
         * otherwise, opposite of residual capacity of the arc node->SINK */
		flow_t term_cap;
	};

    /* arguments constitutes the main graph structure, according to a
     * two-ways forward-star graph structure, see corresponding members;
     * for computing maximum flows in parallel along components, create new
     * instances in parallel with same graph structure and same arrays
     * arc_res_cap and nodes, which must be already allocated */
    Maxflow(index_t node_num, index_t arc_num, const index_t* first_arc,
        const index_t* adj_nodes, const index_t* reverse_arc,
        flow_t* arc_res_cap = nullptr, Flow_node* nodes = nullptr);

    /* frees arc_res_cap and nodes if not provided at construction */
    ~Maxflow();

    /* compute the maximum flow; parameters enable treating only one connected
     * component delimited by arcs with negative capacities */
    void maxflow(index_t component_size, const index_t* component_nodes);

    /* overload for processing the whole graph */
    void maxflow(){ maxflow(number_of_nodes, nullptr); }

    flow_t& terminal_capacity(index_t node); // get/set terminal capacity

    /* set capacities of both arcs of an edge */
    void set_edge_capacities(index_t e, flow_t cap, flow_t cap_rev);

    bool is_sink(index_t node); // tells if node is in the sink tree 

private:
    const index_t number_of_nodes, number_of_arcs;
    /**  two-ways forward-star graph representation  **/
    /* this build on a simple (unoriented) forward-star graph representation:
     * - edges (unoriented arcs) are numeroted so that all nodes originating
     * from a same node are consecutive;
     * - for each node, an array of size number of nodes  + 1 indicates the
     * first edge starting from the node (or, if there are none, starting from
     * the next node); the first value is always zero and the last value is
     * always the total number of edges
     * - for each edge, a second array of size number of edges (half the number
     * of arcs) indicates the adjacent node */
    /* now, there is need to scan all arcs involving a given vertex in linear
     * time; we keep the reverse arcs also in a forward-star representation as
     * above; moreover, it is useful to keep the array of adjacent nodes
     * for the reverse arcs contiguous with the array of adjacent nodes for
     * the edges; likewise, the indices of the first reverse arcs are put
     * contiguously to the indices of the first edges; finally
     * - 'first_arc' is thus of lengthe 2V + 1, the first value is always
     * zero, the (V+1)-th value is always the numer of edges, the last
     * value is always the number of arcs
     * - 'adj_nodes' is thus of length the number of arcs. */
    const index_t *first_arc, *adj_nodes; 
    const index_t* first_rev_arc; // = this will be first_arc + node_num
    const index_t* reverse_arc; // for each arc, the index of its reverse arc

    flow_t* arc_res_cap; // residual capacities
    const bool owns_arc_res_cap;

    Flow_node* nodes; // nodes info
    const bool owns_nodes;

    struct Node_cell
    {
        index_t node;
        Node_cell* next;
    };

    index_t time_counter; // monotonically increasing global counter

    /* memory block for orphan lists */
    DBlock<Node_cell>* node_cell_block;

    /* list of "active" nodes */
    index_t queue_first[2], queue_last[2]; 

    /* list of orphan nodes */
    Node_cell *orphan_first, *orphan_last;

    /* functions for processing queuing list */
    void push_queue(index_t node);
    index_t pop_queue();

    /* functions for processing orphans list */
    void set_orphan_front(index_t node);
    void set_orphan_rear(index_t node);

    /* processing augmenting path */
    index_t grow_tree(index_t node);
    void find_bottleneck(Terminal side, index_t middle_arc,
        flow_t& bottleneck);
    void push_flow(const Terminal side, index_t middle_arc, flow_t flow);
    void process_orphan(index_t node);
};

#define TPL template <typename index_t, typename flow_t>
#define MXFL Maxflow<index_t, flow_t>

TPL inline flow_t& MXFL::terminal_capacity(index_t node)
{ return nodes[node].term_cap; }

TPL inline void MXFL::set_edge_capacities(index_t e, flow_t cap,
    flow_t cap_rev)
{
    arc_res_cap[e] = cap;
    arc_res_cap[reverse_arc[e]] = cap_rev;
}

TPL inline bool MXFL::is_sink(index_t node)
{
    /* without parent a node is considered in the source side;
     * this is arbitrary but coherent */
    return (nodes[node].parent && nodes[node].cut_side == SINK);
}

#undef TPL
#undef MXFL
