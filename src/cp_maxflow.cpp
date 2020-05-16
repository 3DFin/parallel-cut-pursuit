#include "../include/cut_pursuit.hpp"

#define TPL template <typename real_t, typename index_t, typename comp_t, \
    typename value_t>
#define CP Cp<real_t, index_t, comp_t, value_t>

/* constants of the correct type */
#define ZERO ((real_t) 0.0)
/* special constants for nodes' parents;
 * use maximum number of arcs, no arc can have these identifiers */
#define TO_TERMINAL (std::numeric_limits<index_t>::max())
#define ORPHAN (std::numeric_limits<index_t>::max() - 1)
#define NO_PARENT (std::numeric_limits<index_t>::max() - 2)
/* special constant for arcs */
#define NO_PATH (std::numeric_limits<index_t>::max())
/* special constant for nodes */
#define NOT_IN_QUEUE (std::numeric_limits<index_t>::max())
#define NO_NODE (std::numeric_limits<index_t>::max())
/* infinite distance to the terminal */
#define INF_DIST (std::numeric_limits<index_t>::max())
#define NODE_LIST_BLOCK_SIZE 128 // size for memory blocks, see block.hpp

using namespace std;

TPL CP::Maxflow::Maxflow(CP* cp, Flow_node* nodes)
    : first_edge(cp->first_edge), adj_vertices(cp->adj_vertices),
      first_rev_arc(cp->first_rev_arc), reverse_arc(cp->reverse_arc),
      arc_res_cap(cp->arc_res_cap), nodes(nodes), node_cell_block(nullptr){}

/*
	Functions for processing queuing list ("active" nodes).
	node->next_in_queue is the next node in the list
	(or to itself, if it is the last node in the list).
	If node->next_in_queue is NOT_IN_QUEUE iff i is not in the list.

	There are two queues. Active nodes are added
	to the end of the second queue and read from
	the front of the first queue. If the first queue
	is empty, it is replaced by the second queue
	(and the second queue becomes empty).
*/

TPL inline void CP::Maxflow::push_queue(index_t i)
{
	if (nodes[i].next_in_queue == NOT_IN_QUEUE)
	{
		if (queue_last[1] == NO_NODE){
            nodes[queue_last[1]].next_in_queue = i;
        }else{
            queue_first[1] = i;
        }
		queue_last[1] = i;
		nodes[i].next_in_queue = i;
	}
}

/*
	Returns the next active node.
	If it is connected to the sink, it stays in the list,
	otherwise it is removed from the list
*/
TPL inline index_t CP::Maxflow::pop_queue()
{
	index_t i;

	while (true)
	{
		if ((i = queue_first[0]) == NO_NODE){
			queue_first[0] = i = queue_first[1];
			queue_last[0] = queue_last[1];
			queue_first[1] = NO_NODE;
			queue_last[1]  = NO_NODE;
			if (i == NO_NODE){ return i; }
		}

		/* remove it from the active list */
		if (nodes[i].next_in_queue == i){
            queue_first[0] = queue_last[0] = NO_NODE;
        }else{
            queue_first[0] = nodes[i].next_in_queue;
        }
		nodes[i].next_in_queue = NOT_IN_QUEUE;

		/* a node in the list is "active" iff it has a parent */
		if (nodes[i].parent != NO_PARENT){ return i; }
	}
}

TPL inline void CP::Maxflow::set_orphan_front(index_t i)
{
	nodes[i].parent = ORPHAN;
	Node_cell* nc = node_cell_block->New();
	nc->node = i;
	nc->next = orphan_first;
	orphan_first = nc;
}

TPL inline void CP::Maxflow::set_orphan_rear(index_t i)
{
	nodes[i].parent = ORPHAN;
	Node_cell* nc = node_cell_block->New();
	nc->node = i;
	nc->next = nullptr;
	if (orphan_last){ orphan_last->next = nc; }
    else{ orphan_first = nc; }
	orphan_last = nc;
}

TPL void CP::Maxflow::maxflow_init(index_t comp_size,
    const index_t* const comp_nodes)
{
    if (!node_cell_block){
		node_cell_block = new DBlock<Node_cell>(NODE_LIST_BLOCK_SIZE);
	}

    queue_first[0] = queue_last[0] = NO_NODE;
	queue_first[1] = queue_last[1] = NO_NODE;
	orphan_first = nullptr;

	time_counter = 0;

    const index_t iimax = comp_size ? comp_size : node_num;
	for (index_t ii = 0; ii < iimax; ii++){
        i = comp_nodes ? comp_nodes[ii] : ii;
		nodes[i].next_in_queue = NOT_IN_QUEUE;
		nodes[i].timestamp = time_counter;
		if (nodes[i].term_cap > ZERO){ /* i is connected to the source */
			nodes[i].cut_side = SOURCE;
			nodes[i].parent = TO_TERMINAL;
			push_queue(i);
			nodes[i].term_dist = 1;
		}else if (nodes[i].term_cap < ZERO){ /* i is connected to the sink */
			nodes[i].cut_side = SINK;
			nodes[i].parent = TO_TERMINAL;
			push_queue(i);
			nodes[i].term_dist = 1;
		}else{
			nodes[i].parent = NO_PARENT;
		}
	}
}

TPL void CP::Maxflow::find_bottleneck(const Terminal side, index_t middle_arc,
    real_t& bottleneck)
{
    index_t a = side == SOURCE ? reverse_arc[middle_arc] : middle_arc;
    index_t i = adj_vertices[a];
    while ((a = nodes[i].parent) != TO_TERMINAL){
        real_t res_cap = arc_res_cap[side == SOURCE ? reverse_arc[a] : a];
		if (bottleneck > res_cap){ bottleneck = res_cap; }
        i = adj_vertices[a];
	}
    real_t term_cap = side == SOURCE ? nodes[i].term_cap : -nodes[i].term_cap;
	if (bottleneck > term_cap){ bottleneck = term_cap; }
}

TPL void CP::Maxflow::push_flow(const Terminal side, index_t middle_arc, real_t flow)
{
    real_t signed_flow = side == SOURCE ? flow : -flow;
    index_t a = side == SOURCE ? reverse_arc[middle_arc] : middle_arc;
    index_t i = adj_vertices[a];
    while ((a = nodes[i].parent) != TO_TERMINAL){
		arc_res_cap[a] += signed_flow;
		arc_res_cap[reverse_arc[a]] -= signed_flow;
		if (side == SOURCE && arc_res_cap[reverse_arc[a]] == ZERO ||
            side == SINK && arc_res_cap[a] == ZERO){
			set_orphan_front(i); // add i to the beginning of the adoption list
		}
        i = adj_vertices[a];
	}
	nodes[i].term_cap -= signed_flow;
	if (nodes[i].term_cap == ZERO){
		set_orphan_front(i); // add i to the beginning of the adoption list
	}
}

TPL void CP::Maxflow::process_orphan(index_t i)
{
    index_t a0_min = NO_PARENT;
    index_t d_min = INF_DIST;

    const Terminal side = nodes[i].cut_side;

	/* trying to find a new parent */
    for (index_t a0 = first_edge[i]; a0 < first_rev_arc[i + 1]; a0++){
        if (a0 == first_edge[i + 1]){ // switch to reverse edges
            a0 = first_rev_arc[i];
            if (a0 == first_rev_arc[i + 1]){ break; }
        }

        if (side == SOURCE && arc_res_cap[reverse_arc[a0]] <= ZERO ||
            side == SINK && arc_res_cap[a0] <= ZERO){ continue; }

        index_t j = adj_vertices[a0];
        index_t a = nodes[j].parent;

        if (nodes[j].cut_side != side || a == NO_PARENT){ continue; }

        /* checking the origin of j */
        index_t d = 0;
        while (true){
            if (nodes[j].timestamp == time_counter){
                d += nodes[j].term_dist;
                break;
            }
            a = nodes[j].parent;
            d++;
            if (a == TO_TERMINAL){
                nodes[j].timestamp = time_counter;
                nodes[j].term_dist = 1;
                break;
            }
            if (a == ORPHAN){ d = INF_DIST; break; }
            j = adj_vertices[a];
        }

        if (d < INF_DIST){ /* j originates from the good terminal - done */
            if (d < d_min){
                a0_min = a0;
                d_min = d;
            }
            /* set marks along the path */
            j = adj_vertices[a0];
            while (nodes[j].timestamp != time_counter){
                nodes[j].timestamp = time_counter;
                nodes[j].term_dist = d--;
                j = adj_vertices[nodes[j].parent];
            }
        }
    }

	if ((nodes[i].parent = a0_min) != NO_PARENT){
		nodes[i].timestamp = time_counter;
		nodes[i].term_dist = d_min + 1;
	}else{
		/* process neighbors */
        for (index_t a0 = first_edge[i]; a0 < first_rev_arc[i + 1]; a0++){
            if (a0 == first_edge[i + 1]){ // switch to reverse edges
                a0 = first_rev_arc[i];
                if (a0 == first_rev_arc[i + 1]){ break; }
            }

            /* negative value indicate active edge in the cut-pursuit sense,
             * and thus a0 links to another component */
            if (arc_res_cap[a0] < ZERO){ continue; }

            index_t j = adj_vertices[a0];
            index_t a = nodes[j].parent;

            if (nodes[j].cut_side = side && a != NO_PARENT){
				if (side == SOURCE && arc_res_cap[reverse_arc[a0]] > ZERO ||
                    side == SINK && arc_res_cap[a0] > ZERO){ push_queue(j); }
				if (a != TO_TERMINAL && a != ORPHAN && adj_vertices[a] == i){
                    set_orphan_rear(j); // add j to the end of adoption list
				}
			}
		}
	}
}

TPL index_t CP::Maxflow::grow_tree(index_t i)
{
    const Terminal side = nodes[i].cut_side;

    for (index_t a = first_edge[i]; a < first_rev_arc[i + 1]; a++){
        if (a == first_edge[i + 1]){ // switch to reverse edges
            a = first_rev_arc[i];
            if (a == first_rev_arc[i + 1]){ return NO_PATH; }
        }

        if (side == SOURCE && arc_res_cap[a] <= ZERO ||
            side == SINK && arc_res_cap[reverse_arc[a]] <= ZERO){ continue; }

        index_t j = adj_vertices[a];
        if (nodes[j].parent == NO_PARENT){
            nodes[j].cut_side = side;
            nodes[j].parent = reverse_arc[a];
            nodes[j].timestamp = nodes[i].timestamp;
            nodes[j].term_dist = nodes[i].term_dist + 1;
            push_queue(j);
        }else if (nodes[j].cut_side != side){
            return side == SOURCE ? a : reverse_arc[a];
        }else if (nodes[j].timestamp <= nodes[i].timestamp &&
                  nodes[j].term_dist > nodes[i].term_dist){
	        /* trying to make the distance from j the terminal shorter */
            nodes[j].parent = reverse_arc[a];
            nodes[j].timestamp = nodes[i].timestamp;
            nodes[j].term_dist = nodes[i].term_dist + 1;
        }
    }
}

TPL void CP::Maxflow::maxflow(index_t comp_size, const index_t* comp_nodes)
{
    maxflow_init(comp_size, comp_nodes);

    index_t i = NO_NODE;

	/* main loop */
	while (true)
	{
		if (i != NO_NODE){
			nodes[i].next_in_queue = NOT_IN_QUEUE; /* remove active flag */
			if (nodes[i].parent == NO_PARENT){ i = NO_NODE; }
		}
		if (i == NO_NODE){
            i = pop_queue();
            if (i == NO_NODE){ break; }
        }

        index_t a = grow_tree(i);

		if (++time_counter <= 0){
        /* changed type from long to index_t */
        /* can't we prove this won't overflow? */
            std::cerr << "Maxflow: timestamp overflow." << std::endl;
            exit(EXIT_FAILURE);
        }

		if (a == NO_PATH){
            i = NO_NODE;
        }else{ /* augmenting path found */
            nodes[i].next_in_queue = i; /* set active flag */

            /* finding bottleneck capacity */
            real_t bottleneck = arc_res_cap[a];
            find_bottleneck(SOURCE, a, bottleneck);
            find_bottleneck(SINK, a, bottleneck);

            /* augmenting */
            arc_res_cap[reverse_arc[a]] += bottleneck;
            arc_res_cap[a] -= bottleneck;
            push_flow(SOURCE, a, bottleneck);
            push_flow(SINK, a, bottleneck);

			/* adoption */
            Node_cell* nc;
			while ((nc = orphan_first)){
				Node_cell* nc_next = nc->next;
				nc->next = nullptr;

				while ((nc = orphan_first)){
					orphan_first = nc->next;
					index_t j = nc->node;
					node_cell_block->Delete(nc);
					if (!orphan_first){ orphan_last = nullptr; }
                    process_orphan(i);
				}

				orphan_first = nc_next;
			}
        }
	}

    delete node_cell_block; 
    node_cell_block = nullptr; 
}

/* instantiate for compilation */
template class Cp<float, uint32_t, uint16_t>::Maxflow;
template class Cp<double, uint32_t, uint16_t>::Maxflow;
template class Cp<float, uint32_t, uint32_t>::Maxflow;
template class Cp<double, uint32_t, uint32_t>::Maxflow;
