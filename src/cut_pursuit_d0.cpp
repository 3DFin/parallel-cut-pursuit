/*=============================================================================
 * Hugo Raguet 2019
 *===========================================================================*/
#include "cut_pursuit_d0.hpp"
#include <list>
#include <forward_list>

#define ZERO ((real_t) 0.0)
#define ONE ((real_t) 1.0)
#define TWO ((size_t) 2) // avoid overflows
#define EDGE_WEIGHTS_(e) (edge_weights ? edge_weights[(e)] : homo_edge_weight)
/* special flag (no component can have this identifier) */
#define MERGE_INIT (std::numeric_limits<comp_t>::max())

#define TPL template <typename real_t, typename index_t, typename comp_t, \
    typename value_t>
#define CP_D0 Cp_d0<real_t, index_t, comp_t, value_t>

using namespace std;

TPL CP_D0::Cp_d0(index_t V, index_t E, const index_t* first_edge,
    const index_t* adj_vertices, size_t D)
    : Cp<real_t, index_t, comp_t>(V, E, first_edge, adj_vertices, D)
{
    /* ensure handling of infinite values (negation, comparisons) is safe */
    static_assert(numeric_limits<real_t>::is_iec559,
        "Cut-pursuit d0: real_t must satisfy IEEE 754.");

    K = 2;
    split_iter_num = 2;
    split_damp_ratio = ONE;
}

TPL void CP_D0::set_split_param(comp_t K, int split_iter_num,
    real_t split_damp_ratio)
{
    if (split_iter_num < 1){
        cerr << "Cut-pursuit d0: there must be at least one iteration in the "
            "split (" << split_iter_num << " specified)." << endl;
        exit(EXIT_FAILURE);
    }

    if (K < 2){
        cerr << "Cut-pursuit d0: there must be at least two alternative values"
            "in the split (" << K << " specified)." << endl;
        exit(EXIT_FAILURE);
    }

    if (split_damp_ratio <= 0 || split_damp_ratio > ONE){
        cerr << "Cut-pursuit d0: split damping ratio must be between zero "
            "excluded and one included (" << split_damp_ratio << " specified)."
            << endl;
        exit(EXIT_FAILURE);
    }

    this->K = K;
    this->split_iter_num = split_iter_num;
    this->split_damp_ratio = split_damp_ratio;
}

TPL real_t CP_D0::compute_graph_d0()
{
    real_t weighted_contour_length = ZERO;
    #pragma omp parallel for schedule(static) NUM_THREADS(rE) \
        reduction(+:weighted_contour_length)
    for (index_t re = 0; re < rE; re++){
        weighted_contour_length += reduced_edge_weights[re];
    }
    return weighted_contour_length;
}

TPL real_t CP_D0::compute_f()
{
    real_t f = ZERO;
    #pragma omp parallel for schedule(dynamic) NUM_THREADS(D*V, rV) \
        reduction(+:f)
    for (comp_t rv = 0; rv < rV; rv++){
        real_t* rXv = rX + D*rv;
        for (index_t v = first_vertex[rv]; v < first_vertex[rv + 1]; v++){
            f += fv(comp_list[v], rXv);
        }
    }
    return f;
}

TPL real_t CP_D0::compute_objective()
{ return compute_f() + compute_graph_d0(); } // f(x) + ||x||_d0

TPL uintmax_t CP_D0::split_complexity()
{
    uintmax_t complexity = maxflow_complexity(); // graph cut
    complexity += D*V; // account for distance difference and final labeling
    complexity += E; // edges capacities
    if (K > 2){ complexity *= K; } // K alternative labels
    complexity *= split_iter_num; // repeated
    complexity += split_values_complexity(); // init and update
    return complexity*(V - saturated_vert)/V; // account saturation linearly
}

TPL void CP_D0::split_component(comp_t rv, Maxflow<index_t, real_t>* maxflow)
{
    value_t* altX = (value_t*) malloc_check(sizeof(value_t)*D*K);

    index_t comp_size = first_vertex[rv + 1] - first_vertex[rv];
    const index_t* comp_list_rv = comp_list + first_vertex[rv];

    real_t damping = split_damp_ratio;
    for (int split_it = 0; split_it < split_iter_num; split_it++){
        damping += (ONE - split_damp_ratio)/split_iter_num;

        if (split_it == 0){ init_split_values(rv, altX); }
        else{ update_split_values(rv, altX); }

        bool no_reassignment = true;

        if (K == 2){ /* one graph cut is enough */
            for (index_t i = 0; i < comp_size; i++){
                index_t v = comp_list_rv[i];
                /* unary cost for choosing the second alternative */
                maxflow->terminal_capacity(i) = fv(v, altX + D) - fv(v, altX);
            }

            /* set d0 edge capacities within each component */
            index_t e_in_comp = 0;
            for (index_t i = 0; i < comp_size; i++){
                index_t v = comp_list_rv[i];
                for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
                    if (is_bind(e)){
                        real_t cap = damping*EDGE_WEIGHTS_(e);
                        maxflow->set_edge_capacities(e_in_comp++, cap, cap);
                    }
                }
            }

            /* find min cut and set assignment accordingly */
            maxflow->maxflow();

            for (index_t i = 0; i < comp_size; i++){
                index_t v = comp_list_rv[i];
                if (maxflow->is_sink(i) != label_assign[v]){
                    label_assign[v] = maxflow->is_sink(i);
                    no_reassignment = false;
                }
            }

        }else{ /* iterate over all K alternative values */
            for (comp_t k = 0; k < K; k++){
    
            /* check if alternative k has still vertices assigned to it */
            if (!is_split_value(altX[D*k])){ continue; }

            /* set the source/sink capacities */
            bool all_assigned_k = true;
            for (index_t i = 0; i < comp_size; i++){
                index_t v = comp_list_rv[i];
                comp_t l = label_assign[v];
                /* unary cost for changing current value to k-th value */
                if (l == k){
                    maxflow->terminal_capacity(i) = ZERO;
                }else{
                    maxflow->terminal_capacity(i) = fv(v, altX + D*k) -
                        fv(v, altX + D*l);
                    all_assigned_k = false;
                }
            }
            if (all_assigned_k){ continue; }

            /* set d0 edge capacities within each component */
            index_t e_in_comp = 0;
            for (index_t i = 0; i < comp_size; i++){
                index_t u = comp_list_rv[i];
                comp_t lu = label_assign[u];
                for (index_t e = first_edge[u]; e < first_edge[u + 1]; e++){
                    if (!is_bind(e)){ continue; }
                    index_t v = adj_vertices[e];
                    comp_t lv = label_assign[v];
                    /* horizontal and source/sink capacities are modified 
                     * according to Kolmogorov & Zabih (2004); in their
                     * notations, functional E(u,v) is decomposed as
                     *
                     * E(0,0) | E(0,1)    A | B
                     * --------------- = -------
                     * E(1,0) | E(1,1)    C | D
                     *                         0 | 0      0 | D-C    0 |B+C-A-D
                     *                 = A + --------- + -------- + -----------
                     *                       C-A | C-A    0 | D-C    0 |   0
                     *
                     *            constant +      unary terms     + binary term
                     */
                    /* A = E(0,0) is the cost of the current assignment */
                    real_t A = lu == lv ? ZERO : damping*EDGE_WEIGHTS_(e);
                    /* B = E(0,1) is the cost of changing lv to k */
                    real_t B = lu == k ? ZERO : damping*EDGE_WEIGHTS_(e);
                    /* C = E(1,0) is the cost of changing lu to k */
                    real_t C = lv == k ? ZERO : damping*EDGE_WEIGHTS_(e);
                    /* D = E(1,1) = 0 is for changing both lu, lv to k */
                    /* set weights in accordance with orientation u -> v */
                    maxflow->terminal_capacity(i) += C - A;
                    maxflow->terminal_capacity(index_in_comp[v]) -= C;
                    maxflow->set_edge_capacities(e_in_comp++, B + C - A, ZERO);
                }
            }

            /* find min cut and update assignment accordingly */
            maxflow->maxflow();

            for (index_t i = 0; i < comp_size; i++){
                index_t v = comp_list_rv[i];
                if (maxflow->is_sink(i) && label_assign[v] != k){
                    label_assign[v] = k;
                    no_reassignment = false;
                }
            }

            } // end for k
        } // end if K == 2

        if (no_reassignment){ break; }

    } // end for split_it

    free(altX);
}

TPL CP_D0::Merge_info::Merge_info(size_t D) : D(D)
{ value = (value_t*) malloc_check(sizeof(value_t)*D); }

TPL CP_D0::Merge_info::Merge_info(const Merge_info& merge_info) :
    D(merge_info.D), re(merge_info.re), ru(merge_info.ru), rv(merge_info.rv),
    gain(merge_info.gain)
{
    value = (value_t*) malloc_check(sizeof(value_t)*D);
    for (size_t d = 0; d < D; d++){ value[d] = merge_info.value[d]; }
}

TPL CP_D0::Merge_info::~Merge_info()
{ free(value); }

TPL comp_t CP_D0::accept_merge(const Merge_info& candidate)
{
    comp_t ru = merge_components(candidate.ru, candidate.rv);
    value_t* rXu = rX + D*ru;
    for (size_t d = 0; d < D; d++){ rXu[d] = candidate.value[d]; }
    return ru;
}

TPL comp_t CP_D0::compute_merge_chains()
{
    comp_t merge_count = 0;

    /* compute merge candidate lists in parallel */
    list<Merge_info> candidates;
    forward_list<Merge_info> neg_candidates;
    Merge_info merge_info(D);
    // #pragma omp parallel for schedule(static) \
        // NUM_THREADS(update_merge_complexity(), rE)
    for (index_t re = 0; re < rE; re++){
        merge_info.re = re;
        merge_info.ru = reduced_edges[TWO*re];
        merge_info.rv = reduced_edges[TWO*re + 1];
        update_merge_info(merge_info);
        if (merge_info.gain > ZERO){
            candidates.push_front(merge_info);
        }else if (merge_info.gain > -real_inf()){
            neg_candidates.push_front(merge_info);
        }
    }

    /**  positive gains merges: update all gains after each merge  **/
    comp_t last_merge_root = MERGE_INIT;
    while (!candidates.empty()){ 
        typename list<Merge_info>::iterator best_candidate;
        real_t best_gain = -real_inf();

        for (typename list<Merge_info>::iterator
             candidate = candidates.begin(); candidate != candidates.end(); ){
            comp_t ru = get_merge_chain_root(candidate->ru);
            comp_t rv = get_merge_chain_root(candidate->rv);
            if (ru == rv){ /* already merged */
                candidate = candidates.erase(candidate);
                continue;
            }
            candidate->ru = ru;
            candidate->rv = rv;
            if (last_merge_root == ru || last_merge_root == rv){
                update_merge_info(*candidate);
            }
            if (candidate->gain > best_gain){
                best_candidate = candidate;
                best_gain = best_candidate->gain;
            }
            candidate++;
        }

        if (best_gain > ZERO){
            last_merge_root = accept_merge(*best_candidate);
            candidates.erase(best_candidate);
            merge_count++;
        }else{
            break;
        }

    } // end positive gain merge loop

    /* negative gains will be allowed as long as they are not infinity */
    for (typename list<Merge_info>::iterator
         candidate = candidates.begin(); candidate != candidates.end(); ){
        if (candidate->gain == -real_inf()){
            candidate = candidates.erase(candidate);
        }else{
            candidate++;
        }
    }

    /* update all negative gains and transfer to the candidates list */
    while (!neg_candidates.empty()){
        Merge_info& candidate = neg_candidates.front();
        comp_t ru = get_merge_chain_root(candidate.ru);
        comp_t rv = get_merge_chain_root(candidate.rv);
        if (ru != rv){ /* not already merged */
            candidate.ru = ru;
            candidate.rv = rv;
            update_merge_info(candidate);
            if (candidate.gain > -real_inf()){
                candidates.push_front(candidate);
            }
        }
        neg_candidates.pop_front();
    }

    /**  negative gain merges: sort and merge in that order, no update  **/
    candidates.sort(
        [] (const Merge_info& mi1, const Merge_info& mi2) -> bool
        { return mi1.gain > mi2.gain; } ); // decreasing order
    while (!candidates.empty()){ 
        Merge_info& candidate = candidates.front();
        comp_t ru = get_merge_chain_root(candidate.ru);
        comp_t rv = get_merge_chain_root(candidate.rv);
        if (ru != rv){ /* not already merged */
            candidate.ru = ru;
            candidate.rv = rv;
            accept_merge(candidate);
            merge_count++;
        }
        candidates.pop_front();
    }

    return merge_count;
}

/**  instantiate for compilation  **/
#if defined _OPENMP && _OPENMP < 200805
/* use of unsigned counter in parallel loops requires OpenMP 3.0;
 * although published in 2008, MSVC still does not support it as of 2020 */
    template class Cp_d0<float, int32_t, int16_t>;
    template class Cp_d0<double, int32_t, int16_t>;
    template class Cp_d0<float, int32_t, int32_t>;
    template class Cp_d0<double, int32_t, int32_t>;
#else
    template class Cp_d0<float, uint32_t, uint16_t>;
    template class Cp_d0<double, uint32_t, uint16_t>;
    template class Cp_d0<float, uint32_t, uint32_t>;
    template class Cp_d0<double, uint32_t, uint32_t>;
#endif
