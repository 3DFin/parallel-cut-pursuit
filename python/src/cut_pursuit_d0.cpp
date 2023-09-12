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
    split_damp_ratio = 1.0;
    split_values_init_num = 3;
    split_values_iter_num = 3;
}

TPL real_t CP_D0::compute_graph_d0() const
{
    real_t weighted_contour_length = ZERO;
    #pragma omp parallel for schedule(static) NUM_THREADS(rE) \
        reduction(+:weighted_contour_length)
    for (index_t re = 0; re < rE; re++){
        weighted_contour_length += reduced_edge_weights[re];
    }
    return weighted_contour_length;
}

TPL real_t CP_D0::compute_f() const
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

TPL real_t CP_D0::compute_objective() const
{ return compute_f() + compute_graph_d0(); } // f(x) + ||x||_d0

TPL real_t CP_D0::vert_split_cost(const Split_info& split_info, index_t v,
        comp_t k) const
{ return fv(v, split_info.sX + D*k); }

/* compute binary cost of choosing alternatives lu and lv at edge e */
TPL real_t CP_D0::edge_split_cost(const Split_info& split_info, index_t e,
    comp_t lu, comp_t lv) const
{ return lu == lv ? ZERO : EDGE_WEIGHTS_(e); }

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
    // #pragma omp parallel NUM_THREADS(update_merge_complexity(), rE)
    // { cannot populate lists in parallel
    Merge_info merge_info(D);
    // #pragma omp for schedule(static)
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
    // }

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
            update_merge_info(candidate);
            if (candidate.gain > -real_inf()){
                accept_merge(candidate);
                merge_count++;
            }
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
