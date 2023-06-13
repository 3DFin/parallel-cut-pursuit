/*=============================================================================
 * Hugo Raguet 2019
 *===========================================================================*/
#include "cut_pursuit_d0.hpp"

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
    : Cp<real_t, index_t, comp_t>(V, E, first_edge, adj_vertices, D),
      accepted_merge(&reserved_merge_info)
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

TPL CP_D0::Merge_info::Merge_info(size_t D)
{ value = (value_t*) malloc_check(sizeof(value_t)*D); }

TPL CP_D0::Merge_info::~Merge_info()
{ free(value); }

TPL void CP_D0::delete_merge_candidate(index_t re)
{
    delete merge_info_list[re];
    merge_info_list[re] = accepted_merge;
}

TPL void CP_D0::select_best_merge_candidate(index_t re, real_t* best_gain,
    index_t* best_edge)
{
    if (merge_info_list[re] && merge_info_list[re]->gain > *best_gain){
            *best_gain = merge_info_list[re]->gain;
            *best_edge = re;
    }
}

TPL void CP_D0::accept_merge_candidate(index_t re, comp_t& ru, comp_t& rv)
{
    merge_components(ru, rv); // ru now the root of the merge chain
    value_t* rXu = rX + D*ru;
    for (size_t d = 0; d < D; d++){ rXu[d] = merge_info_list[re]->value[d]; }
}

TPL comp_t CP_D0::compute_merge_chains()
{
    comp_t merge_count = 0;
   
    merge_info_list = (Merge_info**) malloc_check(sizeof(Merge_info*)*rE);
    for (index_t re = 0; re < rE; re++){ merge_info_list[re] = nullptr; }

    real_t* best_par_gains =
        (real_t*) malloc_check(sizeof(real_t)*omp_get_num_procs());
    index_t* best_par_edges = 
        (index_t*) malloc_check(sizeof(index_t)*omp_get_num_procs());

    comp_t last_merge_root = MERGE_INIT;

    while (true){ /* merge iteratively as long as gain is positive */
 
        /**  update merge information in parallel  **/
        int num_par_thrds = last_merge_root == MERGE_INIT ?
            compute_num_threads(update_merge_complexity()) :
            /* expected fraction of merge candidates to update is the total
             * number of edges divided by the expected number of edges linking
             * to the last merged component; in turn, this is estimated as
             * twice the number of edges divided by the number of components */
            compute_num_threads(update_merge_complexity()/rV*2);

        for (int thrd_num = 0; thrd_num < num_par_thrds; thrd_num++){
            best_par_gains[thrd_num] = -real_inf();
        }

        /* differences between threads is small: using static schedule */
        #pragma omp parallel for schedule(static) num_threads(num_par_thrds)
        for (index_t re = 0; re < rE; re++){
            if (merge_info_list[re] == accepted_merge){ continue; }
            comp_t ru = reduced_edges[TWO*re];
            comp_t rv = reduced_edges[TWO*re + 1];

            if (last_merge_root != MERGE_INIT){
                /* the roots of their respective chains might have changed */
                ru = get_merge_chain_root(ru);
                rv = get_merge_chain_root(rv);
                /* check if none of them is concerned by the last merge */
                if (last_merge_root != ru && last_merge_root != rv){
                    select_best_merge_candidate(re,
                        best_par_gains + omp_get_thread_num(),
                        best_par_edges + omp_get_thread_num());
                    continue;
                }
            }

            if (ru == rv){ /* already merged */
                delete_merge_candidate(re);
            }else{ /* update information */
                update_merge_candidate(re, ru, rv);
                select_best_merge_candidate(re,
                    best_par_gains + omp_get_thread_num(),
                    best_par_edges + omp_get_thread_num());
            }
        } // end for candidates in parallel

        /**  select best candidate among parallel threads  **/
        real_t best_gain = best_par_gains[0];
        index_t best_edge = best_par_edges[0];
        for (int thrd_num = 1; thrd_num < num_par_thrds; thrd_num++){
            if (best_gain < best_par_gains[thrd_num]){
                best_gain = best_par_gains[thrd_num];
                best_edge = best_par_edges[thrd_num];
            }
        }

        /**  merge best candidate if best gain is positive  **/
        /* we allow for negative gains, as long as its not negative infinity */
        if (best_gain > -real_inf()){
            comp_t ru = get_merge_chain_root(reduced_edges[2*best_edge]);
            comp_t rv = get_merge_chain_root(reduced_edges[2*best_edge + 1]);
            accept_merge_candidate(best_edge, ru, rv); // ru now the root
            delete_merge_candidate(best_edge);
            merge_count++;
            last_merge_root = ru;
        }else{
            break;
        }
   
    } // end merge loop

    free(best_par_gains);
    free(best_par_edges);
    free(merge_info_list); // all merge info must have been deleted

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
