/*=============================================================================
 * Hugo Raguet 2019
 *===========================================================================*/
#include "cut_pursuit_d0.hpp"
#include <list>
#include <forward_list>

#define ZERO ((real_t) 0.0)
#define EDGE_WEIGHTS_(e) (edge_weights ? edge_weights[(e)] : homo_edge_weight)
/* special flag; no edge can have this identifier */
#define EMPTY_CELL (std::numeric_limits<size_t>::max())

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

TPL CP_D0:Merge_info(index_t re) : re(re), value(nullptr) {}

TPL CP_D0:~Merge_info() { free(value); }

TPL comp_t CP_D0::accept_merge(const Merge_info&)
{
    comp_t ru = reduced_edges_u(candidate->re);
    comp_t rv = reduced_edges_v(candidate->re);
    ru = merge_components(ru, rv);
    value_t* rXu = rX + D*ru;
    for (size_t d = 0; d < D; d++){ rXu[d] = candidate->value[d]; }
    return ru;
}

TPL comp_t CP_D0::compute_merge_chains()
{
    comp_t merge_count = 0;

    /* compute merge candidates in parallel */
    Merge_info* merge_candidates = (Merge_info*)
        malloc_check(sizeof(Merge_info*)*rE);
    #pragma omp parallel for NUM_THREADS(update_merge_complexity(), rE) \
        schedule(static)
    for (index_t re = 0; re < rE; re++){
        compt ru = reduced_edges_u(re), rv = reduced_edges_v(re);
        if (ru == rv){
            merge_candidates[re] = nullptr;
        }else{
            merge_candidates[re] = new Merge_info(re);
            update_merge_info(*merge_candidates[re]);
            // if (merge_candidates[re]->gain == -real_inf()){
                // delete merge_candidates[re]; merge_candidates[re] = nullptr;
            // }
        }
    }
    #if 0
    /* forget uninteresting merge candidates;
     * NOTA: they will NOT be reconsidered later in this merge step */
    index_t num_candidates = 0;
    for (index_t re = 0; re < rE; re++){
        if (merge_candidates[re]){
            merge_candidates[num_candidates++] = merge_candidates[re];
        }
    }
    if (!num_candidates){
        free(merge_candidates);
        return 0;
    }
    merge_candidates = (Merge_info*)
        realloc_check(sizeof(merge_info*)*num_candidates);
    /* count positive candidates and put them at the begining */
    index_t num_pos_candidates;
    {
        index_t mc = 0;
        while (mc < num_candidates && merge_candidates[mc]->gain > ZERO){
            mc++;
        }
        num_pos_candidates = mc++;
        while (mc < num_candidates){
            while (mc < num_candidates && merge_candidates[mc]->gain <= ZERO){
                mc++;
            }
            if (mc < num_candidates){
                Merge_info* tmp = merge_candidates[num_pos_candidates];
                merge_candidates[num_pos_candidates] = merge_candidates[mc]
                merge_candidates[mc] = tmp;
                num_pos_candidates++; mc++;
            }
        }
    }
    #endif
    /* count interesting candidates and put them at the begining */
    index_t num_candidates = 0;
    index_t num_discarded = 0;
    while (num_candidates < rE - num_discarded){
        if (merge_candidates[num_candidates]->gain > -real_inf()){
            num_candidates++;
        }else{
            num_discarded++;
            while (num_candidates < rE - num_discarded){
                if (merge_candidates[rE - num_discarded]->gain == -real_inf(){
                    num_discarded++;
                }
            }
        }
        if (num_candidates < rE - num_discarded){
            Merge_info* tmp = merge_candidates[num_candidates];
            merge_candidates[num_candidates] =
                merge_candidates[rE - num_discarded]
            merge_candidates[rE - num_discarded] = tmp;
        }
    }
    if (!num_candidates){
        free(merge_candidates);
        return 0;
    }

    /* sort candidates by decreasing gain order;
     * NOTA: they will NOT be reordered in this merge step */
    sort(merge_candidates, merge_candidates + num_candidates,
        [] (index_t mc1, index_t mc2) -> bool
        { return mc1->gain > mc2->gain; });

    /* linked list structure used for dynamically updating reduced graph
     * structure while merging:
     * - given a component, we need access to the list of merge candidates
     * whose corresponding reduced edge involves the considered component;
     * - to that purpose, we maintain for each component a linked list of such
     * merge candidates; we call "merge candidate cell" the data structure with
     * the merge candidate identifier and the access to the next cell in such a
     * linked list;
     * - each active merge candidates is thus referenced in two such cells: one
     * within both lists of starting and ending components of the corresponding
     * reduced edge;
     * - one can thus compact information mapping unequivocally each merge
     * candidate mc to merge candidate cells identifiers 2*mc and 2*mc + 1;
     * conversely, the merge candidate of a cell mcc is mcc/2
     * - the link list structure can thus be maintained with the following
     * tables:
     *  first_candidate_cell[ru] is the index of the first merge candidate
     *      cell of the list of adjacent candidates for component ru
     *  next_candidate_cell[mcc] is the index of the merge candidate cell
     *      that comes after mcc within the list containing it
     */
    size_t* first_candidate_cell = (size_t*) malloc_check(sizeof(size_t*)*rV);
    size_t* next_candidate_cell = (size_t*) malloc_check(sizeof(size_t*)*2*rE);
    for (comp_t rv = 0; rv < rV; rv++){
        first_candidate_cell[rv] = EMPTY_CELL;
    }
    for (size_t mcc = 0; mcc < ((size_t) 2)*rE; mcc++){
        next_candidate_cell[mcc] = EMPTY_CELL;
    }
    #define GET_CANDIDATE(mcc) merge_candidates[*mcc/2]
    #define FIRST_CELL(mcc, rv) mcc = &first_candidate_cell[rv];
    #define NEXT_CELL(mcc) mcc = &next_candidate_cell[*mcc]
    #define DELETE_CELL(mcc) *mcc = next_candidate_cell[*mcc]
    #define IS_EMPTY(mcc) (*mcc == EMPTY_CELL)
    
    /* construct the linked list structure;
     * last_candidate_cell[ru] is the index of the last merge candidate cell
     *      of the list of adjacent candidates for component ru;
     *      useful only for constructing the list in linear time */
    size_t* last_candidate_cell = (size_t*) malloc_check(sizeof(index_t*)*rV);
    for (comp_t rv = 0; rv < rV; rv++){ last_candidate_cell[rv] = EMPTY_CELL; }
    for (index_t mc = 0; mc < rE; mc++){
        comp_t ru = reduced_edges_u(merge_candidates[mc]->re);
        comp_t rv = reduced_edges_v(merge_candidates[mc]->re);
        #define INSERT_CELL(rv, mcc) \
            if (last_candidate_cell[rv] == EMPTY_CELL){ \
                first_candidate_cell[rv] = mcc; \
                last_candidate_cell[rv] = mcc; \
            }else{ \
                next_candidate_cell[last_candidate_cell[rv]] = mcc; \
                last_candidate_cell[rv] = mcc; \
            }
        size_t mcc_ru = ((size_t) 2)*mc, mcc_rv = ((size_t) 2)*mc + 1;
        INSERT_CELL(ru, mcc_ru); INSERT_CELL(rv, mcc_rv);
    }
    free(last_candidate_cell);

    /**  iterative merge following the above order  **/
    bool possible_updates = true;
    bool allow_negative_gains = false;
    while (possible_updates){
        possible_updates = false;

    for (index_t mc = 0; mc < rE; mc++){
        Merge_info*& candidate = merge_candidates[mc];
        if (!candidate){ continue; }
        if (candidate->gain == real_inf()){ update_merge_info(candidate); }
        if (candidate->gain == -real_inf()){ continue; }
        if (candidate->gain <= ZERO && !allow_negative_gains){ continue; }

        /**  accept the merge (delete it later)  **/
        comp_t ru = reduced_edges_u(candidate->re);
        comp_t rv = reduced_edges_v(candidate->re);
        comp_t ro = accept_merge(candidate); // merge ru and rv
        if (ro != ru){ rv = ru; ru = ro; } // makes sure ru is the root

        /**  update reduced graph structure and adjacent merge candidates  **/

        /* first pass on each list: cleanup list from already deleted
         * candidates, delete current merging candidate, update end vertices of
         * adjacent candiadates of rv */
        size_t *mcc_ru, *mcc_rv;
        FIRST_CELL(mcc_ru, ru);
        while (!IS_EMPTY(mcc_ru)){
            Merge_info*& candidate_ru = GET_CANDIDATE(mcc_ru);
            if (!candidate_ru){ DELETE_CELL(mcc_ru); continue; }
            index_t re_eu = candidate_ru->re;
            comp_t end_ru = reduced_edges_u(re_eu) == ru ?
                reduced_edges_v(re_eu) : reduced_edges_u(re_eu);
            if (end_ru == rv){ DELETE_CELL(mcc_ru); continue; }
            /* flag for later updates */
            candidate_ru->gain = real_inf(); possible_updates = true;
            NEXT_CELL(mcc_ru)
        }
        FIRST_CELL(mcc_rv, rv);
        while (!IS_EMPTY(mcc_rv)){
            Merge_info*& candidate_rv = GET_CANDIDATE(mcc_rv);
            if (!candidate_rv){ DELETE_CELL(mcc_rv); continue; }
            index_t re_ev = candidate_rv->re;
            if (reduced_edges_u(re_ev) == rv){ 
                reduced_edges_u(re_ev) = ru;
                end_rv = reduced_edges_v(re_ev);
            }else{
                reduced_edges_v(re_ev) = ru;
                end_rv = reduced_edges_u(re_ev);
            }
            if (end_rv == ru){ DELETE_CELL(mcc_rv); continue; }
            /* flag for later updates */
            candidate_rv->gain = real_inf(); possible_updates = true;
            NEXT_CELL(mcc_rv);
        }

        /* now search for candidates adjacent to ru and rv with same en vertex;
         * NOTA: for the later, bilinear time cost cannot be avoided; ordering
         * lists by end vertex identifiers would need constant reordering */
        for (FIRST_CELL(mcc_ru, ru); !IS_EMPTY(mcc_ru); NEXT_CELL(mcc_ru)){
            Merge_info*& candidate_ru = GET_CANDIDATE(mcc_ru);
            index_t re_eu = candidate_ru->re;
            comp_t end_ru = reduced_edges_u(re_eu) == ru ?
                reduced_edges_v(re_eu) : reduced_edges_u(re_eu);
            for (FIRST_CELL(mcc_rv, rv); !IS_EMPTY(mcc_rv); NEXT_CELL(mcc_rv)){
                Merge_info*& candidate_rv = GET_CANDIDATE(mcc_rv);
                index_t re_ev = candidate_rv->re;
                comp_t end_rv = reduced_edges_u(re_ev) == rv ?
                    reduced_edges_v(re_ev) : reduced_edges_u(re_ev);
                if (end_ru == end_rv){
                    reduced_edge_weights[re_ru] += reduced_edge_weights[re_rv];
                    reduced_edge_weights[re_rv] = ZERO; // sum must be constant
                    delete candidate_rv; candidate_rv = nullptr;
                    DELETE_CELL(mcc_rv); break;
                }
            }
        }

        /* at that point, mcc_ru is the last (empty) cell of the rv list;
         * concatenate adjacent candidate list of rv after the one of ru  */
        *mcc_ru = first_candidate_cell[rv];

        /* delete current merging candidate */ 
        delete candidate; candidate = nullptr;
    } // endfor mc

        if (!allow_negative_gains){
            possible_updates = true;
            allow_negative_gains = true;
        }

    } // endwhile possible updates

    free(first_candidate_cell);
    free(next_candidate_cell);

    for (index_t re = 0; re < rE; re++){ delete merge_candidates[re]; }
    free(merge_candidates);

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
