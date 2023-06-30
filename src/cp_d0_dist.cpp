/*=============================================================================
 * Hugo Raguet 2018, 2022, 2023
 *===========================================================================*/
#include <random>
#include "cp_d0_dist.hpp"

#define ZERO ((real_t) 0.0)
#define VERT_WEIGHTS_(v) (vert_weights ? vert_weights[(v)] : (real_t) 1.0)

#define TPL template <typename real_t, typename index_t, typename comp_t>
#define CP_D0_DIST Cp_d0_dist<real_t, index_t, comp_t>

using namespace std;

TPL CP_D0_DIST::Cp_d0_dist(index_t V, index_t E, const index_t* first_edge,
    const index_t* adj_vertices, const real_t* Y, size_t D)
    : Cp_d0<real_t, index_t, comp_t>(V, E, first_edge, adj_vertices, D), Y(Y)
{
    vert_weights = coor_weights = nullptr;
    comp_weights = nullptr; 

    loss = quadratic_loss();
    fYY = ZERO;
    fXY = real_inf();

    min_comp_weight = ZERO;
}

TPL CP_D0_DIST::~Cp_d0_dist(){ free(comp_weights); }

TPL real_t CP_D0_DIST::distance(const real_t* Yv, const real_t* Xv) const
{
    real_t dist = 0.0;
    if (loss == quadratic_loss()){
        if (coor_weights){
            for (size_t d = 0; d < D; d++){
                dist += coor_weights[d]*(Yv[d] - Xv[d])*(Yv[d] - Xv[d]);
            }
        }else{
            for (size_t d = 0; d < D; d++){
                dist += (Yv[d] - Xv[d])*(Yv[d] - Xv[d]);
            }
        }
    }else{ // smoothed Kullback-Leibler; just compute cross-entropy here
        const real_t c = ((real_t) 1.0 - loss);
        const real_t q = loss/D;
        if (coor_weights){
            for (size_t d = 0; d < D; d++){
                dist -= coor_weights[d]*(q + c*Yv[d])*log(q + c*Xv[d]);
            }
        }else{
            for (size_t d = 0; d < D; d++){
                dist -= (q + c*Yv[d])*log(q + c*Xv[d]);
            }
        }
    }
    return dist;
}

TPL void CP_D0_DIST::set_loss(real_t loss, const real_t* Y,
    const real_t* vert_weights, const real_t* coor_weights)
{
    if (loss < ZERO || loss > 1.0){
        cerr << "Cut-pursuit d0 distance: loss parameter should be between"
            " 0 and 1 (" << loss << " given)." << endl;
        exit(EXIT_FAILURE);
    }
    if (loss == ZERO){ loss = eps; } // avoid singularities
    this->loss = loss;
    if (Y){ this->Y = Y; }
    this->vert_weights = vert_weights;
    this->coor_weights = coor_weights; 
    /* recompute the constant dist(Y, Y) if necessary */
    real_t fYY_par = ZERO; // auxiliary variable for parallel region
    if (loss != quadratic_loss()){
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V) \
            reduction(+:fYY_par)
        for (index_t v = 0; v < V; v++){
            const real_t* Yv = Y + D*v;
            fYY_par += VERT_WEIGHTS_(v)*distance(Yv, Yv);
        }
    }
    fYY = fYY_par;
}

TPL void CP_D0_DIST::set_split_param(index_t max_split_size, comp_t K,
    int split_iter_num, real_t split_damp_ratio, int split_values_init_num,
    int split_values_iter_num)
{
    Cp<real_t, index_t, comp_t>::set_split_param(max_split_size, K,
        split_iter_num, split_damp_ratio, split_values_init_num,
        split_values_iter_num);
}

TPL void CP_D0_DIST::set_min_comp_weight(real_t min_comp_weight)
{
    if (min_comp_weight < ZERO){
        cerr << "Cut-pursuit d0 distance: min component weight parameter "
            "should be positive (" << min_comp_weight << " given)." << endl;
        exit(EXIT_FAILURE);
    }
    this->min_comp_weight = min_comp_weight;
}

TPL real_t CP_D0_DIST::fv(index_t v, const real_t* Xv) const
{ return VERT_WEIGHTS_(v)*distance(Y + D*v, Xv); }

TPL real_t CP_D0_DIST::compute_f() const
{
    return fXY == real_inf() ?
        Cp_d0<real_t, index_t, comp_t>::compute_f() - fYY : fXY - fYY;
}

TPL void CP_D0_DIST::solve_reduced_problem()
{
    free(comp_weights);
    comp_weights = (real_t*) malloc_check(sizeof(real_t)*rV);

    #pragma omp parallel for schedule(static) NUM_THREADS(2*D*V, rV)
    for (comp_t rv = 0; rv < rV; rv++){
        real_t* rXv = rX + D*rv;
        comp_weights[rv] = ZERO;
        for (size_t d = 0; d < D; d++){ rXv[d] = ZERO; }
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            index_t v = comp_list[i];
            comp_weights[rv] += VERT_WEIGHTS_(v);
            const real_t* Yv = Y + D*v;
            for (size_t d = 0; d < D; d++){ rXv[d] += VERT_WEIGHTS_(v)*Yv[d]; }
        }
        if (comp_weights[rv]){
            for (size_t d = 0; d < D; d++){ rXv[d] /= comp_weights[rv]; }
        } /* maybe one should raise an exception for zero weight component */
    }

    /* fXY can be updated now to avoid computing it twice later */
    if (monitor_evolution()){
        fXY = Cp_d0<real_t, index_t, comp_t>::compute_f();
    }
}

TPL void CP_D0_DIST::set_split_value(Split_info& split_info, comp_t k,
    index_t v) const
{
    const real_t* Yv = Y + D*v;
    real_t* sXk = split_info.sX + D*k;
    for (size_t d = 0; d < D; d++){ sXk[d] = Yv[d]; }
}

TPL void CP_D0_DIST::update_split_info(Split_info& split_info) const
{
    comp_t rv = split_info.rv;
    comp_t& K = split_info.K; // shadow member K
    real_t* sX = split_info.sX;
    real_t* total_weights = (real_t*) malloc_check(sizeof(real_t)*K);
    for (comp_t k = 0; k < K; k++){
        total_weights[k] = ZERO;
        real_t* sXk = sX + D*k;
        for (size_t d = 0; d < D; d++){ sXk[d] = ZERO; }
    }
    for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
        index_t v = comp_list[i];
        comp_t k = label_assign[v];
        total_weights[k] += VERT_WEIGHTS_(v);
        const real_t* Yv = Y + D*v;
        real_t* sXk = sX + D*k;
        for (size_t d = 0; d < D; d++){ sXk[d] += VERT_WEIGHTS_(v)*Yv[d]; }
    }
    for (comp_t k = 0; k < K; k++){
        real_t* sXk = sX + D*k;
        if (total_weights[k]){
            for (size_t d = 0; d < D; d++){ sXk[d] /= total_weights[k]; }
        }else{ // no vertex assigned to k, discard this alternative
            k--; K--;
        }
    }
    free(total_weights);
}

TPL void CP_D0_DIST::update_merge_info(Merge_info& merge_info)
{
    comp_t ru = merge_info.ru;
    comp_t rv = merge_info.rv;
    real_t edge_weight = reduced_edge_weights[merge_info.re];

    real_t* rXu = rX + D*ru;
    real_t* rXv = rX + D*rv;
    real_t wru = comp_weights[ru]/(comp_weights[ru] + comp_weights[rv]);
    real_t wrv = comp_weights[rv]/(comp_weights[ru] + comp_weights[rv]);

    real_t* value = merge_info.value;
    for (size_t d = 0; d < D; d++){ value[d] = wru*rXu[d] + wrv*rXv[d]; }

    real_t gain;

    if (loss == quadratic_loss()){
        gain = edge_weight - comp_weights[ru]*wrv*distance(rXu, rXv);
    }else{
        /* in the following some computations might be saved by factoring
         * multiplications and logarithms, at the cost of readability */
        gain = edge_weight
            + comp_weights[ru]*(distance(rXu, rXu) - distance(rXu, value))
            + comp_weights[rv]*(distance(rXv, rXv) - distance(rXv, value));
    }

    if (gain > ZERO || comp_weights[ru] < min_comp_weight
                    || comp_weights[rv] < min_comp_weight){
        merge_info.gain = gain;
    }else{
        merge_info.gain = -real_inf();
    }
}

TPL size_t CP_D0_DIST::update_merge_complexity()
{ return rE*2*D; /* each update is only linear in D */ }

TPL comp_t CP_D0_DIST::accept_merge(const Merge_info& candidate)
{
    comp_t ru = Cp_d0<real_t, index_t, comp_t>::accept_merge(candidate);
    comp_t rv = ru == candidate.ru ? candidate.rv : candidate.ru;
    comp_weights[ru] += comp_weights[rv];
    return ru;
}

TPL index_t CP_D0_DIST::merge()
{
    index_t deactivation = Cp_d0<real_t, index_t, comp_t>::merge();
    free(comp_weights); comp_weights = nullptr;
    return deactivation;
}

TPL real_t CP_D0_DIST::compute_evolution() const
{
    real_t dif = ZERO;
    #pragma omp parallel for schedule(dynamic) reduction(+:dif) \
        NUM_THREADS(D*(V - saturated_vert), rV)
    for (comp_t rv = 0; rv < rV; rv++){
        if (is_saturated[rv]){ continue; }
        const real_t* rXv = rX + D*rv;
        real_t distXX = loss == quadratic_loss() ? ZERO : distance(rXv, rXv);
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            index_t v = comp_list[i];
            const real_t* lrXv = last_rX + D*last_comp_assign[v];
            dif += VERT_WEIGHTS_(v)*(distance(rXv, lrXv) - distXX);
        }
    }
    real_t amp = compute_f();
    return amp > eps ? dif/amp : dif/eps;
}

/**  instantiate for compilation  **/
#if defined _OPENMP && _OPENMP < 200805
/* use of unsigned counter in parallel loops requires OpenMP 3.0;
 * although published in 2008, MSVC still does not support it as of 2020 */
template class Cp_d0_dist<float, int32_t, int16_t>;
template class Cp_d0_dist<double, int32_t, int16_t>;
template class Cp_d0_dist<float, int32_t, int32_t>;
template class Cp_d0_dist<double, int32_t, int32_t>;
#else
template class Cp_d0_dist<float, uint32_t, uint16_t>;
template class Cp_d0_dist<double, uint32_t, uint16_t>;
template class Cp_d0_dist<float, uint32_t, uint32_t>;
template class Cp_d0_dist<double, uint32_t, uint32_t>;
#endif
