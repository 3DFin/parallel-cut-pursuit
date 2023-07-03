/*=============================================================================
 * Hugo Raguet 2018, 2022, 2023
 *===========================================================================*/
#include <random>
#include "cp_d0_dist.hpp"

#define ZERO ((real_t) 0.0)
#define ONE ((real_t) 1.0)
#define VERT_WEIGHTS_(v) (vert_weights ? vert_weights[(v)] : ONE)
#define COOR_WEIGHTS_(d) (coor_weights ? coor_weights[(d)] : ONE)

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
    real_t dist = ZERO;
    size_t Q = loss; // number of coordinates for quadratic part
    if (Q != 0){ /* quadratic part */
        for (size_t d = 0; d < Q; d++){
            dist += COOR_WEIGHTS_(d)*(Yv[d] - Xv[d])*(Yv[d] - Xv[d]);
        }
    }
    if (Q != D){ /* smoothed Kullback-Leibler;
                    just compute cross-entropy here */
        real_t distKL = ZERO;
        const real_t s = loss < ONE ? loss : eps;
        const real_t c = ONE - s;  
        const real_t u = s/(D - Q);
        for (size_t d = Q; d < D; d++){
            distKL -= (u + c*Yv[d])*log(u + c*Xv[d]);
        }
        dist += COOR_WEIGHTS_(Q)*distKL;
    }
    return dist;
}

TPL void CP_D0_DIST::set_loss(real_t loss, const real_t* Y,
    const real_t* vert_weights, const real_t* coor_weights)
{
    if (loss < ZERO || (loss > ONE && ((size_t) loss) != loss) || loss > D){
        cerr << "Cut-pursuit d0 distance: loss parameter should be positive,"
            "either in (0,1) or an integer that do not exceed the dimension "
            "(" << loss << " given)." << endl;
        exit(EXIT_FAILURE);
    }
    if (loss == ZERO){ loss = eps; } // avoid singularities
    this->loss = loss;
    if (Y){ this->Y = Y; }
    this->vert_weights = vert_weights;
    if (ZERO < loss && loss < ONE && coor_weights){
        cerr << "Cut-pursuit d0 distance: no sense in weighting coordinates of"
            " the probability space in Kullback-Leibler divergence." << endl;
        exit(EXIT_FAILURE);
    }
    this->coor_weights = coor_weights; 
    if (loss == quadratic_loss()){ fYY = ZERO; return; }
    /* recompute the constant dist(Y, Y) for Kullback-Leibler */
    const size_t Q = loss; // number of coordinates for quadratic part
    const real_t s = loss < ONE ? loss : eps;
    const real_t c = ONE - s;  
    const real_t u = s/(D - Q);
    real_t fYY_par = ZERO; // auxiliary variable for parallel region
    #pragma omp parallel for schedule(static) NUM_THREADS(V*(D - loss), V) \
        reduction(+:fYY_par)
    for (index_t v = 0; v < V; v++){
        const real_t* Yv = Y + D*v;
        real_t H_Yv = ZERO;
        for (size_t d = Q; d < D; d++){
            H_Yv -= (u + c*Yv[d])*log(u + c*Yv[d]);
        }
        fYY_par += VERT_WEIGHTS_(v)*H_Yv;
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
        if (comp_weights[rv] <= ZERO){
            cerr << "Cut-pursuit d0 distance: nonpositive total component "
                "weight; something went wrong." << endl;
            exit(EXIT_FAILURE);
        }
        for (size_t d = 0; d < D; d++){ rXv[d] /= comp_weights[rv]; }
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

    real_t gain = edge_weight;
    size_t Q = loss; // number of coordinates for quadratic part

    if (Q != 0){
        /* quadratic gain */
        real_t gainQ = ZERO;
        for (size_t d = 0; d < Q; d++){
            gainQ -= COOR_WEIGHTS_(d)*(rXu[d] - rXv[d])*(rXu[d] - rXv[d]);
        }
        gain += comp_weights[ru]*wrv*gainQ;
    }

    if (gain > ZERO || comp_weights[ru] < min_comp_weight
                    || comp_weights[rv] < min_comp_weight){
        real_t* value = merge_info.value;
        for (size_t d = 0; d < D; d++){ value[d] = wru*rXu[d] + wrv*rXv[d]; }

        if (Q != D){
            /* smoothed Kullback-Leibler gain */
            real_t gainKLu = ZERO, gainKLv = ZERO;
            const real_t s = loss < ONE ? loss : eps;
            const real_t c = ONE - s;  
            const real_t u = s/(D - Q);
            for (size_t d = Q; d < D; d++){
                real_t u_value_d = u + c*value[d];
                real_t u_rXu_d = u + c*rXu[d];
                real_t u_rXv_d = u + c*rXv[d];
                gainKLu -= (u_rXu_d)*log(u_rXu_d/u_value_d);
                gainKLv -= (u_rXv_d)*log(u_rXv_d/u_value_d);
            }
            gain += COOR_WEIGHTS_(Q)*
                (comp_weights[ru]*gainKLu + comp_weights[rv]*gainKLv);
        }
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
        real_t distXX = ZERO;
        if (loss != quadratic_loss()){
            const size_t Q = loss; // number of coordinates for quadratic part
            const real_t s = loss < ONE ? loss : eps;
            const real_t c = ONE - s;  
            const real_t u = s/(D - Q);
            for (size_t d = Q; d < D; d++){
                distXX -= (u + c*rXv[d])*log(u + c*rXv[d]);
            }
            distXX *= COOR_WEIGHTS_(Q);
        }
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
