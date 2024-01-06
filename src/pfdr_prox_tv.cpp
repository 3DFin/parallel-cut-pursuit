/*=============================================================================
 * Hugo Raguet 2023
 *===========================================================================*/
#include <cmath>
#include "pfdr_prox_tv.hpp"
#include "omp_num_threads.hpp"

/* constants of the correct type */
#define ZERO ((real_t) 0.0)
#define ONE ((real_t) 1.0)
#define HALF ((real_t) 0.5)

#define Ga_(v, vd)  (gashape == SCALAR ? ga : \
                      gashape == MONODIM ? Ga[(v)] : Ga[(vd)])
#define M_(v, vd)   (metric_shape == SCALAR ? ONE : \
                      metric_shape == MONODIM ? M[(v)] : M[(vd)])

#define TPL template <typename real_t, typename vertex_t>
#define PFDR_PROX_TV Pfdr_prox_tv<real_t, vertex_t>

using namespace std;

TPL PFDR_PROX_TV::Pfdr_prox_tv(vertex_t V, index_t E, const vertex_t* edges,
    const real_t* Y, index_t D, D1p d1p, const real_t* d1p_coor_weights,
    Condshape metric_shape, const real_t* M)
    : Pfdr_d1<real_t, vertex_t>(V, E, edges, D, d1p, d1p_coor_weights,
        metric_shape), Y(Y), metric_shape(metric_shape), M(M)
{
    /* ensure handling of infinite values (negation, comparisons) is safe */
    static_assert(numeric_limits<real_t>::is_iec559,
        "PFDR prox TV: real_t must satisfy IEEE 754.");
    set_lipschitz_param(M, ONE, metric_shape);
}

TPL void PFDR_PROX_TV::compute_hess_f()
{
    if (metric_shape == SCALAR){
        ga = ONE;
    }else if (metric_shape == MONODIM){
        for (vertex_t v = 0; v < V; v++){ Ga[v] = M[v]; }
    }else{
        for (index_t i = 0; i < V*D; i++){ Ga[i] = M[i]; }
    }
}

TPL void PFDR_PROX_TV::compute_Ga_grad_f()
{
    #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V)
    for (vertex_t v = 0; v < V; v++){
        index_t vd = D*v;
        for (index_t d = 0; d < D; d++){
            Ga_grad_f[vd] = Ga_(v, vd)*M_(v, vd)*(X[vd] - Y[vd]);
            vd++;
        }
    }
}

TPL real_t PFDR_PROX_TV::compute_f()
{
    real_t obj = ZERO;
    #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V) \
        reduction(+:obj)
    for (vertex_t v = 0; v < V; v++){
        index_t vd = D*v;
        for (index_t d = 0; d < D; d++){
            obj += M_(v, vd)*(X[vd] - Y[vd])*(X[vd] - Y[vd]);
            vd++;
        }
    }
    return HALF*obj;
}

/**  instantiate for compilation  **/
#if defined _OPENMP && _OPENMP < 200805
/* use of unsigned counter in parallel loops requires OpenMP 3.0;
 * although published in 2008, MSVC still does not support it as of 2020 */
template class Pfdr_prox_tv<float, int16_t>;
template class Pfdr_prox_tv<float, int32_t>;
template class Pfdr_prox_tv<double, int16_t>;
template class Pfdr_prox_tv<double, int32_t>;
#else
template class Pfdr_prox_tv<float, uint16_t>;
template class Pfdr_prox_tv<float, uint32_t>;
template class Pfdr_prox_tv<double, uint16_t>;
template class Pfdr_prox_tv<double, uint32_t>;
#endif
