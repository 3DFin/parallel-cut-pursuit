/*=============================================================================
 * Hugo Raguet 2021, 2023
 *===========================================================================*/
#include <cmath>
#include "cp_prox_tv.hpp"
#include "pfdr_prox_tv.hpp"

#define ZERO ((real_t) 0.0) // avoid conversions
#define HALF ((real_t) 0.5) // avoid conversions
#define M_(v, vd) (metric_shape == SCALAR ? ONE : \
                   metric_shape == MONODIM ? M[(v)] : M[(vd)])

#define TPL template <typename real_t, typename index_t, typename comp_t>
#define CP_PROX_TV Cp_prox_tv<real_t, index_t, comp_t>
#define PFDR Pfdr_prox_tv<real_t, comp_t>

using namespace std;

TPL CP_PROX_TV::Cp_prox_tv(index_t V, index_t E, const index_t* first_edge,
    const index_t* adj_vertices, const real_t* Y, size_t D, D1p d1p,
    const real_t* d1p_coor_weights, Condshape metric_shape, const real_t* M)
    : Cp_d1<real_t, index_t, comp_t>(V, E, first_edge, adj_vertices, D, D1p)
{
    Y = nullptr;
    /* TODO: subgradient retrieval */
    // Gd1 = nullptr;

    K = 2;
    split_iter_num = 1;
    split_damp_ratio = 1.0;
    split_values_init_num = 2;
    split_values_iter_num = 2;

    pfdr_rho = 1.0; pfdr_cond_min = 1e-2; pfdr_dif_rcd = 0.0;
    pfdr_dif_tol = 1e-2*dif_tol; pfdr_it = pfdr_it_max = 1e4;
}

/* TODO: subgradient retrieval */
/* TPL void CP_PROX_TV::set_d1_subgradients(real_t* Gd1)
{
    this->Gd1 = Gd1;
} */

TPL void CP_PROX_TV::set_pfdr_param(real_t rho, real_t cond_min,
    real_t dif_rcd, int it_max, real_t dif_tol)
{
    this->pfdr_rho = rho;
    this->pfdr_cond_min = cond_min;
    this->pfdr_dif_rcd = dif_rcd;
    this->pfdr_it_max = it_max;
    this->pfdr_dif_tol = dif_tol;
}

TPL void CP_PROX_TV::solve_reduced_problem()
{
    /**  compute reduced matrix  **/
    real_t *rY, *rM; // reduced observations and metric shape
    rY = (real_t*) malloc_check(sizeof(real_t)*rV*D);
    rM = metric_shape == MULTIDIM ?
        (real_t*) malloc_check(sizeof(real_t)*rV*D) :
        (real_t*) malloc_check(sizeof(real_t)*rV);

    #pragma omp parallel for schedule(dynamic) NUM_THREADS(V, rV)
    for (comp_t rv = 0; rv < rV; rv++){
        rYv = rY + D*rv;
        rMv = rMv + metric_shape == MULTIDIM ? D*rv : rv;
        for (size_t d = 0; d < D; d++){
            rYv[d] = ZERO;
            if (d == 0 || metric_shape == MULTIDIM){ rMv[d] = ZERO; }
            /* run along the component rv */
            for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
                index_t v = comp_list[i];
                size_t vd = D*v + d;
                rYv[d] += M_(v, vd)*Y[vd];
                if (d == 0 || metric_shape == MULTIDIM){ rMv[d] += M_(v, vd); }
            }
            rYv[d] /= metric_shape == MULTIDIM ? rMv[d] : rMv[0];
        }
    }
    
    if (rV == 1){ /**  single connected component  **/

        for (size_t d = 0; d < D; d++){ rX[d] = rY[d]; }

    }else{ /**  preconditioned forward-Douglas-Rachford  **/

        Pfdr_prox_tv<real_t, comp_t> *pfdr =
            new Pfdr_prox_tv<real_t, comp_t>(rV, rE, reduced_edges, rY, D,
                d1p == D11 ? PFDR::D11 : PFDR::D12, d1p_coor_weights,
                metric_shape == MULTIDIM ? PFDR::MULTIDIM : PFDR::MONODIM, rM);
        
        pfdr->set_edge_weights(reduced_edge_weights);
        pfdr->set_conditioning_param(pfdr_cond_min, pfdr_dif_rcd);
        pfdr->set_relaxation(pfdr_rho);
        pfdr->set_algo_param(pfdr_dif_tol, sqrt(pfdr_it_max), pfdr_it_max,
            verbose);
        pfdr->set_iterate(rX);
        pfdr->initialize_iterate();

        pfdr_it = pfdr->precond_proximal_splitting();

        pfdr->set_iterate(nullptr); // prevent rX to be free()'d
        delete pfdr;

    }

    free(rY); free(rM);
}

TPL void CP_PROX_TV::compute_grad()
{
    /**  gradient of smooth part of d12 penalization  **/
    Cp_d1<real_t, index_t, comp_t>::compute_grad();

    /**  gradient of quadratic term  **/ 
    #pragma omp parallel for schedule(static) NUM_THREADS(V - saturated_vert)
    for (index_t v = 0; v < V; v++){
        comp_t rv = comp_assign[v];
        if (is_saturated[rv]){ continue; }

        real_t* Gv = G + D*v;
        const real_t* rXv = rX + D*rv;
        const real_t* Yv = Y + D*v;

        size_t vd = D*v;
        for (size_t d = 0; d < D; d++){
            Gv[d] += M_(v, vd)*(rXv[d] - Yv[d]);
            vd++;
        }
    }
}

TPL real_t CP_PROX_TV::compute_objective()
{
    real_t obj = ZERO;

    #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V) \
        reduction(+:obj)
    for (index_t v = 0; v < V; v++){
        rXv = rX + D*comp_assign[v];
        size_t vd = D*v;
        for (size_t d = 0; d < D; d++){
            obj += M_(v, vd)*(rXv[d] - Y[vd])*(rXv[d] - Y[vd]);
            vd++;
        }
    }
    obj *= HALF;

    obj += compute_graph_d1(); // ||x||_d1

    return obj;
}

/**  instantiate for compilation  **/
#if defined _OPENMP && _OPENMP < 200805
/* use of unsigned counter in parallel loops requires OpenMP 3.0;
 * although published in 2008, MSVC still does not support it as of 2020 */
template class Cp_prox_tv<double, int32_t, int16_t>;
template class Cp_prox_tv<float, int32_t, int16_t>;
template class Cp_prox_tv<double, int32_t, int32_t>;
template class Cp_prox_tv<float, int32_t, int32_t>;
#else
template class Cp_prox_tv<double, uint32_t, uint16_t>;
template class Cp_prox_tv<float, uint32_t, uint16_t>;
template class Cp_prox_tv<double, uint32_t, uint32_t>;
template class Cp_prox_tv<float, uint32_t, uint32_t>;
#endif
