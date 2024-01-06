/*=============================================================================
 * Derived class for preconditioned forward-Douglas–Rachford algorithm for the
 * proximal operator of the graph total variation, that is square quadratic
 * difference with penalization norm over finite differences:
 * 
 * minimize functional over a graph G = (V, E)
 *
 *        F(x) = 1/2 ||y - x||_M^2 + ||x||_d1p
 *
 * where y,x in R^{D-by-V}, M in R^{V-by-V}, and 
 *      ||x||_M^2 = sum_{v in V} ||x_u - x_v||_{m_uv}^2.
 *  (the norm is a diagonaly weighted squared l2-norm),
 * and  ||x||_d1p = sum_{uv in E} w_uv ||x_u - x_v||_p.
 *  (the norm can be a l1 or a l2-norm)
 *
 * Parallel implementation with OpenMP API.
 *
 * H. Raguet and L. Landrieu, Preconditioning of a Generalized Forward-Backward
 * Splitting and Application to Optimization on Graphs, SIAM Journal on Imaging
 * Sciences, 2015, 8, 2706-2739
 *
 * Hugo Raguet 2023
 *===========================================================================*/
#pragma once
#include "pfdr_graph_d1.hpp"

/* vertex_t is an integer type able to represent the number of vertices */
template <typename real_t, typename vertex_t>
class Pfdr_prox_tv : public Pfdr_d1<real_t, vertex_t>
{
public:

    /**  type resolution for base template class members
     * https://isocpp.org/wiki/faq/templates#nondependent-name-lookup-members
     **/
    using typename Pfdr_d1<real_t, vertex_t>::index_t;
    using typename Pfdr_d1<real_t, vertex_t>::D1p;
    using Pfdr_d1<real_t, vertex_t>::D11;
    using Pfdr_d1<real_t, vertex_t>::D12;
    /* reuse the Conshape type (see pcd_fwd_doug_rach.hpp) for the shape of
     * the metric M in the quadratic part (see declaration of M) */
    using typename Pfdr<real_t, vertex_t>::Condshape;
    using Pfdr<real_t, vertex_t>::SCALAR;
    using Pfdr<real_t, vertex_t>::MONODIM;
    using Pfdr<real_t, vertex_t>::MULTIDIM;

    /**  constructor, destructor  **/

    /* see members Y, M for details on the quadratic part */
    Pfdr_prox_tv(vertex_t V, index_t E, const vertex_t* edges, const real_t* Y,
        index_t D, D1p d1p = D12, const real_t* d1p_coor_weights = nullptr,
        Condshape metric_shape = SCALAR, const real_t* M = nullptr);

    /* delegation for monodimensional setting */
    Pfdr_prox_tv(vertex_t V, index_t E, const vertex_t* edges, const real_t* Y)
        : Pfdr_prox_tv(V, E, edges, Y, 1, D11);

    /* the destructor does not free pointers which are supposed to be provided 
     * by the user (adjacency graph structure given at construction, 
     * monitoring arrays, matrix and observation arrays); it does free the rest
     * (iterate, auxiliary variables etc.), but this can be prevented by
     * copying the corresponding pointer member and set it to null before
     * deleting */

private:
    /**  quadratic problem  **/

    const real_t* Y; /* observations, array of length V */
    const Condshape metric_shape; /* shape of the metric M */
    const real_t* M; /* diagonal metric (positive weights) on squared l2-norm;
        * null pointer if metric_shape is SCALAR
        * array or length V if metric_shape is MONODIM
        * D-by-V array, column major format, if metric_shape is MULTIDIM */

    /**  specialization of base virtual methods  **/

    /* approximate diagonal hessian of quadratic functional */
    void compute_hess_f() override;

    /* compute the gradient of the quadratic functional in Pfdr::Ga_grad_f */
    void compute_Ga_grad_f() override;

    /* quadratic functional; in the precomputed A^t A version, 
     * a constant 1/2||Y||^2 is omited */
    real_t compute_f() override; 

    /**  type resolution for base template class members
     * https://isocpp.org/wiki/faq/templates#nondependent-name-lookup-members
     **/
    using Pfdr_d1<real_t, vertex_t>::V;
    using Pfdr_d1<real_t, vertex_t>::E;
    using Pfdr<real_t, vertex_t>::set_lipschitz_param;
    using Pfdr<real_t, vertex_t>::Ga_grad_f;
    using Pfdr<real_t, vertex_t>::Ga;
    using Pfdr<real_t, vertex_t>::ga;
    using Pfdr<real_t, vertex_t>::gashape;
    using Pfdr<real_t, vertex_t>::D;
    using Pcd_prox<real_t>::X;
    using Pcd_prox<real_t>::last_X;
    using Pcd_prox<real_t>::cond_min;
    using Pcd_prox<real_t>::dif_tol;
    using Pcd_prox<real_t>::dif_rcd;
    using Pcd_prox<real_t>::iterate_evolution;
    using Pcd_prox<real_t>::eps;
    using Pcd_prox<real_t>::malloc_check;
};
