/*=============================================================================
 * Derived class for cut-pursuit algorithm for the proximal operator of the 
 * total variation, that is square quadratic difference with penalization norm
 * over finite differences:
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
 * using cut-pursuit approach with preconditioned forward-Douglas-Rachford 
 * splitting algorithm.
 *
 * Parallel implementation with OpenMP API.
 *
 * H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing Nonsmooth
 * Functionals with Graph Total Variation, International Conference on Machine
 * Learning, PMLR, 2018, 80, 4244-4253
 *
 * Hugo Raguet 2021, 2023
 *===========================================================================*/
#pragma once
#include "cut_pursuit_d1.hpp"

/* real_t is the real numeric type, used for the base field and for the
 * objective functional computation;
 * index_t must be able to represent the number of vertices and of (undirected)
 * edges in the main graph;
 * comp_t must be able to represent the number of constant connected components
 * in the reduced graph */
template <typename real_t, typename index_t, typename comp_t>
class Cp_prox_tv : public Cp_d1<real_t, index_t, comp_t>
{

    /**  type resolution for base template class members
     * https://isocpp.org/wiki/faq/templates#nondependent-name-lookup-members
     **/
private:
    using Cp<real_t, index_t, comp_t>::dif_tol;
public:
    /* for multidimensional data, type of graph total variation */
    using typename Cp_d1<real_t, index_t, comp_t>::D1p;
    using Cp_d1<real_t, index_t, comp_t>::D11;
    using Cp_d1<real_t, index_t, comp_t>::D12;
    /* shape of metric in quadratic part;
     * SCALAR for a scalar operator (upright shape);
     * MONODIM for a diagonal operator, constant along coordinates in case of
     * multidimensional data points (D > 1);
     * MULTIDIM for a diagonal operator, with possibly different values at
     * each coordinate of each data point */
    enum Condshape {SCALAR, MONODIM, MULTIDIM};

public:
    /**  constructor, destructor  **/

    /* see members Y, M for details on the quadratic part */
    Cp_prox_tv(index_t V, index_t E, const index_t* first_edge,
        const index_t* adj_vertices, const real_t* Y, size_t D = 1,
        D1p d1p = D11, const real_t* d1p_coor_weights = nullptr,
        Condshape metric_shape = IDENTITY, const real_t* M = nullptr);

    /* delegation for monodimensional setting */
    Cp_prox_tv(index_t V, index_t E, const index_t* first_edge,
        const index_t* adj_vertices, const real_t* Y) :
        Cp_prox_tv(V, E, first_edge, adj_vertices, Y, 1, D11);

    /* the destructor does not free pointers which are supposed to be provided 
     * by the user (forward-star graph structure given at construction, 
     * monitoring arrays, matrix and observation arrays); IT DOES FREE THE REST
     * (components assignment and reduced problem elements, etc.), but this can
     * be prevented by getting the corresponding pointer member and setting it
     * to null beforehand */

    /* set an array for storing d1 subgradients, see member Gd1 for details */
    /* TODO: subgradient retrieval */
    // void set_d1_subgradients(real_t* Gd1);

    void set_pfdr_param(real_t rho, real_t cond_min, real_t dif_rcd,
        int it_max, real_t dif_tol);

    /* overload for default dif_tol parameter */
    void set_pfdr_param(real_t rho = 1.0, real_t cond_min = 1e-3,
        real_t dif_rcd = 0.0, int it_max = 1e4)
    { set_pfdr_param(rho, cond_min, dif_rcd, it_max, 1e-3*dif_tol); }

private:
    /**  main problem, quadratic part  **/

    const real_t* Y; /* observations, array of length V */
    const Condshape metric_shape; /* shape of the metric M */
    const real_t* M; /* diagonal metric (positive weights) on squared l2-norm;
        * null pointer if metric_shape is IDENTITY
        * array or length V if metric_shape is MONODIM
        * D-by-V array, column major format, if metric_shape is MULTIDIM */

    /* TODO: subgradient retrieval */
    // real_t* Gd1; // subgradients of d1 (total variation penalization)

    /**  reduced problem  **/

    real_t pfdr_rho, pfdr_cond_min, pfdr_dif_rcd, pfdr_dif_tol;
    int pfdr_it, pfdr_it_max;

    /**  cut-pursuit steps  **/

    /* split */

    void compute_grad() override;

    /* compute reduced values */
    void solve_reduced_problem() override;

    /* relative iterate evolution in l2 norm and components saturation */
    real_t compute_evolution(bool compute_dif) override;

    real_t compute_objective() override;

    /**  type resolution for base template class members
     * https://isocpp.org/wiki/faq/templates#nondependent-name-lookup-members
     **/
    using Cp_d1<real_t, index_t, comp_t>::G;
    using Cp_d1<real_t, index_t, comp_t>::compute_graph_d1;
    using Cp<real_t, index_t, comp_t>::split_iter_num;
    using Cp<real_t, index_t, comp_t>::split_damp_ratio;
    using Cp<real_t, index_t, comp_t>::split_values_init_num;
    using Cp<real_t, index_t, comp_t>::split_values_iter_num;
    using Cp<real_t, index_t, comp_t>::K;
    using Cp<real_t, index_t, comp_t>::rX;
    using Cp<real_t, index_t, comp_t>::is_saturated;
    using Cp<real_t, index_t, comp_t>::saturated_comp;
    using Cp<real_t, index_t, comp_t>::saturated_vert;
    using Cp<real_t, index_t, comp_t>::V;
    using Cp<real_t, index_t, comp_t>::E;
    using Cp<real_t, index_t, comp_t>::D;
    using Cp<real_t, index_t, comp_t>::first_edge;
    using Cp<real_t, index_t, comp_t>::adj_vertices; 
    using Cp<real_t, index_t, comp_t>::rV;
    using Cp<real_t, index_t, comp_t>::rE;
    using Cp<real_t, index_t, comp_t>::comp_assign;
    using Cp<real_t, index_t, comp_t>::comp_list;
    using Cp<real_t, index_t, comp_t>::label_assign;
    using Cp<real_t, index_t, comp_t>::first_vertex;
    using Cp<real_t, index_t, comp_t>::reduced_edges;
    using Cp<real_t, index_t, comp_t>::reduced_edge_weights;
    using Cp<real_t, index_t, comp_t>::verbose;
    using Cp<real_t, index_t, comp_t>::malloc_check;
};
