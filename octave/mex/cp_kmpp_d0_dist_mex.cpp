/*=============================================================================
 * [Comp, rX, it, Obj, Time, Dif] = cp_kmpp_d0_dist_mex(loss, Y, first_edge,
 *      adj_vertices, [options])
 *
 * options is a struct with any of the following fields [with default values]:
 *
 *      reverse_arc [none], edge_weights [1.0], vert_weights [none],
 *      coor_weights [none], cp_dif_tol [1e-3], cp_it_max [10], K [2],
 *      split_iter_num [2], kmpp_init_num [3], kmpp_iter_num [3],
 *      verbose [true], max_num_threads [none], balance_parallel_split [true]
 * 
 *  Hugo Raguet 2019, 2020
 *===========================================================================*/
#include <cstdint>
#include <cstring>
#include "mex.h"
#include "../../include/cp_kmpp_d0_dist.hpp"
#include "../../include/graph_tools.hpp"

using namespace std;

/* index_t must be able to represent twice the number of vertices plus one and
 * twice the number of edges plus one in the main graph;
 * comp_t must be able to represent the number of constant connected components
 * plus one in the reduced graph, as well as the dimension D */
typedef uint32_t index_t;
# define mxINDEX_CLASS mxUINT32_CLASS
# define INDEX_CLASS_NAME "uint32"
/* comment the following if more than 65535 components are expected */
typedef uint16_t comp_t;
# define mxCOMP_CLASS mxUINT16_CLASS
/* uncomment the following if more than 65535 components are expected */
// typedef uint32_t comp_t;
// #define mxCOMP_CLASS mxUINT32_CLASS

/* function for checking optional parameters */
static void check_opts(const mxArray* options)
{
    if (!options){ return; }

    if (!mxIsStruct(options)){
        mexErrMsgIdAndTxt("MEX", "Cut-pursuit d0 distance: "
            "fifth parameter 'options' should be a structure, (%s given).",
            mxGetClassName(options));
    }

    const int num_allow_opts = 13;
    const char* opts_names[] = {"reverse_arc", "edge_weights", "vert_weights",
        "coor_weights", "cp_dif_tol", "cp_it_max", "K", "split_iter_num",
        "kmpp_init_num", "kmpp_iter_num", "verbose", "max_num_threads",
        "balance_parallel_split"};

    const int num_given_opts = mxGetNumberOfFields(options);

    for (int given_opt = 0; given_opt < num_given_opts; given_opt++){
        const char* opt_name = mxGetFieldNameByNumber(options, given_opt);
        int allow_opt;
        for (allow_opt = 0; allow_opt < num_allow_opts; allow_opt++){
            if (strcmp(opt_name, opts_names[allow_opt]) == 0){ break; }
        }
        if (allow_opt == num_allow_opts){
            mexErrMsgIdAndTxt("MEX", "Cut-pursuit d0 distance: "
                "option '%s' unknown.", opt_name);
        }
    }
}

/* function for checking parameter type */
static void check_arg_class(const mxArray* arg, const char* arg_name,
    mxClassID class_id, const char* class_name)
{
    if (mxGetNumberOfElements(arg) > 1 && mxGetClassID(arg) != class_id){
        mexErrMsgIdAndTxt("MEX", "Cut-pursuit d0 distance: "
            "parameter '%s' should be of class %s (%s given).",
            arg_name, class_name, mxGetClassName(arg), class_name);
    }
}

/* resize memory buffer allocated by mxMalloc and create a row vector */
template <typename type_t>
static mxArray* resize_and_create_mxRow(type_t* buffer, size_t size,
    mxClassID class_id)
{
    mxArray* row = mxCreateNumericMatrix(0, 0, class_id, mxREAL);
    if (size){
        mxSetM(row, 1);
        mxSetN(row, size);
        buffer = (type_t*) mxRealloc((void*) buffer, sizeof(type_t)*size);
        mxSetData(row, (void*) buffer);
    }else{
        mxFree((void*) buffer);
    }
    return row;
}

/* template for handling both single and double precisions */
template <typename real_t, mxClassID mxREAL_CLASS>
static void cp_kmpp_d0_dist_mex(int nlhs, mxArray *plhs[], int nrhs,
    const mxArray *prhs[])
{
    /***  get inputs  ***/

    const char* real_class_name = mxREAL_CLASS == mxDOUBLE_CLASS ?
        "double" : "single";

    /**  sizes and loss  **/

    real_t loss = mxGetScalar(prhs[0]);
    size_t D = mxGetM(prhs[1]);
    index_t V = mxGetN(prhs[1]);
    if (V == 1 && D > 1){ // column vector given
        V = D;
        D = 1;
    }
    const real_t* Y = (real_t*) mxGetData(prhs[1]);

    /**  graph structure  **/

    check_arg_class(prhs[2], "first_edge", mxINDEX_CLASS, INDEX_CLASS_NAME);
    check_arg_class(prhs[3], "adj_vertices", mxINDEX_CLASS, INDEX_CLASS_NAME);

    const index_t *first_edge = (index_t*) mxGetData(prhs[2]);
    const index_t *adj_vertices = (index_t*) mxGetData(prhs[3]);
    const index_t first_edge_length = mxGetNumberOfElements(prhs[2]);
    const index_t adj_vertices_length = mxGetNumberOfElements(prhs[3]);

    index_t E;
    const index_t* reverse_arc;

    if (nrhs < 5 || !mxGetField(prhs[4], 0, "reverse_arc")){
        if (first_edge_length != (V + 1)){
            mexErrMsgIdAndTxt("MEX", "Cut-pursuit d0 distance: "
                "third parameter 'first_edge' should contain |V| + 1 = %d "
                "elements (%d given).", (V + 1), first_edge_length);
        }
        E = adj_vertices_length;

        /* compute and store two-ways forward-star graph structure */
        index_t* first_edge_rev = (index_t*)
            mxMalloc(sizeof(index_t)*(2*V + 1));
        index_t* adj_vertices_rev = (index_t*) mxMalloc(sizeof(index_t)*2*E);
        index_t* rev_arc = (index_t*) mxMalloc(sizeof(index_t)*2*E);

        forward_star_to_reverse<index_t, index_t>(V, E, first_edge, 
            adj_vertices, first_edge_rev, adj_vertices_rev, rev_arc);

        first_edge = first_edge_rev;
        adj_vertices = adj_vertices_rev;
        reverse_arc = rev_arc;
    }else{
        if (first_edge_length != (2*V + 1)){
            mexErrMsgIdAndTxt("MEX", "Cut-pursuit d0 distance: "
                "when option 'reverse_arc' is provided, third parameter"
                "'first_edge' should contain 2|V| + 1 = %d elements "
                "(%d given).", (2*V + 1), first_edge_length);
        }
        E = adj_vertices_length/2;

        reverse_arc = (index_t*) mxGetData(mxGetField(prhs[4], 0,
            "reverse_arc"));
    }


    /**  optional parameters  **/

    const mxArray* options = nrhs > 4 ? prhs[4] : nullptr;

    check_opts(options);
    const mxArray* opt;

    /* loss and penalizations */
    #define GET_REAL_OPT(NAME) \
        const real_t* NAME = nullptr; \
        if (opt = mxGetField(options, 0, #NAME)){ \
            check_arg_class(opt, #NAME, mxREAL_CLASS, real_class_name); \
            NAME = (real_t*) mxGetData(opt); \
        }

    GET_REAL_OPT(vert_weights)
    GET_REAL_OPT(coor_weights)
    GET_REAL_OPT(edge_weights)
    real_t homo_edge_weight = 1.0;
    if (opt && mxGetNumberOfElements(opt) == 1){
        edge_weights = nullptr;
        homo_edge_weight = mxGetScalar(opt);
    }

    /* algorithmic parameters */
    #define GET_SCAL_OPT(NAME, DFLT) \
        NAME = (opt = mxGetField(options, 0, #NAME)) ? mxGetScalar(opt) : DFLT;

    real_t GET_SCAL_OPT(cp_dif_tol, 1e-3);
    int GET_SCAL_OPT(cp_it_max, 10);
    int GET_SCAL_OPT(K, 2);
    int GET_SCAL_OPT(split_iter_num, 2);
    int GET_SCAL_OPT(kmpp_init_num, 3);
    int GET_SCAL_OPT(kmpp_iter_num, 3);
    bool GET_SCAL_OPT(verbose, true);
    int GET_SCAL_OPT(max_num_threads, 0);
    bool GET_SCAL_OPT(balance_parallel_split, true);

    /***  prepare output; rX (plhs[1]) is created later  ***/

    plhs[0] = mxCreateNumericMatrix(1, V, mxCOMP_CLASS, mxREAL);
    comp_t* Comp = (comp_t*) mxGetData(plhs[0]);
    plhs[2] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    int* it = (int*) mxGetData(plhs[2]);

    real_t* Obj = nlhs > 3 ?
        (real_t*) mxMalloc(sizeof(real_t)*(cp_it_max + 1)) : nullptr;
    double* Time = nlhs > 4 ?
        (double*) mxMalloc(sizeof(double)*(cp_it_max + 1)) : nullptr;
    real_t* Dif = nlhs > 5 ?
        (real_t*) mxMalloc(sizeof(double)*cp_it_max) : nullptr;

    /***  cut-pursuit with preconditioned forward-Douglas-Rachford  ***/

    Cp_d0_dist<real_t, index_t, comp_t> *cp =
        new Cp_d0_dist<real_t, index_t, comp_t>
            (V, E, first_edge, adj_vertices, reverse_arc, Y, D);

    cp->set_loss(loss, Y, vert_weights, coor_weights);
    cp->set_edge_weights(edge_weights, homo_edge_weight);
    cp->set_cp_param(cp_dif_tol, cp_it_max, verbose);
    cp->set_split_param(K, split_iter_num);
    cp->set_kmpp_param(kmpp_init_num, kmpp_iter_num);
    cp->set_parallel_param(max_num_threads, balance_parallel_split);
    cp->set_monitoring_arrays(Obj, Time, Dif);

    cp->set_components(0, Comp); // use the preallocated component array Comp

    *it = cp->cut_pursuit();

    /* copy reduced values */
    comp_t rV = cp->get_components();
    real_t* cp_rX = cp->get_reduced_values();
    plhs[1] = mxCreateNumericMatrix(D, rV, mxREAL_CLASS, mxREAL);
    real_t* rX = (real_t*) mxGetData(plhs[1]);
    for (size_t rvd = 0; rvd < rV*D; rvd++){ rX[rvd] = cp_rX[rvd]; }
    
    cp->set_components(0, nullptr); // prevent Comp to be free()'d
    delete cp;

    /**  resize monitoring arrays and assign to outputs  **/
    if (nlhs > 3){
        plhs[3] = resize_and_create_mxRow(Obj, *it + 1, mxREAL_CLASS);
    }
    if (nlhs > 4){
        plhs[4] = resize_and_create_mxRow(Time, *it + 1, mxDOUBLE_CLASS);
    }
    if (nlhs > 5){
        plhs[5] = resize_and_create_mxRow(Dif, *it, mxREAL_CLASS);
    }

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    /* real type is determined by second parameter Y */
    if (mxIsDouble(prhs[1])){
        cp_kmpp_d0_dist_mex<double, mxDOUBLE_CLASS>(nlhs, plhs, nrhs, prhs);
    }else{
        cp_kmpp_d0_dist_mex<float, mxSINGLE_CLASS>(nlhs, plhs, nrhs, prhs);
    }
}
