/*=============================================================================
 * [Comp, rX, List, Gtv, Obj, Time, Dif] = cp_prox_tv(Y, first_edge,
 *      adj_vertices, [options])
 *
 * options is a struct with any of the following fields [with default values]:
 *
 *      edge_weights [1.0], cp_dif_tol [1e-4], cp_it_max [10], pfdr_rho [1.0],
 *      pfdr_cond_min [1e-2], pfdr_dif_rcd [0.0],
 *      pfdr_dif_tol [1e-2*cp_dif_tol], pfdr_it_max [1e4], verbose [1e3],
 *      max_num_threads [none], balance_parallel_split [true]
 * 
 *  Hugo Raguet 2022
 *===========================================================================*/
#include <cstdint>
#include <cstring>
#include "mex.h"
#include "cp_prox_tv.hpp"

using namespace std;

/* index_t must be able to represent the number of vertices and of (undirected)
 * edges in the main graph;
 * comp_t must be able to represent the number of constant connected components
 * in the reduced graph */
#if defined _OPENMP && _OPENMP < 200805
/* use of unsigned iterator in parallel loops requires OpenMP 3.0;
 * although published in 2008, MSVC still does not support it as of 2020 */
    typedef int32_t index_t;
    # define mxINDEX_CLASS mxINT32_CLASS
    # define INDEX_T_STRING "int32"
    /* comment the following if more than 32767 components are expected */
    typedef int16_t comp_t;
    # define mxCOMP_CLASS mxINT16_CLASS
    /* uncomment the following if more than 32767 components are expected */
    // typedef int32_t comp_t;
    // #define mxCOMP_CLASS mxINT32_CLASS
#else
    typedef uint32_t index_t;
    #define mxINDEX_CLASS mxUINT32_CLASS
    #define INDEX_T_STRING "uint32"
    /* comment the following if more than 65535 components are expected */
    typedef uint16_t comp_t;
    #define mxCOMP_CLASS mxUINT16_CLASS
    /* uncomment the following if more than 65535 components are expected */
    // typedef uint32_t comp_t;
    // #define mxCOMP_CLASS mxUINT32_CLASS
#endif

/* function for checking optional parameters */
static void check_opts(const mxArray* options)
{
    if (!options){ return; }

    if (!mxIsStruct(options)){
        mexErrMsgIdAndTxt("MEX", "Cut-pursuit prox TV: "
            "fourth parameter 'options' should be a structure, (%s given).",
            mxGetClassName(options));
    }

    const int num_allow_opts = 11;
    const char* opts_names[] = {"edge_weights", "cp_dif_tol", "cp_it_max",
        "pfdr_rho", "pfdr_cond_min", "pfdr_dif_rcd", "pfdr_dif_tol",
        "pfdr_it_max", "verbose", "max_num_threads", "balance_parallel_split"};

    const int num_given_opts = mxGetNumberOfFields(options);

    for (int given_opt = 0; given_opt < num_given_opts; given_opt++){
        const char* opt_name = mxGetFieldNameByNumber(options, given_opt);
        int allow_opt;
        for (allow_opt = 0; allow_opt < num_allow_opts; allow_opt++){
            if (strcmp(opt_name, opts_names[allow_opt]) == 0){ break; }
        }
        if (allow_opt == num_allow_opts){
            mexErrMsgIdAndTxt("MEX", "Cut-pursuit d1 quadratic l1 bounds: "
                "option '%s' unknown.", opt_name);
        }
    }
}

/* function for checking parameter type */
static void check_arg_class(const mxArray* arg, const char* arg_name,
    mxClassID class_id, const char* class_name)
{
    if (mxGetNumberOfElements(arg) > 1 && mxGetClassID(arg) != class_id){
        mexErrMsgIdAndTxt("MEX", "Cut-pursuit d1 quadratic l1 bounds: "
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
static void cp_prox_tv_mex(int nlhs, mxArray *plhs[], int nrhs,
    const mxArray *prhs[])
{
    /***  get inputs  ***/

    const char* real_class_name = mxREAL_CLASS == mxDOUBLE_CLASS ?
        "double" : "single";

    /** quadratic functional **/

    index_t V = mxGetNumberOfElements(prhs[0]);
    const real_t* Y = (real_t*) mxGetData(prhs[0]);

    /**  graph structure  **/

    check_arg_class(prhs[1], "first_edge", mxINDEX_CLASS, INDEX_T_STRING);
    check_arg_class(prhs[2], "adj_vertices", mxINDEX_CLASS, INDEX_T_STRING);

    const index_t* first_edge = (index_t*) mxGetData(prhs[1]);
    const index_t* adj_vertices = (index_t*) mxGetData(prhs[2]);
    index_t E = mxGetNumberOfElements(prhs[2]);

    size_t first_edge_length = mxGetNumberOfElements(prhs[1]);
    if (first_edge_length != (size_t) V + 1){
        mexErrMsgIdAndTxt("MEX", "Cut-pursuit prox TV: "
            "third parameter 'first_edge' should contain |V| + 1 = %d "
            "elements (%d given).", (size_t) V + 1, first_edge_length);
    }

    /**  optional parameters  **/

    const mxArray* options = nrhs > 3 ? prhs[3] : nullptr;

    check_opts(options);
    const mxArray* opt;

    /* penalizations */
    const real_t* edge_weights = nullptr;
    real_t homo_edge_weights = 1.0;
    if (options && (opt = mxGetField(options, 0, "edge_weights"))){
        check_arg_class(opt, "edge_weights", mxREAL_CLASS, real_class_name);
        if (mxGetNumberOfElements(opt) > 1){
            edge_weights = (real_t*) mxGetData(opt);
        }else{
            homo_edge_weights = mxGetScalar(opt);
        }
    }

    /* algorithmic parameters */
    #define GET_SCAL_OPT(NAME, DFLT) \
        NAME = (opt = mxGetField(options, 0, #NAME)) ? mxGetScalar(opt) : DFLT;

    real_t GET_SCAL_OPT(cp_dif_tol, 1e-4);
    int GET_SCAL_OPT(cp_it_max, 10);
    real_t GET_SCAL_OPT(pfdr_rho, 1.0);
    real_t GET_SCAL_OPT(pfdr_cond_min, 1e-2);
    real_t GET_SCAL_OPT(pfdr_dif_rcd, 0.0);
    real_t GET_SCAL_OPT(pfdr_dif_tol, 1e-2*cp_dif_tol);
    int GET_SCAL_OPT(pfdr_it_max, 1e4);
    int GET_SCAL_OPT(verbose, 1e3);
    int GET_SCAL_OPT(max_num_threads, 0);
    bool GET_SCAL_OPT(balance_parallel_split, true);

    /***  prepare output; rX (plhs[1]) is created later  ***/

    plhs[0] = mxCreateNumericMatrix(1, V, mxCOMP_CLASS, mxREAL);
    comp_t *Comp = (comp_t*) mxGetData(plhs[0]);

    real_t* Gtv = nlhs > 3 ?
        (real_t*) mxMalloc(sizeof(real_t)*E) : nullptr;
    real_t* Obj = nlhs > 4 ?
        (real_t*) mxMalloc(sizeof(real_t)*(cp_it_max + 1)) : nullptr;
    double* Time = nlhs > 5 ?
        (double*) mxMalloc(sizeof(double)*(cp_it_max + 1)) : nullptr;
    real_t *Dif = nlhs > 6 ?
        (real_t*) mxMalloc(sizeof(real_t)*cp_it_max) : nullptr;

    /***  cut-pursuit with preconditioned forward-Douglas-Rachford  ***/

    Cp_prox_tv<real_t, index_t, comp_t> *cp =
        new Cp_prox_tv<real_t, index_t, comp_t>
            (V, E, first_edge, adj_vertices);

    cp->set_edge_weights(edge_weights, homo_edge_weights);
    cp->set_observation(Y);
    cp->set_d1_subgradients(Gtv);
    cp->set_cp_param(cp_dif_tol, cp_it_max, verbose);
    cp->set_pfdr_param(pfdr_rho, pfdr_cond_min, pfdr_dif_rcd, pfdr_it_max,
        pfdr_dif_tol);
    cp->set_parallel_param(max_num_threads, balance_parallel_split);
    cp->set_monitoring_arrays(Obj, Time, Dif);
    cp->set_components(0, Comp); // use the preallocated component array Comp

    int cp_it = cp->cut_pursuit();

    /* get number of components and list of indices */
    const index_t* first_vertex;
    const index_t* comp_list;
    comp_t rV = cp->get_components(nullptr, &first_vertex, &comp_list);

    /* copy reduced values */
    real_t* cp_rX = cp->get_reduced_values();
    plhs[1] = mxCreateNumericMatrix(rV, 1, mxREAL_CLASS, mxREAL);
    real_t* rX = (real_t*) mxGetData(plhs[1]);
    for (comp_t rv = 0; rv < rV; rv++){ rX[rv] = cp_rX[rv]; }

    /* get number of components and list of indices */
    if (nlhs > 2){
        plhs[2] = mxCreateCellMatrix(1, rV); // list of arrays
        for (comp_t rv = 0; rv < rV; rv++){
            index_t comp_size = first_vertex[rv+1] - first_vertex[rv];
            mxArray* mxList_rv = mxCreateNumericMatrix(1, comp_size,
                mxINDEX_CLASS, mxREAL);
            index_t* List_rv = (index_t*) mxGetData(mxList_rv);
            for (index_t i = 0; i < comp_size; i++){
                List_rv[i] = comp_list[first_vertex[rv] + i];
            }
            mxSetCell(plhs[2], rv, mxList_rv);
        }
    }
    
    cp->set_components(0, nullptr); // prevent Comp to be free()'d
    delete cp;

    /**  resize monitoring arrays and assign to outputs  **/
    if (nlhs > 2){
        plhs[2] = resize_and_create_mxRow(Obj, cp_it + 1, mxREAL_CLASS);
    }
    if (nlhs > 3){
        plhs[3] = resize_and_create_mxRow(Time, cp_it + 1, mxDOUBLE_CLASS);
    }
    if (nlhs > 4){
        plhs[4] = resize_and_create_mxRow(Dif, cp_it, mxREAL_CLASS);
    }

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    /* real type is determined by the first parameter Y */
    if (mxIsDouble(prhs[0])){
        cp_prox_tv_mex<double, mxDOUBLE_CLASS>(nlhs, plhs, nrhs, prhs);
    }else{
        cp_prox_tv_mex<float, mxSINGLE_CLASS>(nlhs, plhs, nrhs, prhs);
    }
}
