/*=============================================================================
 * [first_edge_reverse, adj_vertices_reverse, reverse_arc] =
 *      forward_star_to_reverse_mex(first_edge, adj_vertices)
 * 
 *  Hugo Raguet 2020
 *===========================================================================*/
#include <cstdint>
#include "mex.h"
#include "../../include/graph_tools.hpp"

/* template for handling several index types */
template <typename index_t, mxClassID mxINDEX_CLASS>
static void forward_star_to_reverse_mex(int nlhs, mxArray **plhs, int nrhs,
    const mxArray **prhs)
{
    index_t V = mxGetNumberOfElements(prhs[0]) - 1;
    index_t E = mxGetNumberOfElements(prhs[1]);

    const index_t* first_edge = (index_t*) mxGetData(prhs[0]);
    const index_t* adj_vertices = (index_t*) mxGetData(prhs[1]);

    plhs[0] = mxCreateNumericMatrix(1, 2*V + 1, mxINDEX_CLASS, mxREAL);
    index_t* first_edge_reverse = (index_t*) mxGetData(plhs[0]);
    plhs[1] = mxCreateNumericMatrix(1, 2*E, mxINDEX_CLASS, mxREAL);
    index_t* adj_vertices_reverse = (index_t*) mxGetData(plhs[1]);
    plhs[2] = mxCreateNumericMatrix(1, 2*E, mxINDEX_CLASS, mxREAL);
    index_t* reverse_arc = (index_t*) mxGetData(plhs[2]);

    forward_star_to_reverse<index_t, index_t>(V, E, first_edge, adj_vertices,
        first_edge_reverse, adj_vertices_reverse, reverse_arc);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    if (mxGetClassID(prhs[0]) == mxUINT16_CLASS &&
        mxGetClassID(prhs[1]) == mxUINT16_CLASS){
        forward_star_to_reverse_mex<uint16_t, mxUINT16_CLASS>(nlhs, plhs,
            nrhs, prhs);
    }else if (mxGetClassID(prhs[0]) == mxUINT32_CLASS &&
              mxGetClassID(prhs[1]) == mxUINT32_CLASS){
        forward_star_to_reverse_mex<uint32_t, mxUINT32_CLASS>(nlhs, plhs,
            nrhs, prhs);
    }else if (mxGetClassID(prhs[0]) == mxUINT64_CLASS &&
              mxGetClassID(prhs[1]) == mxUINT64_CLASS){
        forward_star_to_reverse_mex<uint64_t, mxUINT64_CLASS>(nlhs, plhs,
            nrhs, prhs);
    }else{
        mexErrMsgIdAndTxt("MEX", "Forward-star to reverse: arguments must be"
            "of a same unsigned integer (16, 32, 64) class (%s and %s given).",
            mxGetClassName(prhs[0]), mxGetClassName(prhs[1]));
    }
}
