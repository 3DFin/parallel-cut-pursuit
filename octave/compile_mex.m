origDir = pwd; % remember working directory
cd(fileparts(which('compile_mex.m'))); 
if ~exist('bin/'), mkdir('bin/'); end
try
    % compilation flags 
    [~, CXXFLAGS] = system('mkoctfile -p CXXFLAGS');
    [~, LDFLAGS] = system('mkoctfile -p LDFLAGS');
    % some versions introduces a newline character (10)
    % in the output of 'system'; this must be removed
    if CXXFLAGS(end)==10, CXXFLAGS = CXXFLAGS(1:end-1); end
    if LDFLAGS(end)==10, LDFLAGS = LDFLAGS(1:end-1); end
    CXXFLAGSorig = CXXFLAGS;
    LDFLAGSorig = LDFLAGS;
    % _GLIBCXX_PARALLEL is only useful for libstdc++ users
    % MIN_OPS_PER_THREAD roughly controls parallelization, see doc in README.md
    CXXFLAGS = [CXXFLAGS ' -Wextra -Wpedantic -std=c++11 -fopenmp -g0 ' ...
        '-D_GLIBCXX_PARALLEL -DMIN_OPS_PER_THREAD=10000'];
    LDFLAGS = [LDFLAGS ' -fopenmp'];
    setenv('CXXFLAGS', CXXFLAGS);
    setenv('LDFLAGS', LDFLAGS);

    %{
    mex -I../include mex/cp_d1_ql1b_mex.cpp ../src/cp_d1_ql1b.cpp ...
        ../src/cut_pursuit_d1.cpp ../src/cut_pursuit.cpp ... 
        ../src/maxflow.cpp ../src/matrix_tools.cpp ...
        ../src/pfdr_d1_ql1b.cpp ../src/pfdr_graph_d1.cpp ...
        ../src/pcd_fwd_doug_rach.cpp ../src/pcd_prox_split.cpp ...
        -output bin/cp_d1_ql1b
    clear cp_d1_ql1b
    %}

    %{
    mex -I../include mex/cp_d1_lsx_mex.cpp ../src/cp_d1_lsx.cpp ...
        ../src/cut_pursuit_d1.cpp ../src/cut_pursuit.cpp ...
        ../src/maxflow.cpp ../src/proj_simplex.cpp ...
        ../src/pfdr_d1_lsx.cpp ../src/pfdr_graph_d1.cpp ...
        ../src/pcd_fwd_doug_rach.cpp ../src/pcd_prox_split.cpp ...
        -output bin/cp_d1_lsx
    clear cp_d1_lsx
    %}

    % %{
    mex -I../include mex/cp_d0_dist_mex.cpp ../src/cp_d0_dist.cpp ...
        ../src/cut_pursuit_d0.cpp ../src/cut_pursuit.cpp ...
        ../src/maxflow.cpp ...
        -output bin/cp_d0_dist_
    clear cp_d0_dist_
    %}

    %{
    mex -I../include mex/cp_prox_tv_mex.cpp ../src/cp_prox_tv.cpp ...
        ../src/cut_pursuit_d1.cpp ../src/cut_pursuit.cpp ... 
        ../src/maxflow.cpp ../src/pfdr_d1_ql1b.cpp ../src/pfdr_graph_d1.cpp ...
        ../src/pcd_fwd_doug_rach.cpp ../src/pcd_prox_split.cpp ...
        ../src/matrix_tools.cpp ...
        -output bin/cp_prox_tv
    clear cp_prox_tv
    %}

    if exist('cut_pursuit.o'), system('rm *.o'); end
catch % if an error occur, makes sure not to change the working directory
    % back to original environment
    setenv('CXXFLAGS', CXXFLAGSorig);
    setenv('LDFLAGS', LDFLAGSorig);
    cd(origDir);
	rethrow(lasterror);
end
% back to original environment
setenv('CXXFLAGS', CXXFLAGSorig);
setenv('LDFLAGS', LDFLAGSorig);
cd(origDir);
