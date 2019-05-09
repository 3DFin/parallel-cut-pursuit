  %-------------------------------------------------------------------------%
  %  script for illustrating cp_kmpp_d0_dist on labeling of 3D point cloud  %
  %-------------------------------------------------------------------------%
% References:
% L. Landrieu and G. Obozinski, Cut Pursuit: fast algorithms to learn
% piecewise constant functions on general weighted graphs, SIAM Journal on
% Imaging Science, 10(4):1724-1766, 2017
%
% L. Landrieu et al., A structured regularization framework for spatially
% smoothing semantic labelings of 3D point clouds, ISPRS Journal of
% Photogrammetry and Remote Sensing, 132:102-118, 2017
%
% Hugo Raguet 2019
cd(fileparts(which('example_labeling_3D_d0.m')));
addpath('bin/');

%%%  classes involved in the task  %%%
classNames = {'road', 'vegetation', 'facade', 'hardscape', ...
    'scanning artifacts', 'cars'};
classId = uint8(1:6)';

%%%  parameters; see octave/doc/cp_pfdr_d1_lsx_mex.m  %%%
options = struct; % reinitialize
% options.cp_dif_tol = 1e-3;
% options.cp_it_max = 10;
% options.K = 2;
% options.split_iter_num = 2;
% options.kmpp_init_num = 3;
% options.kmpp_iter_num = 3;
% options.verbose = true;
% options.max_num_threads = 8;
% options.balance_parallel_split = true;

%%%  initialize data  %%%
% For details on the data and parameters, see H. Raguet, A Note on the
% Forward-Douglas--Rachford Splitting for Monotone Inclusion and Convex
% Optimization Optimization Letters, 2018, 1-24
load('../data/labeling_3D.mat')
% penalization adjusted for d0 norm by trial-and-error
options.edge_weights = 3*homo_d1_weight; 

% compute prediction performance of random forest
[~, ML] = max(y, [], 1);
F1 = zeros(1, length(classId));
for k=1:length(classId)
    predk = ML == classId(k);
    truek = ground_truth == classId(k);
    F1(k) = 2*sum(predk & truek)/(sum(predk) + sum(truek));
end
fprintf('\naverage F1 of random forest prediction: %.2f\n\n', mean(F1));
clear predk truek

%%%  solve the optimization problem  %%%
tic;
[Comp, rX] = cp_kmpp_d0_dist_mex(loss, y, first_edge, adj_vertices, options);
time = toc;
x = rX(:, Comp + 1); % rX is components values, Comp is components assignments
clear Comp rX;
fprintf('Total MEX execution time %.0f s\n\n', time);

% compute prediction performance of spatially regularized prediction
[~, ML] = max(x, [], 1);
F1 = zeros(1, length(classId));
for k=1:length(classId)
    predk = ML == classId(k);
    truek = ground_truth == classId(k);
    F1(k) = 2*sum(predk & truek)/(sum(predk) + sum(truek));
end
fprintf('\naverage F1 of spatially regularized prediction: %.2f\n\n', ...
    mean(F1));
clear predk truek
