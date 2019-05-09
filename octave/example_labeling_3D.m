  %------------------------------------------------------------------------%
  %  script for illustrating cp_pfdr_d1_lsx on labeling of 3D point cloud  %
  %------------------------------------------------------------------------%
% Reference: H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing
% Nonsmooth Functionals with Graph Total Variation, International Conference on
% Machine Learning, PMLR, 2018, 80, 4244-4253
%
% Hugo Raguet 2017, 2018, 2019
cd(fileparts(which('example_labeling_3D.m')));
addpath('bin/');

%%%  classes involved in the task  %%%
classNames = {'road', 'vegetation', 'facade', 'hardscape', ...
    'scanning artifacts', 'cars'};
classId = uint8(1:6)';

%%%  parameters; see octave/doc/cp_pfdr_d1_lsx_mex.m  %%%
options = struct; % reinitialize
% options.cp_dif_tol = 1e-3;
% options.cp_it_max = 10;
options.pfdr_rho = 1.5;
% options.pfdr_cond_min = 1e-2;
% options.pfdr_dif_rcd = 0.0;
% options.pfdr_dif_tol = 1e-3*options.cp_dif_tol;
% options.pfdr_it_max = 1e4;
% options.pfdr_verbose = 1e2;
% options.max_num_threads = 0;
% options.balance_parallel_split = true;

%%%  initialize data  %%%
% For details on the data and parameters, see H. Raguet, A Note on the
% Forward-Douglas--Rachford Splitting for Monotone Inclusion and Convex
% Optimization Optimization Letters, 2018, 1-24
load('../data/labeling_3D.mat')
options.edge_weights = homo_d1_weight;

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
[Comp, rX] = cp_pfdr_d1_lsx_mex(loss, y, first_edge, adj_vertices, options);
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
