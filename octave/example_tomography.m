       %------------------------------------------------------------%
       %  script for testing cp_pfdr_d1_ql1b on tomography problem  %
       %------------------------------------------------------------%
% Reference: H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing
% Nonsmooth Functionals with Graph Total Variation, International Conference on
% Machine Learning, PMLR, 2018, 80, 4244-4253
%
% Hugo Raguet 2017, 2018, 2019
cd(fileparts(which('example_tomography.m')));
addpath('bin/');

%%%  general parameters  %%%
plot_results = true;
print_results = false; % requires color encapsulated postscript driver on your
                      % system

%%%  parameters; see octave/doc/cp_pfdr_d1_ql1b_mex.m %%%
options = struct; % reinitialize
options.cp_dif_tol = 1e-3;
% options.cp_it_max = 10;
options.pfdr_rho = 1.5;
% options.pfdr_cond_min = 1e-3;
% options.pfdr_dif_rcd = 0;
options.pfdr_dif_tol = 1e-1*options.cp_dif_tol;
% options.pfdr_it_max = 1e4;
% options.verbose = 1e3;
% options.max_num_threads = 8;
options.balance_parallel_split = false;

%%%  initialize data  %%%
% Simulated tomography: Shepp-Logan phantom 64x64 with 7 projections;
% TV Graph connectivity is around 3 pixel radius;
% Penalization parameters computed with SURE methods, heuristics adapted from
% H. Raguet: A Signal Processing Approach to Voltage-Sensitive Dye Optical
% Imaging, Ph.D. Thesis, Paris-Dauphine University, 2014
load('../data/tomography.mat')
options.edge_weights = d1_weights;
options.low_bnd = 0.0;
options.upp_bnd = 1.0;

tic;
[Comp, rX] = cp_pfdr_d1_ql1b_mex(y, A, first_edge, adj_vertices, options);
time = toc;
x = rX(Comp + 1); % rX is components values, Comp is components assignment
clear Comp rX;
fprintf('Total MEX execution time %.1f s\n\n', time);

if plot_results %%% plot and print results  %%%
    figure(1), clf, colormap('gray');
    imagesc(x0); axis image; set(gca, 'Xtick', [], 'Ytick', []);
    title('ground truth');
    if print_results
        fprintf('print ground truth... ');
        print(gcf, '-depsc', 'tomography_ground_truth');
        fprintf('done.\n');
    end

    figure(2), clf, colormap('gray');
    imagesc(reshape(x, size(x0))); axis image; set(gca, 'Xtick', [], ...
        'Ytick', []);
    title('reconstruction');
    if print_results
        fprintf('print reconstruction... ');
        print(gcf, '-depsc', 'tomography_reconstruction');
        fprintf('done.\n');
    end

end
