         %-----------------------------------------------------------%
         %  script for illustrating cp_prox_tv on denoising problem  %
         %-----------------------------------------------------------%
% Reference: H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing
% Nonsmooth Functionals with Graph Total Variation, International Conference on
% Machine Learning, PMLR, 2018, 80, 4244-4253
%
% Hugo Raguet 2022
%% needs grid_to_graph and cp_d1_ql1b
cd(fileparts(which('example_prox_tv.m')));
addpath('bin/');

%%%  general parameters  %%%
plot_results = true;
print_results = false;
% image_file = '../data/Paulette.png'
image_file = '../data/SheppLogan_phantom.png'
noise_level = .2
la_tv = 1.5*noise_level
la_tv = 100;

%%%  initialize data  %%%
fprintf('load image and apply transformation... ');
x0 = imread(image_file);
x0 = x0(1:2:end, 1:2:end, 1);
x0 = rescale(x0(:,:,1));
y = x0 + noise_level*randn(size(x0));
fprintf('done.\n');

fprintf('generate adjacency graph... ');
[first_edge, adj_vertices, edge_weights] = grid_to_graph(uint32(size(x0)), 2);
edge_weights = la_tv./sqrt(edge_weights)/4;
fprintf('done.\n');

%% plot the observations
if plot_results
    figure(1), clf, colormap('gray');
    subplot(1, 3, 1), imagesc(x0);
    set(gca, 'xtick', [], 'ytick', []); axis image; title('Original');
    subplot(1, 3, 2), imagesc(max(0, min(y, 1)));
    set(gca, 'xtick', [], 'ytick', []); axis image;
    title(sprintf('Noisy (%.2f dB)',
        10*log10(prod(size(x0))/sum(sum((y - x0).^2)))));
    drawnow('expose');
end
% %}

%%%  solve the optimization problem  %%%
tic;
options.edge_weights = edge_weights;
options.max_num_threads = 0;
[Comp, rX] = cp_prox_tv(y, first_edge, adj_vertices, options);
time = toc;
x = reshape(rX(Comp + 1), size(x0)); % rX is components values, Comp is components assignment
clear Comp rX;
fprintf('Total MEX execution time %.1f s\n\n', time);

%%  plot results
if plot_results
    figure(1)
    subplot(1, 3, 3), imagesc(max(0, min(x, 1)));
    set(gca, 'xtick', [], 'ytick', []); axis image;
    title(sprintf('CP PROX TV (%.2f dB)',
        10*log10(prod(size(x0))/sum(sum((x - x0).^2)))));
    drawnow('expose');
    if print_results
        fprintf('print results... ');
        print(gcf, '-depsc', sprintf('%s/images.eps', experiment_dir));
        fprintf('done.\n');
    end
end
