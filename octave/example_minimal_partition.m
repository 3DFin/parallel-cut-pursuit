  %------------------------------------------------------------------------%
  %  script for illustrating cp_kmpp_d0_dist on minimal partition problem  %
  %------------------------------------------------------------------------%
% L. Landrieu and G. Obozinski, Cut Pursuit: fast algorithms to learn
% piecewise constant functions on general weighted graphs, SIAM Journal on
% Imaging Science, 10(4):1724-1766, 2017
%
% Hugo Raguet 2019
cd(fileparts(which('example_minimal_partition.m')));
addpath('bin/');

%%%  parameters; see octave/doc/cp_kmpp_d0_dist_mex.m  %%%
plot_graphs = true;
loss = 1;
options = struct; % reinitialize
% options.cp_dif_tol = 1e-3;
% options.cp_it_max = 10;
% options.K = 2;
% options.split_iter_num = 2;
% options.kmpp_init_num = 3;
% options.kmpp_iter_num = 3;
% options.verbose = true;

% %{
%%%  initialize data  %%%
V = 1e3;
E = 10*V;
knn = 3;
edges_method = 'knn'; % 'knn' | 'delaunay' | 'random'

% embedding on the plane and connectivity
% space_coor = rand(V, 2);
[coorX, coorY] = meshgrid(0:(ceil(sqrt(V)) - 1));
space_coor = [coorX(1:V)' coorY(1:V)']/(ceil(sqrt(V)) - 1);
clear coorX coorY;
switch edges_method
case 'delaunay'
    fprintf('get delaunay triangulation... '); drawnow;
    TRI = delaunay(space_coor);
    Euv = [TRI(:,1), TRI(:,2); TRI(:,2), TRI(:,3)];
case 'knn'
    knn = min(knn, V-1);
    fprintf('get %d-nn graph... ', knn); drawnow;
    Euv = knnGraph(space_coor, knn);
case 'random'
    fprintf('draw %d random edges... ', E); drawnow;
    Euv = ceil(V*rand(E,2));
    Euv(Euv(:,1)==Euv(:,2),:) = [];
end
fprintf('done.\n');

% sort unique versions of edges
Euv = sort(Euv, 2);
Euv = unique(Euv, 'rows');
Eu = Euv(:,1); Ev = Euv(:,2); clear Euv;
E = size(Eu, 1);

% options.edge_weights = []; % set edge weights uniformly equal to one
options.edge_weights = 1; % set edge weights homogeneous
% options.edge_weights = rand(2*E,1); % set edge weights at random

options.vert_weights = []; % set vertex weights uniformly equal to one
% options.vert_weights = rand(V, 1);

% signal
Y = 100*rand(1, V);
% Y = 1:V;

% convert adjacency to forward star representation
% [first_edge, reindex] = adjacency_to_forward_star_mex(V, Eu);
[first_edge, reindex] = graph_adjacency_to_forward_star(V, Eu);
adj_vertices = Ev(reindex); clear Ev;
if numel(options.edge_weights) > 1
    options.edge_weights = options.edge_weights(reindex);
end
clear reindex

% convert to correct format
first_edge = uint32(first_edge - 1);
adj_vertices = uint32(adj_vertices - 1);

if plot_graphs && V < 1e4
    fprintf('plot resulting graph... '); drawnow
    % check resulting graph
    figure(1); clf;
    plot_graph(space_coor, first_edge + 1, adj_vertices + 1, ...
        options.edge_weights, options.vert_weights, Y);
    colorbar;
    fprintf('done.\n');
end
%}

%%%  solve the optimization problem  %%%
tic;
coor_weights = [];
[Comp, rX, it, Obj, Time, Dif] = cp_kmpp_d0_dist_mex(loss, Y, first_edge, ...
    adj_vertices, options);
time = toc;
X = rX(:, Comp + 1); % rX is components values, Comp is components assignement
% clear Comp rX;
fprintf('Total MEX execution time %.0f s\n\n', time);

if plot_graphs && V < 1e4
    fprintf('plot resulting graph... '); drawnow
    % check resulting graph
    figure(2); clf;
    plot_graph(space_coor, first_edge + 1, adj_vertices + 1, ...
        options.edge_weights, options.vert_weights, X);
    colorbar;
    fprintf('done.\n');
end
