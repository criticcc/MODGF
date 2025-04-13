clear;
clc;

addpath(fullfile(pwd, 'model'));
addpath(fullfile(pwd, 'lib'));
addpath(fullfile(pwd, 'data'));

data_folder = fullfile(pwd, 'data');
result_folder = fullfile(pwd, 'result');

if ~exist(result_folder, 'dir')
    mkdir(result_folder);
end

data_files = dir(fullfile(data_folder, '*.mat'));

k_values = [5,15,25,35,45,55,65];  % number of neighbors
K = 3 % graph filter order
gamma_values = logspace(-5, 5, 11);

alpha_values = 0.1:0.1:0.9;  

profile on;
for file = data_files'
    fprintf('Processing: %s\n', file.name);
    
    load(fullfile(data_folder, file.name));
    num = length(X);

    auc_matrix = zeros(length(k_values), length(gamma_values), length(alpha_values));
    auc_difference_avg = zeros(length(k_values), length(gamma_values));
    auc_consensus_avg = zeros(length(k_values), length(gamma_values));

    for k_idx = 1:length(k_values)
        k = k_values(k_idx);
        fprintf('k: %d\n', k);
        
        for gamma_idx = 1:length(gamma_values)
            gamma = gamma_values(gamma_idx);
            fprintf('gamma: %.1e\n', gamma);
            
            auc_difference = zeros(num, 1);
            auc_consensus = zeros(num, 1);
            
            % Run 5 times
            for i = 1:num
                Xs = X{i};
                gnd = out_label{i};
                numview = length(Xs);
                
                tSet = zeros(numview, 1);
                for j = 1:numview
                    Xs{j} = zscore(Xs{j});
                    Ds{1, j} = EuDist2(Xs{1, j}', Xs{1, j}');
                    tSet(j) = max(median(Ds{1, j}, 'all'), eps);
                end

                [score_difference, score_consensus] = model_modify(Xs, k, K, gamma, tSet, gamma);

                for alpha_idx = 1:length(alpha_values)
                    alpha = alpha_values(alpha_idx);
                    combined_score = alpha * score_difference + (1 - alpha) * score_consensus;
                    [~, ~, ~, auc_combined] = perfcurve(gnd, combined_score, 1);
                    auc_matrix(k_idx, gamma_idx, alpha_idx) = auc_matrix(k_idx, gamma_idx, alpha_idx) + auc_combined / num; % Normalize AUC sum
                end
            end
            % Calculate average AUC
            auc_difference_avg(k_idx, gamma_idx) = mean(auc_difference);
            auc_consensus_avg(k_idx, gamma_idx) = mean(auc_consensus);
        end
    end

    % Find best K, gamma, and alpha
    [best_auc, best_idx] = max(auc_matrix(:));
    [best_k_idx, best_gamma_idx, best_alpha_idx] = ind2sub(size(auc_matrix), best_idx);
    best_k = k_values(best_k_idx);
    best_gamma = gamma_values(best_gamma_idx);
    best_alpha = alpha_values(best_alpha_idx);

    best_result = [best_k, best_gamma, best_alpha, auc_difference_avg(best_k_idx, best_gamma_idx), auc_consensus_avg(best_k_idx, best_gamma_idx), best_auc];

    result_matrix = auc_matrix;

    % Save the result
    result_filename = fullfile(result_folder, strrep(file.name, '.mat', '_result.mat'));
    save(result_filename, 'best_result', 'result_matrix');

    fprintf('Dataset file %s processed.\n', file.name);
    fprintf('Best parameters: K = %d, gamma = %.1e, Alpha = %.1f, Best AUC = %.5f\n', best_k, best_gamma, best_alpha, best_auc);
end
profile off;
profile viewer;
fprintf('All datasets processed.\n');