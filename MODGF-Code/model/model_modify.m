function [score_difference, score_consensus] = model_modify(Xi, k, K, lambda, tSet,gamma)
    num_view = length(Xi);  

    for v = 1:num_view
        Xi{v} = Xi{v}';
    end

    n = size(Xi{1}, 1);  
    P = cell(num_view, K);  

    % Construct KNN matrix S_v
    for v = 1:num_view
        options = [];
        options.NeighborMode = 'KNN';
        options.k = k;
        options.WeightMode = 'HeatKernel';
        options.t = tSet(v);
        options.bSelfConnected = 1;

        S_v = constructW(Xi{v}, options); 
        D_v = diag(sum(S_v, 2));           
        P_v = 0.5 * (eye(n) + D_v^(-0.5) * S_v * D_v^(-0.5)); 

      
         P{v, 1} = P_v;
         P{v, 1} = P_v;  
        for k_idx = 2:K
            P{v, k_idx} = P{v, k_idx - 1} * P_v;  
        end
    end
    for v = 1:num_view
        for k_idx = 1:K
            P_norm = norm(P{v, k_idx}, 'fro');
            if P_norm > 0
                P{v, k_idx} = P{v, k_idx} / P_norm;
            end
        end
    end
    W_opt = ones(K, num_view) / K; 

    W_opt = W_opt ./ sum(W_opt, 1);
    disp(W_opt);

    max_iter = 10;  
    tol = 1e-6;  
    prev_obj_value = inf; 

    % Alternating optimization
    for iter = 1:max_iter
         obj_value = 0; 
%         Z_v = cell(num_view, 1);
%         Q_vu = cell(num_view, num_view);
% 
%         for v = 1:num_view
%             Xv = Xi{v};
%             Gv = zeros(n, n);
%             for k_idx = 1:K
%                 Gv = Gv + W_opt(k_idx, v) * P{v, k_idx};
%             end
% 
%             % Calculate δ_v^2
%             delta_v2 = mean(sum((Xv - Gv * Xv).^2, 2));
% 
%             % Update Z_v diagonal matrix
%             Z_v{v} = diag(2 ./ delta_v2 .* exp(-sum((Xv - Gv * Xv).^2, 2) ./ delta_v2));
%         end
% 
%         for v = 1:num_view
%             for u = v+1:num_view
%                 Gv = zeros(n, n);
%                 Gu = zeros(n, n);
% 
%                 for k_idx = 1:K
%                     Gv = Gv + W_opt(k_idx, v) * P{v, k_idx};
%                     Gu = Gu + W_opt(k_idx, u) * P{u, k_idx};
%                 end
% 
%                 % Calculate δ_vu^2
%                 delta_vu2 = mean(sum((Gv - Gu).^2, 2));
% 
%                 % Update Q_vu diagonal matrix
%                 Q_vu{v, u} = diag(2 ./ (delta_vu2 + eps) .* exp(-sum((Gv - Gu).^2, 2) ./ (delta_vu2 + eps)));
% 
%                 Q_vu{u, v} = Q_vu{v, u};  % Symmetric assignment
%             end
%         end

        for v = 1:num_view
            H_v = zeros(K);
            f_v = zeros(K, 1);
            Xv = Xi{v};
            for k_idx = 1:K
                Gv_k = P{v, k_idx} * Xv;  
                for l = 1:K
                    tmp = P{v, l} * Xv;
                    H_v(k_idx, l) = H_v(k_idx, l) + 2*sum(sum(Gv_k .* (Z_v{v} * tmp)));
                end
                tmp = P{v, k_idx} * Xv;
                
                f_v(k_idx) = -2 * sum(sum(Xv .*  tmp));
               
            end
            
            % Construct H matrix and f vector for category anomaly part
            for u = 1:num_view

                if u ~= v
                    % Calculate Gu
                    Gu = zeros(n, n);
                    for k_idx = 1:K
                        Gu = Gu + W_opt(k_idx, u) * P{u, k_idx};
                    end
                    
                    for k_idx = 1:K
                        Pu_Pv_diff_norm = norm( P{v, k_idx} - Gu, 'fro');
                        tmp1 = P{v, k_idx};
                        term_2 = sum(sum(tmp1'.* Gu)); 
                        f_v(k_idx) = f_v(k_idx) -  2*lambda * term_2;
                        for l = 1:K
                            tmp2 = P{v, l};
                            Q_vu{v, u} = eye(n);
                            term_1 = sum(sum(tmp1 .* tmp2));
                                                                                  
                            H_v(k_idx, l) = H_v(k_idx, l) + 2*lambda * term_1;
                       
                        end                    
                    end
                end
            end
            disp(f_v);

            H_v = (H_v + H_v') / 2;

           
            A_eq = ones(1, K);
            b_eq = 1;
            lb = zeros(K, 1);

            W_init = W_opt(:, v);
            initial_obj_value = 0.5 * W_init' * H_v * W_init + f_v' * W_init;

            [W_v_opt, fval] = quadprog(H_v, f_v, [], [], A_eq, b_eq, lb, [], [], optimoptions('quadprog', 'Display', 'none'));

            W_opt(:, v) = W_v_opt;

            obj_value = obj_value + fval;

        end
            % Check if the change in objective function is less than the threshold
            if abs(prev_obj_value - obj_value) < tol
                fprintf('Optimization converged after %d iterations.\n', iter);
                break;
            end

            prev_obj_value = obj_value;  % Update previous objective function value

    end

    % Calculate attribute anomaly score
    score_difference = zeros(n, 1);
    for v = 1:num_view
        Xv = Xi{v};
        Gv = zeros(n, n);
        for k_idx = 1:K
            Gv = Gv + W_opt(k_idx, v) * P{v, k_idx};
        end
        score_difference = max(score_difference, sqrt(sum((Xv - Gv * Xv).^2, 2)));
    end

   
    score_consensus = zeros(n, 1);
    epsilon =0.1;  
    for v = 1:num_view
        for u = v+1:num_view
            Gv = zeros(n, n);
            Gu = zeros(n, n);

            for k_idx = 1:K
                Gv = Gv + W_opt(k_idx, v) * P{v, k_idx};
                Gu = Gu + W_opt(k_idx, u) * P{u, k_idx};
            end

            
            W_ij = 1 ./ sqrt(score_difference .* score_difference' + epsilon);

            M_diff = (Gv - Gu);
            consensus_score_temp = sqrt(sum(W_ij .* (M_diff .^ 2), 2));
            score_consensus = max(score_consensus, consensus_score_temp);
        end
    end

    % Normalize scores
    score_difference = (score_difference - min(score_difference)) / (max(score_difference) - min(score_difference));
    score_consensus = (score_consensus - min(score_consensus)) / (max(score_consensus) - min(score_consensus));

end