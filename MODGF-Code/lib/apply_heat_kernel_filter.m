function X_filtered = apply_heat_kernel_filter(X, G, method)
    % X: 数据矩阵 (d x n)
    % G: 邻接矩阵 (n x n)
    % method: 'median' 或 'mean'，用于选择时间参数 t 的计算方法
    
    % 检查输入参数
    if nargin < 3
        method = 'median';
    end

    % 计算度矩阵 D
    D = diag(sum(G, 2));

    % 计算图拉普拉斯矩阵 L
    L = D - G;

    % 选择时间参数 t
    if strcmp(method, 'median')
        t = choose_t_median_distance(X');
    elseif strcmp(method, 'mean')
        t = choose_t_mean_distance(X');
    else
        error('Invalid method. Choose either "median" or "mean".');
    end

    % 计算热核滤波器 H_t = exp(-tL)
    H_t = expm(-t * L);

    % 输出 H_t 和 X 的形状
%     disp('Shape of H_t:');
%     disp(size(H_t));
%     disp('Shape of X:');
%     disp(size(X));

    % 应用热核滤波器到数据矩阵 X
    X_filtered = H_t * X;
    
end


function t = choose_t_median_distance(X)
    % 计算所有数据点对之间的距离
    D = EuDist2(X, X, 0); % 使用自定义的 EuDist2 函数，计算距离的平方
    
    % 计算距离的中位数
    t = median(D(:));
end

function t = choose_t_mean_distance(X)
    % 计算所有数据点对之间的距离
    D = EuDist2(X, X, 0); % 使用自定义的 EuDist2 函数，计算距离的平方
    
    % 计算距离的均值
    t = mean(D(:));
end
