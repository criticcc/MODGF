function X_filtered = apply_heat_kernel_filter_symL_with_t(X, G, t)
    % X: 数据矩阵 (d x n)
    % G: 邻接矩阵 (n x n)
    
    % 计算度矩阵 D
    D = diag(sum(G, 2));

    % 计算对称归一化的拉普拉斯矩阵 L_sym
    D_inv_sqrt = diag(1 ./ sqrt(diag(D) + eps));
    L_sym = eye(size(G)) - D_inv_sqrt * G * D_inv_sqrt;

    % 计算热核滤波器 H_t = exp(-tL_sym)
    H_t = expm(-t * L_sym);

    % 应用热核滤波器到数据矩阵 X
    X_filtered = H_t * X;
end
