function X_high = apply_laplacian_high_pass_filter(X, S)
    % 计算度矩阵 D
    D = diag(sum(S, 2));
    % 计算拉普拉斯矩阵 L
    L = D - S;
    % 应用高通滤波器
    X_high = L * X;
end
