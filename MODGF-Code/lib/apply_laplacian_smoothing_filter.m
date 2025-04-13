function X_smoothed = apply_laplacian_smoothing_filter(X, G, alpha)
    % 计算度矩阵
    D = diag(sum(G, 2));
    % 计算拉普拉斯矩阵
    L = D - G;
    % 计算拉普拉斯平滑滤波器
    H_alpha = inv(eye(size(L)) + alpha * L);
    % 应用拉普拉斯平滑滤波器
    X_smoothed = H_alpha * X;
end
