function plot_spectral_energy_distribution(W, F)
    % 获取数据的维度
    [n, d] = size(F);
    
    % 计算度矩阵 D
    D = diag(sum(W, 2));
    
    % 计算归一化拉普拉斯矩阵 L_norm
    D_inv_sqrt = D^(-0.5);
    L_norm = eye(n) - D_inv_sqrt * W * D_inv_sqrt;
    
    % 特征分解
    [eigvecs, eigvals_matrix] = eig(L_norm);
    eigvals = diag(eigvals_matrix);
    
    % 初始化 SED 矩阵
    SED = zeros(n, d);
    
    % 对每一个图信号进行图傅里叶变换并计算 SED
    for i = 1:d
        % 提取第 i 列的图信号 f
        f = F(:, i);
        
        % 图傅里叶变换
        f_hat = eigvecs' * f;
        
        % 计算谱能量分布 SED
        SED(:, i) = abs(f_hat).^2;
    end
    
    % 计算平均谱能量分布
    mean_SED = mean(SED, 2);
    
    % 绘制平均谱能量分布
    figure;
    bar(eigvals, mean_SED, 'barwidth', 0.5);
    xlabel('\lambda');
    ylabel('SED');
    title('Mean Spectral Energy Distribution');
end
