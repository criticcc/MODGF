function [score_difference, score_consensus] = model(tSet, Xi, k, method)
    % score_difference n*1
    options = [];
    options.NeighborMode = 'KNN';
    options.WeightMode = 'HeatKernel';
    options.k = k;

    if nargin < 4
        method = 'max'; % 默认使用最大值
    end

    num_view = length(Xi);
    for i = 1:num_view
        %注意，这里由于X是d*n的，根据数据格式来决定要不要转置
        Xi{1,i} = Xi{1,i}';
    end

    n = size(Xi{1}, 1); % Xi{1}的行数, 即数据点数

    % 初始化异常矩阵和邻接矩阵。
    Ei = zeros(n, n, num_view); % 创建一个 n*n*num_view的全零三维矩阵
    Gi = Ei;
    disp('邻接矩阵维度');
    disp(size(Gi));

    % 初始化分数
    score_difference_max = zeros(n, 1);
    score_difference_min = inf(n, 1);
    score_difference_avg = zeros(n, 1);

    for i = 1:num_view
        options.t = tSet(i);
        Gi(:,:,i) = constructW(Xi{1,i}, options); % 构建第 i 个视图的邻接矩阵 Gi(:,:,i)
        num = sum((Gi(:,:,i) > 0), 2) - 1; % 计算每个数据点的邻居数量（不包括自身），存储在向量 num 中

        % 计算对称归一化的拉普拉斯矩阵
        D = diag(sum(Gi(:,:,i), 2));
        D_inv_sqrt = diag(1 ./ sqrt(diag(D) + eps));
        %Gi(:,:,i) = D_inv_sqrt * Gi(:,:,i) * D_inv_sqrt;
        L_sym = D - Gi(:,:,i);
        %L_sym = D_inv_sqrt * (D - Gi(:,:,i)) * D_inv_sqrt;

        % 计算热核滤波器
        H_t = expm(-options.t * L_sym);

        % 计算当前视图中每个节点的分数（行求和）
        current_view_scores = sum(H_t, 2);

        % 在所有视图中取每个节点的最大分数
        score_difference_max = max(score_difference_max, current_view_scores);
        % 在所有视图中取每个节点的最小分数
        score_difference_min = min(score_difference_min, current_view_scores);
        % 在所有视图中取每个节点的平均分数
        score_difference_avg = score_difference_avg + current_view_scores;
    end

    % 计算平均值
    score_difference_avg = score_difference_avg / num_view;

    % 根据方法选择属性异常分数
    switch method
        case 'max'
            score_difference = score_difference_max;
        case 'min'
            score_difference = score_difference_min;
        case 'avg'
            score_difference = score_difference_avg;
        otherwise
            error('Unknown method: %s', method);
    end

    disp('score_difference before normalized:');
    disp(['Min: ', num2str(min(score_difference)), ', Max: ', num2str(max(score_difference)), ', Mean: ', num2str(mean(score_difference)), ', Std: ', num2str(std(score_difference))]);

    % 归一化属性异常得分
    score_difference = (score_difference - min(score_difference)) / (max(score_difference) - min(score_difference));
    disp('score_difference statistics:');
    disp(['Min: ', num2str(min(score_difference)), ', Max: ', num2str(max(score_difference)), ', Mean: ', num2str(mean(score_difference)), ', Std: ', num2str(std(score_difference))]);

    % 对每个视图的邻接矩阵进行对称归一化处理
    for i = 1:num_view
        D = diag(sum(Gi(:,:,i), 2));
        D_inv_sqrt = diag(1 ./ sqrt(diag(D) + eps));
        Gi(:,:,i) = D_inv_sqrt * Gi(:,:,i) * D_inv_sqrt;
    end

    % 计算类别异常得分
    Z = zeros(n, n, num_view * (num_view - 1) / 2);
    idx = 1;
    for u = 1:num_view
        for v = u + 1:num_view
            Z(:,:,idx) = abs(Gi(:,:,u) - Gi(:,:,v));
            idx = idx + 1;
        end
    end

    Z_hat = sum(Z, 3);
    disp('Z statistics:');
    disp(['Min: ', num2str(min(Z(:))), ', Max: ', num2str(max(Z(:))), ', Mean: ', num2str(mean(Z(:))), ', Std: ', num2str(std(Z(:)))]);

    % 使用属性异常得分的结果进行加权
    W = 1 ./ sqrt((score_difference + eps) * (score_difference' + eps));
    disp('W statistics before 设置上限:');
    disp(['Min: ', num2str(min(W(:))), ', Max: ', num2str(max(W(:))), ', Mean: ', num2str(mean(W(:))), ', Std: ', num2str(std(W(:))),',median:',num2str(median(W(:)))]);

    % 设置一个上限，防止权重过大
    W = min(W, 5);
    W = max(W, eps); % 确保权重不为零
    disp('W statistics:');
    disp(['Min: ', num2str(min(W(:))), ', Max: ', num2str(max(W(:))), ', Mean: ', num2str(mean(W(:))), ', Std: ', num2str(std(W(:)))]);

    % 计算一致性得分
    Z_hat = W .* Z_hat;
    disp('Z_hat after weighting statistics:');
    disp(['Min: ', num2str(min(Z_hat(:))), ', Max: ', num2str(max(Z_hat(:))), ', Mean: ', num2str(mean(Z_hat(:))), ', Std: ', num2str(std(Z_hat(:)))]);

    % 计算一致性得分
    score_consensus = sum(Z_hat, 2);

    % 输出调试信息
    disp('score_consensus before normalization statistics:');
    disp(['Min: ', num2str(min(score_consensus)), ', Max: ', num2str(max(score_consensus)), ', Mean: ', num2str(mean(score_consensus)), ', Std: ', num2str(std(score_consensus))]);

    % 归一化一致性得分
    score_consensus = (score_consensus - min(score_consensus)) / (max(score_consensus) - min(score_consensus));

    % 输出调试信息
    disp('score_consensus after normalization statistics:');
    disp(['Min: ', num2str(min(score_consensus)), ', Max: ', num2str(max(score_consensus)), ', Mean: ', num2str(mean(score_consensus)), ', Std: ', num2str(std(score_consensus))]);

    return;
end
