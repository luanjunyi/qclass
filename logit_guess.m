function y_label = logit_guess(X, y, X_query, y_query)
    Original_X = X;
    Original_y = y;
    
    X_cv = X(4001:end, :);
    y_cv = y(4001:end);
    X = X(1:4000, :);
    y = y(1:4000);
    
    best = 0;
    best_lambda = 0;
    for lambda = [0.0001:0.0005:0.005];
        y_cv_train = logistic(X, y, X_cv, lambda);
        cur = mean(double(y_cv_train == y_cv));
        if cur > best
            fprintf('got better lambda: %f, accuracy: %f\n', lambda, cur);
            best = cur;
            best_lambda = lambda;
        end
    end
    %     
    y_label = logistic(Original_X, Original_y, X_query, best_lambda);
    fprintf('best lambda: %f, test accuracy: %f\n', best_lambda, mean(double(y_label == y_query)));
end

function y_label = logistic(X, y, X_query, lambda)
    query_num = size(X_query, 1);
    y_label = zeros(query_num, 1);

    [m, n] = size(X);
    X = [ones(m, 1) X];
    initial_theta = zeros(n + 1, 1);

    options = optimset('GradObj', 'on', 'MaxIter', 5000, 'Display','iter');
    %fprintf('Logistic regression start training\n');
    %fprintf('initial error:%f, press any key to continue...\n', lrCostFunction(initial_theta, X, y, lambda));
    [theta, J, exit_flag] = fmincg(@(t)(lrCostFunction(t, X, y, lambda)), initial_theta, options);

    p = predict(theta, X);
    %fprintf('Logistic regression train accuracy: %f\n', mean(double(p == y)) * 100);
    X_query = [ones(query_num, 1) X_query];
    y_label = predict(theta, X_query);
    
end
    
    