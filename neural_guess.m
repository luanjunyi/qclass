function y_label = neural_guess(X, y, X_query)
    input_layer_size = size(X, 2);
    hidden_layer_size_space = 10:10:100;
    lambda_space = [0.01 0.05 0.1 1 10];
    num_labels = 2;
    
    best_hidden_size = 0;
    best_lambda = 0;
    best_cv_accur = 0;
    best_theta1 = 0;
    best_theta2 = 0;
    
    y = y + 1; % y used to be 0,1, now is 1,2
    
    m = size(X, 1);
    cv_size = floor(m * .3);
    cv_idx = randsample(size(X, 1), cv_size);
    
    X_cv = X(cv_idx, :);
    y_cv = y(cv_idx);
    
    train_idx = setdiff(1:m, cv_idx);
    X_train = X(train_idx, :);
    y_train = y(train_idx);

    for hidden_layer_size = hidden_layer_size_space
        for lambda = lambda_space
            fprintf('training using lambda = %f, hidden size = %d\n', lambda, hidden_layer_size);
            
            initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
            initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
            initial_nn_params = [initial_Theta1(:); initial_Theta2(:)];
            
            costFunction = @(p) nnCostFunction(p, ...
                                               input_layer_size, ...
                                               hidden_layer_size, ...
                                               num_labels, X_train, y_train, lambda);
            options = optimset('MaxIter', 200);
            
            [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
            
            Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                             hidden_layer_size, (input_layer_size + 1));

            Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                             num_labels, (hidden_layer_size + 1));
            
            pred = nn_predict(Theta1, Theta2, X_train);
            fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_train)) * 100);
            
            pred = nn_predict(Theta1, Theta2, X_cv);
            accur = mean(double(pred == y_cv));
            fprintf('Cross validation accuracy: %f\n', accur);
            if accur > best_cv_accur
                best_cv_accur = accur;
                best_lambda = lambda;
                best_hidden_size = hidden_layer_size;
                best_theta1 = Theta1;
                best_theta2 = Theta2;
            end
            
        end
    end

    fprintf('best hidden size: %d, best lambda: %f\n', best_hidden_size, best_lambda);
    hidden_layer_size = best_hidden_size;
    lambda = best_lambda;
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
    initial_nn_params = [initial_Theta1(:); initial_Theta2(:)];
    
    costFunction = @(p) nnCostFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, X, y, lambda);
    options = optimset('MaxIter', 200);
    
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));
    
    pred = nn_predict(Theta1, Theta2, X);
    fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
    

    y_label = nn_predict(Theta1, Theta2, X_query);
    y_label = y_label - 1;
end