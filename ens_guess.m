function y_label = ens_guess(X, y, X_query, y_query)
    m = size(X, 1);
    cv_size = floor(m * .3);
    cv_idx = randsample(size(X, 1), cv_size);
    
    X_cv = X(cv_idx, :);
    y_cv = y(cv_idx);
    
    train_idx = setdiff(1:m, cv_idx);
    X_train = X(train_idx, :);
    y_train = y(train_idx);
    
    iter = [2000:500:4000];
    
    best_accur = 0.0;
    best_num = 0;
             
    for num = iter
        fprintf('calling fitensamble with iter = %d\n', num);
        ens = fitensemble(X_train, y_train, 'AdaBoostM1', num, 'Tree');
        
        y_pred = predict(ens, X_cv);
        accu = mean(double(y_pred == y_cv));
        fprintf('cross validation accuracy with num = %d is %f\n', num, accu);
        
        if accu > best_accur
            best_accur = accu;
            best_num = num;
        end
        
        y_pred = predict(ens, X_query);
        accu = mean(double(y_pred == y_query));
        fprintf('testing set accuracy with num = %d is %f\n', num, accu);
        

    end
    
    fprintf('train with num = %d\n', best_num);
    ens = fitensemble(X, y, 'AdaBoostM1', best_num, 'Tree');
    y_label = predict(ens, X_query);
end