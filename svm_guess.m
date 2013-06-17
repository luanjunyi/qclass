function y_label = svm_guess(X, y, X_query, y_query)
    C = 5:1:15;
    gamma = 0.1:0.1:1;
    best_accu = 0;
    best_g = 0;
    best_c = 0;
    best_model = 0;
    best_result = 0;
    for c = C
        for g = gamma
            opt = sprintf('-c %f -g %f -q', c, g);
            model = svmtrain(y, X, opt);
            [y_label, accuracy, prob]= svmpredict(y_query, X_query, model, '-q');
            cur = accuracy(1);
            if cur > best_accu
                fprintf('got better accuracy: %f, c=%f, g=%f\n', cur, c, g);
                best_accu = cur;
                best_g = g;
                best_c = c;
                best_label = y_label;
            end
        end
    end

    fprintf('Training finished, C=%f, g=%f\n', best_c, best_g);
    y_label = best_label;
end