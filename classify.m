clear ; close all; clc

[X, y, X_query, X_query_id] = load_data('input00.txt', 0);
y_query = load_cv_label('output00.txt', 0);

fprintf('X size:\n');
disp(size(X));
fprintf('y size:\n');
disp(size(y));
%fflush(stdout);

% y_result = rand_guess(X, y, X_query);
y_result = neural_guess(X, y, X_query);
% y_result = svm_guess(X, y, X_query, (y_query + 1) / 2);
% y_result = logit_guess(X, y, X_query, (y_query + 1) / 2);
% y_result = ens_guess(X, y, X_query, (y_query + 1) / 2);

% map 0, 1 back to -1 +1
y_result = y_result * 2 - 1;
accuracy = mean(double(y_result == y_query)) * 100;
fprintf('test set accuracy:%f%%\n', accuracy);