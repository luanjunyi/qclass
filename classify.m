clear ; close all; clc

[X, y, X_query, X_query_id] = load_data('input00.txt', 0);
fprintf('X size:\n');
disp(size(X));
fprintf('y size:\n');
disp(size(y));
fflush(stdout);

% y_result = rand_guess(X, y, X_query);
y_result = logit_guess(X, y, X_query);

% map 0, 1 back to -1 +1
y_result = y_result * 2 - 1;
fout = fopen('out.txt', 'w');
for i = 1:size(X_query_id, 2)
    fprintf(fout, '%s %+d\n', X_query_id{i}, y_result(i));
end
fclose(fout);



