function y_label = logit_guess(X, y, X_query)
  query_num = size(X_query, 1)
  y_label = zeros(query_num, 1);

  [m, n] = size(X);
  X = [ones(m, 1) X];
  initial_theta = zeros(n + 1, 1);
  lambda = 1;
  options = optimset('GradObj', 'on', 'MaxIter', 500000, 'Display','iter');
  fprintf('Logistic regression start training\n');
  fprintf('initial error:%f, press any key to continue...\n', lrCostFunction(initial_theta, X, y, lambda));
  pause;
  [theta, J, exit_flag] = fmincg(@(t)(lrCostFunction(t, X, y, lambda)), initial_theta, options);

  p = predict(theta, X);
  fprintf('Logistic regression train accuracy: %f\n', mean(double(p == y)) * 100);
  X_query = [ones(query_num, 1) X_query];
  y_label = predict(theta, X_query);
end