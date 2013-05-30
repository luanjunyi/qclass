function y_label = neural_guess(X, y, X_query)
  input_layer_size = size(X, 2);
  hidden_layer_size = 25;
  num_labels = 2;
  lambda = 1;
  
  y = y + 1; % y used to be 0,1, now is 1,2

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