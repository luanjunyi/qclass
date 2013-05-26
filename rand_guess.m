function labels = parseCell(X, y, X_query)
  query_num = size(X_query, 1)
  labels = (rand(query_num, 1) > 0.5);
end