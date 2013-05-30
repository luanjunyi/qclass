function [X, y, X_query, X_query_id] =  load_data(input_file_name, force)
  if exist('data.mat', 'file') == 2 & ~force
      fprintf('loading from data.mat\n');
      load('data.mat');
  else
      fprintf('data.mat not found, readling input00.txt, this may take some time\n');
      fd = fopen(input_file_name);
      %fd = fopen('small_input.txt');

      line = fgets(fd);
      line = str2num(line);
      record_num = line(1);
      feature_num = line(2);
      X = [];
      y = [];
      fprintf('recorde num:%d, feature_num:%d\n', record_num, feature_num);
      fprintf('reading training set\n');
      %fflush(stdout);
      for i = 1:record_num
          if mod(i, 100) == 0
              fprintf('%d records read\n', i);
              %fflush(stdout);
          end
          line = fgets(fd);
          line = strsplit(line, ' ');
          yi = str2num(cell2mat(line(2)));
          y = [y; (yi == 1)];
          xi = zeros(1, feature_num);
          line = line(3:end);
          [idx val] = cellfun(@parseCell, line);
          xi(idx) = val;
          X(end + 1, :) = xi;
      end


      line = fgets(fd);
      query_num = str2num(line);
      X_query = [];
      X_query_id = {};

      fprintf('reading %d query set\n', query_num);
      %fflush(stdout);
      for i = 1:query_num
          if mod(i, 100) == 0
              fprintf('%d records read\n', i);
              %fflush(stdout);
          end

          line = fgets(fd);
          line = strsplit(line, ' ');
          xi = zeros(1, feature_num);
          X_query_id(end + 1) = line(1);
          line = line(2:end);
          [idx val] = cellfun(@parseCell, line);
          xi(idx) = val;
          X_query(end + 1, :) = xi;
      end

      fclose(fd);

      fprintf('scaling features\n');
      %fflush(stdout);
      one = ones(size(X, 1), 1);
      X_mean = one * mean(X);
      X_range = one * max(range(X), 0.0000001);
      X = (X - X_mean) ./ X_range;
      one = ones(size(X_query, 1), 1);
      X_mean = one * mean(X_query);
      X_range = one * max(range(X_query), 0.0000001);

      X_query = (X_query - X_mean) ./ X_range;
      
      save('data.mat');
      fprintf('saved variables to data.mat\n');
  end
end