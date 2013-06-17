function y = load_cv_label(input_file_name, force)
  if exist('cv_label.mat', 'file') == 2 && ~force
      fprintf('loading cross validation from cv_label.mat\n');
      load('cv_label.mat');
  else
      fprintf('cv_label.mat not found, readling output00.txt, this may take some time\n');
      fd = fopen(input_file_name);
      y = [];
      line = fgets(fd);
      while line ~= -1
          line = strsplit(line, ' ');
          yi = str2num(cell2mat(line(2)));
          y = [y; yi];
          line = fgets(fd);          
      end
      
      save('cv_label.mat');
      fprintf('saved variables to cv_label.mat\n');
end