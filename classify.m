clear ; close all; clc

fd = fopen('input00.txt');

line = fgets(fd);
line = str2num(line);
record_num = line(1);
feature_num = line(2);
fprintf('recorde num:%d, feature_num:%d\n', record_num, feature_num);

X = [];
y = [];

for i = 1:record_num
   line = fgets(fd);
   line = strsplit(line);
   yi = str2num(cell2mat(line(2)));
   y(end + 1) = (yi == 1);
   xi = zeros(1, feature_num);
   line = line(3:end);
   [idx val] = cellfun(@parseCell, line);
   xi(idx) = val;
   X(end + 1, :) = xi;
end

X;
y;