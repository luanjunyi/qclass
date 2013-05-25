function [idx val] = parseCell(cell)
  parts = strsplit(cell, ':');
  idx = str2num(cell2mat(parts(1)));
  val = str2num(cell2mat(parts(2)));
end