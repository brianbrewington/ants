function new=quantize(direct,NDirections);

% QUANTIZE Break directions (in radians) into quantized direction numbers.
%
%          USAGE:  new=quantize(direct,NDirections);

pr=pi/NDirections;
step=2*pi/NDirections;
idx=find(direct>=(NDirections-1)*pr);
direct(idx)=direct(idx)-2*pi;

for k=0:NDirections-1,
  test=(direct<=(step*k)+(-NDirections+1)*pr);
  found=find(test);
  if (~isempty(found)),
    % at least one fit the bill...assign it!
    new(found)=ones(size(found))*(k+1);
    direct(found)=101*ones(size(found));
    % we set the value to 101 so that it will not be found the next time around.
  end
end


