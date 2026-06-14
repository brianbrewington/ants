function v=sameplac(ants);

% SAMEPLAC returns a vector for updating the state of all ants.
%
%       USAGE:  v=sameplac(ants);

sz=max(max(ants(1:2,:)));
[r,NA]=size(ants);
m=sparse(sz,sz);
v=zeros(1,NA);

for i=1:NA,
  x=ants(1,i);
  y=ants(2,i);
  for j=1:NA,
    if (x==ants(1,j))&(y==ants(2,j)),
      % another one here.
      v(i)=v(i)+1;
    end
  end
end
v=v-1;
