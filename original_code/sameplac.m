function pct=sameplac(ants,food,MASKSIZE,MASK,WorldSize);

% SAMEPLAC returns a vector for updating the state of all ants.
%
%       USAGE:  pct=sameplac(ants,food,MASKSIZE,MASK,WorldSize);

[r,NA]=size(ants);

x=ants(1,:);y=ants(2,:);

m=full(sparse(x,y,ones(1,NA),WorldSize,WorldSize));
a=[m,m,m;m,m,m;m,m,m];
f=full(spones(food).*m);
b=[f,f,f;f,f,f;f,f,f];

x=x+WorldSize;lx=x-MASKSIZE;hx=x+MASKSIZE;
y=y+WorldSize;ly=y-MASKSIZE;hy=y+MASKSIZE;

for k=1:NA,
  numants(k)=sum(sum(MASK.*a(lx(k):hx(k),ly(k):hy(k))));
  numonfood(k)=sum(sum(MASK.*b(lx(k):hx(k),ly(k):hy(k))));
end

pct=numonfood./numants;

