function pct=sameplac(ants,food,MASKSIZE,MASK,WorldSize);

% SAMEPLAC returns a vector for updating the state of all ants.
%
%       USAGE:  pct=sameplac(ants,food,MASKSIZE,WorldSize);

sz=max(max(ants(1:2,:)));
[r,NA]=size(ants);
m=sparse(sz,sz);

x=ants(1,:);y=ants(2,:);
m=full(sparse(x,y,ones(1,NA),WorldSize,WorldSize));
a=[m,m,m;m,m,m;m,m,m];
f=full(spones(food).*m);
b=[f,f,f;f,f,f;f,f,f];

for k=1:NA,
  % screen region of "a" with MASK.
  numants=sum(MASK.*
end










l=3*WorldSize+2*MASKSIZE;
s=2^nextpow2(l);
mft=fft2(MASK,s,s);
aft=fft2(a,s,s);
bft=fft2(b,s,s);
LocalAnts=real(ifft2(mft.*aft));
LocalFood=real(ifft2(bft.*mft));

LocalAnts=LocalAnts(1:l,1:l);
LocalFood=LocalFood(1:l,1:l);

offset=MASKSIZE+WorldSize;
LocalFood=LocalFood(offset+1:offset+WorldSize,offset+1:offset+WorldSize);
LocalAnts=LocalAnts(offset+1:offset+WorldSize,offset+1:offset+WorldSize);

for k=1:NA,
  pct(k)=LocalFood(x(k),y(k))./LocalAnts(x(k),y(k));
end

ltz=find(pct<0);
pct(ltz)=zeros(size(ltz));

