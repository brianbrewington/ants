allowed=ones(1,NActions);
if (energy(k)<=0.5+2*ENCOST),
  % not enough energy left to communicate.
   allowed(2)=0;
end
if ((onfood(k)==1)|(round(ants(3,k))>=NEnergyStates-3)),
  % no food here, or we have a full tank, so we can't eat.
  allowed(1)=0;
end
allowed=find(allowed);
