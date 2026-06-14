if ((onfood(k)==1)|(round(ants(3,k))>=NEnergyStates-1)),
  % no food here, or we have a full tank, so we can't eat.
  allowed=[2:NActions];
else
  allowed=[1:NActions];
end
