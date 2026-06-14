% PICKACTS Choose actions from a QTable for the artificial ant problem.
%  

energy=round(state1/1000);
onfood=rem(state1,10);

for k=1:length(state1),
  legal;
  if (ants(RANDA,k)==1),
    acts(k)=allowed(floor(length(allowed)*rand(1))+1);
  else
    j=Qindex(state1(k));
    la=length(allowed);
    Qvec=QTable(j,allowed);
    if (length(find(Qvec))~=la),
      % this state not yet fully explored; choose a random action.
      acts(k)=allowed(floor(la*rand(1))+1);
    else
      [ignored_value,indx]=min(Qvec);
      acts(k)=allowed(indx);
    end
  end
  ants(RANDA,k)=qrand(STEP,ants(RANDA,k));
end

