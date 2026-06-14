% PICKACTS Choose actions from a QTable for the artificial ant problem.
%  

energy=round(state1/100);
onfood=rem(state1,10);

for k=1:length(state1),
  j=Qindex(state1(k));
  Qvec=QTable(j,:);
  legal;
  if (length(find(Qvec(allowed)))==0),
    % this state not yet been explored; choose a random action.
    acts(k)=allowed(floor(length(allowed)*rand(1))+1);
  else
    % at least one portion of the state explored; exploit it.
    [ignored_value,indx]=min(Qvec(allowed));
    acts(k)=allowed(indx);
  end
end

