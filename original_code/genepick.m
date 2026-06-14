% PICKACTS Choose actions from a QTable for the artificial ant problem.
%  

energy=round(state1/100);
onfood=rem(state1,10);

for k=1:length(state1),
  % need to choose an action for each ant in a particular state.
  j=Qindex(state1(k));
  Qvec=QTable(j,:);
  legal;
    % "legal" returns a vector of allowed actions for this ant
    % in the current state. "preact" is a preferred action, which
    % may or may not be legal in that state.
  preact=ants(ATTR+j,k);
  if (sum(allowed==preact)>0),
    % preact is allowed; use it.
    acts(k)=preact;
  else
    % preact is not allowed; don't use it...use DO NOTHING.
    acts(k)=3;
  end
end

