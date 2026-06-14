function c=actcost(a,bonus);

% ACTCOST Return the cost of an action.
%
%         USAGE:  c=actcost(a,bonus);

if (a==1),
  c=-1*(2+bonus);
else
  % all other actions...
  c=0;
end

