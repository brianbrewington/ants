function c=actcost(a,bonus);

% ACTCOST Return the cost of an action.
%
%         USAGE:  c=actcost(a,bonus);

if (a==1),
  c=-1*(1+10*bonus);
  if (rand(1)>0.95),disp(['c=',num2str(c)]);end
else
  % all other actions...
  c=0;
end

