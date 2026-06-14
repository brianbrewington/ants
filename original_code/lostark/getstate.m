function state=getstate(ants,NEnergyStates,SAMEPLACE);

% GETSTATE Report the state of all ants.
%
%          USAGE:  state=getstate(ants)
%

%the first digit is just the energy level remaining.  
%Note that the ant can be in "purgatory" (energy=0) for a time before it is removed.
energy=100*round(ants(3,:));
lowenergy=find(energy<0);
highenergy=find(energy>100*NEnergyStates);
energy(lowenergy)=zeros(1,length(lowenergy));
energy(highenergy)=100*(NEnergyStates+1).*ones(1,length(highenergy));

% next, we note how many other ants there are at the same place.
sameplace=10*ants(10,:);
toplevel=find(sameplace>10*SAMEPLACE);
sameplace(toplevel)=10*SAMEPLACE*ones(1,length(toplevel));

% the last digit is the ant's local food sensation (1->not on food, 2->on food)
onfood=ants(5,:);

state=energy+sameplace+onfood;
