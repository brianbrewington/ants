function state=getstate(ants,NEnergyStates,SAMEPLACE);

% GETSTATE Report the state of all ants.
%
%          USAGE:  state=getstate(ants,NEnergyStates,SAMEPLACE);
%

%the first digit is just the energy level remaining.  
%Note that the ant can be in "purgatory" (energy=0) for a time before it is removed.
energy=1000*round(ants(3,:));
lowenergy=find(energy<0);
highenergy=find(energy>1000*NEnergyStates);
energy(lowenergy)=zeros(1,length(lowenergy));
energy(highenergy)=1000*(NEnergyStates+1).*ones(1,length(highenergy));

% next, we note how many other ants there are at the same place.

sameplace=100*floor((SAMEPLACE+1)*ants(10,:));
highval=find(sameplace>=100*(SAMEPLACE+1));
sameplace(highval)=100*SAMEPLACE*ones(size(highval));

% next, we note whether the last communication was received from a food site or not.
tofood=10*ants(11,:);

% the last digit is the ant's local food sensation (1->not on food, 2->on food)
onfood=ants(5,:);

state=energy+sameplace+tofood+onfood;
