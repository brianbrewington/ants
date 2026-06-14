function [ants,best]=newants(ants,dead,best,ATTR);

% NEWANTS replaces dead ants in the array "ants" with
%         new ones chosen from BEST.
%
%         USAGE:  [ants,best]=newants(ants,dead,best,ATTR);
%

if (length(dead)),
  % for each dead ant, we see if any part of it is worth keeping.
  
  ants(1,:)=floor(WorldSize*rand(1,NAnts))+1;
  ants(2,:)=floor(WorldSize*rand(1,NAnts))+1;
  ants(3,:)=NEnergyStates*ones(1,NAnts);
  ants(4,:)=(2*pi*rand(1,NAnts))-pi*ones(1,NAnts);
  ants(5,:)=ones(1,NAnts);
  ants(6,:)=zeros(1,NAnts);
  ants(7:8,:)=ants(1:2,:);
  ants(9,:)=zeros(1,NAnts);
  ants(10,:)=sameplac(ants);
  ants(11,:)=ones(1,NAnts);
  % the remaining pieces of the ant's state are the control poicies.
  ants(ATTR+1:ATTR+1+NTotalStates)=floor(NActions*rand(NTotalStates,NAnts))+1;
  
end
