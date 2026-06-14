function [ants,BEST]=newants(ants,dead,BEST,ATTR,WorldSize,MASKSIZE,NEnergyStates,NTotalStates,NActions,animated);

% NEWANTS replaces dead ants in the array "ants" with
%         new ones chosen from BEST.
%
%         USAGE:  [ants,BEST]=newants(ants,dead,BEST,ATTR,WorldSize,MASKSIZE,NEnergyStates,NTotalStates,NActions,animated);
%

ND=length(dead);
% no changes necessary if no ants died.
if (ND~=0),
  % for each dead ant, we see if any part of it is worth keeping; update "BEST" accordingly
  [lowage,lowindx]=min(BEST(ATTR,:));
  for j=1:ND,
    k=dead(j);
    age=ants(ATTR,k);
    if (age>lowage),
      % replace ant recorded at lowindx.
      BEST(:,lowindx)=ants(:,k);
      [lowage,lowindx]=min(BEST(ATTR,:));
    end
  end

  % now, having updated the gene pool, introduce new ants.
  ants(1,dead)=floor(WorldSize*rand(1,ND))+1;
  ants(2,dead)=floor(WorldSize*rand(1,ND))+1;
  ants(3,dead)=NEnergyStates*ones(1,ND);
  ants(4,dead)=(2*pi*rand(1,ND))-pi*ones(1,ND);
  ants(5,dead)=ones(1,ND);
  ants(6,dead)=zeros(1,ND);
  ants(7:8,dead)=ants(1:2,dead);
  ants(9,dead)=MASKSIZE*rand(1,ND);
  ants(10,:)=sameplac(ants);
  ants(11,dead)=ones(1,ND);
  % the remaining pieces of the ant's state are the control poicies; these are chosen "genetically."
  ants(ATTR+1:ATTR+NTotalStates,dead)=policy(BEST,ND,ATTR,NActions);

  % also need to draw the new ants.
  for k=1:ND,
    j=dead(k);
    ants(6,j)=line(ants(1,j),ants(2,j),'linestyle','*','color',[1 0 1],...
        'erasemode','xor','markersize',5,'visible',animated);
  end
end

