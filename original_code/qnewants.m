function ants=qnewants(ants,dead,food,WorldSize,MASKSIZE,NEnergyStates,NTotalStates,NActions,animated);

% QNEWANTS replaces dead ants in the array "ants" with new ones chosen at random.
%
%         USAGE:  ants=qnewants(ants,dead,food,WorldSize,MASKSIZE,NEnergyStates,NTotalStates,NActions,animated);
%

ND=length(dead);
MASK=incircle(MASKSIZE);
% introduce new ants.
ants(1,dead)=floor(WorldSize*rand(1,ND))+1;
ants(2,dead)=floor(WorldSize*rand(1,ND))+1;
ants(3,dead)=NEnergyStates*ones(1,ND);
ants(4,dead)=floor(WorldSize*rand(1,ND))+1;
ants(5,dead)=ones(1,ND);
ants(6,dead)=zeros(1,ND);
ants(7:8,dead)=ants(1:2,dead);
ants(9,dead)=floor(WorldSize*rand(1,ND))+1;
ants(10,dead)=sameplac(ants(:,dead),food,MASKSIZE,MASK,WorldSize);
ants(11,dead)=ones(1,ND);

for k=1:ND,
  % need to draw the new ants.
  j=dead(k);
  ants(6,j)=line(ants(1,j),ants(2,j),'linestyle','*','color',[1 0 1],...
        'erasemode','xor','markersize',5,'visible',animated);
  % check to see if new ants are on food.
  if food(ants(1,j),ants(2,j)),
    ants(5,j)=2;
  end  
end





