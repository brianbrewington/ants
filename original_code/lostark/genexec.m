% ANTEXEC.M
%
% A script for testing different approaches to artificial "agent life":
%       (1) Genetic-type algorithm, where each entity carries its own controller
%       (2) Global Q-learning type controller, from which all entities
%           derive their optimal control policy.
%
% The problem may map nicely to that of agents "living" in a network, looking for
% "food" in the form of information. 

clear ants;
clear markov;
clear Qindex;
clear commlines;

if (animated(1:2)=='of')&(graphical=='on'),
  noplots=text(ANTAX,1,1,'Animation is turned off.');
end
  
R_EXP=0.05;                                          % exponent for randomizing actions

InitAnts=NAnts;                                             
% Currently, the actions are:
%       1.  Eat (not allowed when not at a food site or when an ant has enough energy)
%       2.  Communicate your location to the other ants
%       3.  Do nothing at all (NOP, as it were)
%       4.  Move to the preferred location
%       5.  Listen for the location of the closest broadcasting ant
%       6.  Randomly alter direction, and move a random distance (between 1 and MASKSIZE) in that direction.

ActionIndex= ['Eat       ';
              'Broadcast ';
              'Do Nothing';
              'Teleport  ';
              'Listen    ';
              'RandomMove'];

% array "ants" holds the ant data; each column is an ant and each row is an ant attribute.
% for the "genetic" version, it will also include a control policy for each ant, as this 
% would be part of each ants' state.
% Row 1:  X-grid locations (integers)
% Row 2:  Y-grid locations (also integers)
% Row 3:  Ant energies remaining
% Row 4:  Sensed direction (in radians) to closest, most recent broadcaster.
% Row 5:  Whether (if 2) or not (if 1) the ant is on a food location
% Row 6:  The ant's graphics handle
% Row 7:  Actual X coordinate (real number)
% Row 8:  Actual Y coordinate (real number)
% Row 9:  Distance to desired location along the direction in row 4.
% Row 10: Number of other ants at currently at the same grid location.

StateIndex= ['X-Grid   ';
             'Y-Grid   ';
             'Energy   ';
             'Direction';
             'On Food  ';
             'Handle   ';
             'Real X   ';
             'Real Y   ';
             'Radius   ';
             'SamePlace'];

% array "food" is a sparse matrix indicating the amount of food at each location.
food=makefood(FoodDensity,MaxFoodSize,WorldSize);

% locations and preferences are initially random.
ants(1,:)=floor(WorldSize*rand(1,NAnts))+1;
ants(2,:)=floor(WorldSize*rand(1,NAnts))+1;
ants(3,:)=NEnergyStates*ones(1,NAnts);
ants(4,:)=(2*pi*rand(1,NAnts))-pi*ones(1,NAnts);
ants(5,:)=ones(1,NAnts);
ants(6,:)=zeros(1,NAnts);
ants(7:8,:)=ants(1:2,:);
ants(9,:)=zeros(1,NAnts);
ants(10,:)=sameplac(ants);
ATTR=10;
% the remaining pieces of the ant's state are the control poicies.
ants(ATTR+1:ATTR+1+NTotalStates)=floor(NActions*rand(NTotalStates,NAnts))+1;

for k=1:NAnts,
  if food(ants(1,k),ants(2,k)),
    ants(5,k)=2;
  end
end

% the QTable records the outcomes of the actions as a "cost-to-go" for
%       each action performed from a given state.  We will fill it in
%       as actions are performed.  Initially, it is all zeroes.
Q0=sparse(NTotalStates,NActions);
axes(ANTAX);
set(ANTAX,'xlim',[0 WorldSize+1],'ylim',[0 WorldSize+1]);
cla;
[xfood,yfood]=find(food);
FOODLINE=line(xfood,yfood,'linestyle','+','color',[1 1 0],...
        'erasemode','xor','markersize',4,'visible',animated);
for j=1:NAnts,
  ants(6,j)=line(ants(1,j),ants(2,j),'linestyle','*','color',[1 0 1],...
        'erasemode','xor','markersize',5,'visible',animated);
end
  
DEADDATA=[];
axes(METRAX);
DLINE=line(1,1,'erasemode','normal','visible','on');
set(DLINE,'xdata',[],'ydata',[]);
title('Time to extinction (cycles)');

% The states are enumerated by hashing to decimal numerals. Each number so defined
%       by the function "getstate.m" corresponds to a particular row of the QTable,
%       and this row is found by reading the appropriate entry from Qindex.
%
% We record three pieces of state information for the optimization (there is additional,
%       hidden state that is not explicitly used in the optimization).  The first item is the 
%       energy level, the second is records the desire(1)  to move (or its abscence, 0), and
%       the third is whether the ant is on a food source or not.

% Qindex is a hash table, which maps states to a particular row of the matrix.
Qindex=sparse(1,1);
tempQvec=[1:NTotalStates];
indx=1;
fakeant=[1 1 0 1 1 1 1 1 1 0]';
for k=0:NEnergyStates,
  fakeant(3)=k;
  for j=0:SAMEPLACE,
    fakeant(10)=j;
    for m=1:2,
      fakeant(5)=m;
      Qindex(getstate(fakeant,NEnergyStates,SAMEPLACE),1)=tempQvec(indx);
      indx=indx+1;
    end
  end
end

[qi,qj,qv]=find(Qindex);

state1=getstate(ants,NEnergyStates,SAMEPLACE);
QTable=full(Q0);
mv=max(qi);
markov=sparse(mv,mv);
commlines=[];
foodeaten=[];
STEP=1;
ALLDEAD=1;
RSTEP=1;
RANDOM=1;
eaten=0;
stopped=0;

mainloop;
