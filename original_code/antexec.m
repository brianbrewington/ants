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
clear Qindex;
clear commlines;

R_EXP=0.05;   % exponent for randomizing actions

InitAnts=NAnts;                                             
% Currently, the actions are:
%       1.  Eat (not allowed when not at a food site or when an ant has enough energy)
%       2.  Communicate your location to the other ants
%       3.  Do nothing at all (NOP, as it were)
%       4.  Move to the preferred location -- (x,y) coordinates.
%       5.  Listen for the location of the closest broadcasting ant
%       6.  Randomly alter direction, and move a random distance (between 1 and MASKSIZE) in that direction.

ActionIndex= ['Eat       ';
              'Broadcast ';
              'Do Nothing';
              'Teleport  ';
              'Listen    ';
              'RandomMove';
              '(NO DATA) '];
              

% array "ants" holds the ant data; each column is an ant and each row is an ant attribute.
% Row 1:  X-grid locations (integers)
% Row 2:  Y-grid locations (also integers)
% Row 3:  Ant energies remaining
% Row 4:  Sensed x-coordinate of the closest, most recent broadcaster.
% Row 5:  Whether (if 2) or not (if 1) the ant is on a food location
% Row 6:  The ant's graphics handle
% Row 7:  Actual X coordinate (real number)
% Row 8:  Actual Y coordinate (real number)
% Row 9:  Sensed y-coordinate corresponding to that in row 4.
% Row 10: Degree to which ants are located on food in local communication area, quantized into SAMEPLACE bins.
% Row 11: Whether the last communication received was from an ant on a food site
% Row 12: Randomized actions (1) or learned actions (0)

% in the interest of attempting to be "object-oriented," array references will be made using the following 5-character constants:
XGRID=1;YGRID=2;
ENERG=3;
XDEST=4;YDEST=9;
ONFOO=5;
HANDL=6;
REALX=7;
REALY=8;
SAMEP=10;
TOFOO=11;
RANDA=12;

StateIndex= ['X-Grid    ';
             'Y-Grid    ';
             'Energy    ';
             'X-dest    ';
             'On Food   ';
             'Handle    ';
             'Real X    ';
             'Real Y    ';
             'Y-dest    ';
             'Conscience';
             'To Food   ';
             'Random Act'];

% array "food" is a sparse matrix indicating the amount of food at each location.
food=makefood(FoodDensity,MaxFoodSize,WorldSize);

MASK=full(incircle(MASKSIZE));

% locations and preferences are initially random.
ants(XGRID,:)=floor(WorldSize*rand(1,NAnts))+1;
ants(YGRID,:)=floor(WorldSize*rand(1,NAnts))+1;
ants(ENERG,:)=NEnergyStates*ones(1,NAnts);
ants(XDEST,:)=floor(WorldSize*rand(1,NAnts))+1;
ants(ONFOO,:)=ones(1,NAnts);
ants(HANDL,:)=zeros(1,NAnts);
ants(REALX:REALY,:)=ants(1:2,:);
ants(YDEST,:)=floor(WorldSize*rand(1,NAnts))+1;
ants(SAMEP,:)=sameplac(ants,food,MASKSIZE,MASK,WorldSize);
ants(TOFOO,:)=ones(1,NAnts);
ants(RANDA,:)=ones(1,NAnts);    % all ants intialized to act randomly at first.

for k=1:NAnts,
  if food(ants(XGRID,k),ants(YGRID,k)),
    ants(ONFOO,k)=2;
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
  ants(HANDL,j)=line(ants(XGRID,j),ants(YGRID,j),'linestyle','*','color',[1 0 1],...
        'erasemode','xor','markersize',5,'visible',animated);
end
  
DEADDATA=[];
CODATA=[];
OPTDATA=[];
FOODDATA=[];
DEADTIME=[];
dd=0;
colocation=0;
optimal=0;
foodeaten=0;

axes(METRAX);
DLINE=line(1,1,'erasemode','normal','visible','on','color',[0 0 1]);
set(DLINE,'xdata',[],'ydata',[]);
CLINE=line(1,1,'erasemode','normal','visible','on','color',[0 1 0]);
set(CLINE,'xdata',[],'ydata',[]);
OLINE=line(1,1,'erasemode','normal','visible','on','color',[1 0 0]);
set(OLINE,'xdata',[],'ydata',[]);
FLINE=line(1,1,'erasemode','normal','visible','on','color',[1 1 0]);
set(FLINE,'xdata',[],'ydata',[]);


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
fakeant=[1;
         1;
         0;
         1;
         1;
         1;
         1;
         1;
         1;
         0;
         1];

% the following FOR-loops move through all possible states and assign each a position in the hash table.
for k=0:NEnergyStates,
  fakeant(3)=k;
  for j=0:SAMEPLACE,
    fakeant(10)=(j/(SAMEPLACE+1));
    for n=1:2,
      fakeant(11)=n;
        for m=1:2,
        fakeant(5)=m;
        Qindex(initstat(fakeant,NEnergyStates,SAMEPLACE),1)=tempQvec(indx);
        indx=indx+1;
      end
    end
  end
end

[qi,qj,qv]=find(Qindex);

state1=getstate(ants,NEnergyStates,SAMEPLACE);
QTable=full(Q0);
mv=max(qi);
commlines=[];
STEP=1;
ALLDEAD=1;
RSTEP=1;
RANDOM=1;
eaten=0;
stopped=0;

mainloop;
