% ANT.M
%
% A script for testing different approaches to artificial "agent life":
%       (1) Genetic-type algorithm, where each entity carries its own controller
%       (2) Global Q-learning type controller, from which all entities
%           derive their optimal control policy.
%
% The problem may map nicely to that of agents "living" in a network, looking for
% food.

clear ants;
clear markov;

% ----------------------------------------
% CONSTANT DECLARATIONS
% ----------------------------------------

% these are the numbers of possible actions, states, and initial number of ants
NAnts=5;                                            % initial number of ants.
NEnergyStates=9;                                    % number of nonzero energy states: must be less than 10.
NDirections=4;                                      % number of directions
NActions=9;                                         % number of actions
NTotalStates=(NEnergyStates+1)*(NDirections+1)*2;   % number of states
FoodDensity=0.2;                                    % density of food locations
MaxFoodSize=10;                                     % largest allowable quantity of food at a single location
WorldSize=11;                                       % the grid has sidelength WorldSize
HALFSIZE=floor(WorldSize/2);                        % Each side has length 2*HALFSIZE + 1                                      
MASK=incircle(HALFSIZE);                            % the communications mask screens out ants which are "out of range."
BETAEXP=0.50;                                       % exponent reducing beta: must be in the range (0.5,1]
GAMMA=0.99;                                         % future cost discount factor
ENCOST=0.05;                                        % standard energy deduction per round
R_EXP=0.5;                                          % exponent for randomizing actions

InitAnts=NAnts;                                             
% currently, the actions are:
%       1.  eat (not allowed when not at a food site or when an ant has enough energy)
%       2.  communicate your location to the other ants
%       3.  spontaneously procreate (that is, produce more ants)
%       4.  do nothing at all (NOP, as it were)
%       5.  move in the direction 1
%       6.  move in the direction 2
%       7.  move in the direction 3
%       8.  move in the direction 4
%       9.  listen for incoming communications
ActionIndex= ['Eat       ';
              'Broadcast ';
              'Procreate ';
              'Do Nothing';
              'Move 1    ';
              'Move 2    ';
              'Move 3    ';
              'Move 4    ';
              'Listen    '];

% The states are enumerated by forming 3-digit numerals in which the first digit is the 
%       energy, the second is the desired direction of travel, and the third is whether
%       the ant is on a food source or not.  Direction 0 is reserved for having no desire
%       to move in any direction.  In the QTable, the states appear along the vertical
%       axis in numerical order, and the actions are as enumerated above.

% array "ants" holds the ant data; each column is an ant and each row is an ant attribute.
% for the "genetic" version, it will also include a control policy for each ant, as this 
% would be part of each ants' state.
% Row 1:  X-grid locations
% Row 2:  Y-grid locations
% Row 3:  ant energies remaining
% Row 4:  Preferred directions of travel...real numbers between [-Pi,Pi],
% Row 5:  Whether or not the ant is on a food location
StateIndex= ['X-Grid   ';
             'Y-Grid   ';
             'Energy   ';
             'Direction';
             'On Food  '];

% array "food" is a sparse matrix indicating the amount of food at each location.
food=makefood(FoodDensity,MaxFoodSize,WorldSize);

% locations and preferences are initially random.
ants(1,:)=floor(WorldSize*rand(1,NAnts))+1;
ants(2,:)=floor(WorldSize*rand(1,NAnts))+1;
ants(3,:)=NEnergyStates*ones(1,NAnts);
ants(4,:)=floor((NDirections+1)*rand(1,NAnts));
ants(5,:)=ones(1,NAnts);
for k=1:NAnts,
  if food(ants(1,k),ants(2,k)),
    ants(5,k)=2;
  end
end

% the QTable records the outcomes of the actions as a "cost-to-go" for
%       each action performed from a given state.  We will fill it in
%       as actions are performed.  Initially, it is all zeroes.
Q0=sparse(NTotalStates,NActions);
fig=gcf;
clf;
[x,y]=find(food);
v=plot(ants(1,:),ants(2,:),'m*');
hold on;
fd=plot(x,y,'y+');
set(gca,'YDir','reverse');
axis('image');
hold off;

fig2=figure;
DEADDATA=NEnergyStates/ENCOST;
DPLOT=plot(DEADDATA);
title('Time to extinction (cycles)');

% Qindex is a lookup table, which maps states to a particular row of the matrix.
Qindex=sparse(1000,1);
tempQvec=[1:NTotalStates];
indx=1;
for k=0:NEnergyStates,
  for j=1:NDirections+1,
    for m=1:2,
      Qindex(100*k+10*(j-1)+m,1)=tempQvec(indx);
      indx=indx+1;
    end
  end
end
[qi,qj,qv]=find(Qindex);

state1=getstate(ants);
QTable=full(Q0);
mv=max(qi);
markov=sparse(mv,mv);
STEP=1;
ALLDEAD=1;
RSTEP=1;
RANDOM=1;

% ----------------------------------------
% MAIN LOOP
% ----------------------------------------

while (length(DEADDATA)<250000),
    BETA=(1/STEP)^BETAEXP;
    RANDOM=qrandom(STEP,R_EXP);    
    % 1.) Kill off ants with no remaining energy:  if an ant dies, he should be replaced, so the
    %           population can continue to survive.
    dead=find(state1<100);
    ants(:,dead)=[];
    if (min(size(ants))==0),
      disp('All ants are now dead; restarting population.');
      food=makefood(FoodDensity,MaxFoodSize,WorldSize);
      DEADTIME=STEP-ALLDEAD;
      DEADDATA=[DEADDATA,DEADTIME];
      ALLDEAD=STEP;
      figure(fig2);
      xd=get(DPLOT,'xdata');
      set(DPLOT,'ydata',DEADDATA,'xdata',[xd,length(xd)]);
      NAnts=InitAnts;
      % locations and preferences are initially random.
      ants(1,:)=floor(WorldSize*rand(1,NAnts))+1;
      ants(2,:)=floor(WorldSize*rand(1,NAnts))+1;
      ants(3,:)=NEnergyStates*ones(1,NAnts);
      ants(4,:)=floor((NDirections+1)*rand(1,NAnts));
      ants(5,:)=ones(1,NAnts);
      for k=1:NAnts,
        if food(ants(1,k),ants(2,k)),
          ants(5,k)=2;
        end
      end
    end
    [temp,NAnts2]=size(ants);
    
    state1=getstate(ants);
    NAnts=NAnts2;

    if (rem(STEP,250)==0),
      disp(['Evaluating STEP number ',num2str(STEP)]);
    end
    figure(fig);    
    set(v,'xdata',ants(1,:),'ydata',ants(2,:));
    [x,y]=find(food);
    set(fd,'xdata',x,'ydata',y);
    drawnow;

    % 2.) select ant actions, based on the QTable or randomizer.
    clear acts;
    if (RANDOM),
      % choose a random vector of actions.
      randacts;
    else
      % use the estimated optimal actions.
      pickacts;
    end
    
    % 3.) perform ant actions; record food consumed in "eaten."
    doacts;
    eaten=eaten+eatentemp;

    % 4.) determine what state transitions occurred, among the live ants.
    state2=getstate(ants);
        % NOTE:  since new ants can be created, they will not have experienced
        % "transitions."  Therefore, only the first NAnts columns are treated.
    s1=state1(1:NAnts);s2=state2(1:NAnts);
    transitions=[s1;s2];
    for mk=1:length(s1),
      markov(s1(mk),s2(mk))=markov(s1(mk),s2(mk))+1;
    end
    [temp,NAnts]=size(ants);

    % 3.) update the Q table
    qupdate;
    
    % 4.) update the food states:  The food replaced equals the food eaten, randomly distributed.
    if ((rand(1)<FoodDensity)&(eaten>0)),
      food=newfood(food,eaten,WorldSize,MaxFoodSize);
      eaten=0;
    end
    % 5.) update the step and states. 
    STEP=STEP+1;
    state1=state2;
end
