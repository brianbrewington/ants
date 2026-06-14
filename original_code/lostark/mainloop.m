
% ----------------------------------------
% MAIN LOOP
% ----------------------------------------

while (~stopped),
  if (paused==0),
    set(StepBox,'string',['Step=',num2str(STEP)]);
    BETA=(1/STEP)^BETAEXP;
    RANDOM=qrand(STEP,RANDOM);
        
    % 1.) Kill off ants with no remaining energy:  if an ant dies, he should be replaced, so the
    %           population can continue to survive.
    
    dead=find(state1<100);
    delete(ants(6,dead));
    ants(:,dead)=[];
    if (min(size(ants))==0),
      alldead;
    end
    [temp,NAnts2]=size(ants);
    
    state1=getstate(ants,NEnergyStates,SAMEPLACE);
    NAnts=NAnts2;

    if (rem(STEP,250)==0),
      set(StatusBar,'string',['Now evaluating STEP number ',num2str(STEP)]);
      save qttemp ants QTable STEP
    end
    
    axes(ANTAX);
    [xfoodtemp,yfood]=find(food);
    if (length(xfoodtemp)~=length(xfood)),
      % a food location was completely consumed; update the display.
      xfood=xfoodtemp;
      delete(FOODLINE);
      FOODLINE=line(xfood,yfood,'erasemode','xor','linestyle','+',...
                'color',[1 1 0],'markersize',4,'visible',animated);
      drawnow;
    end
    xfood=xfoodtemp;
    drawnow;
    
    % 2.) select ant actions, based on the QTable or randomizer.

    acts=zeros(1,length(state1));
    if (RANDOM),
      % choose a random vector of actions.
      randacts;
      set(RandomBox,'string','Using random actions.');
    else
      % use the estimated optimal actions.
      pickacts;
      set(RandomBox,'string','Using optimal actions.');      
    end
    
    % 3.) perform ant actions; record food consumed in "eaten."
    doacts;
    ants(10,:)=sameplac(ants);    
    eaten=eaten+eatentemp;
    foodeaten=[foodeaten,eatentemp];

    % 4.) determine what state transitions occurred, among the live ants.
    state2=getstate(ants,NEnergyStates,SAMEPLACE);
        % NOTE:  since new ants can be created, they will not have experienced
        % "transitions."  Therefore, only the first NAnts columns are treated.
    s1=state1(1:NAnts);s2=state2(1:NAnts);
    transitions=[s1;s2];
    %for mk=1:length(s1),
    %  markov(s1(mk),s2(mk))=markov(s1(mk),s2(mk))+1;
    %end
    [temp,NAnts]=size(ants);

    % 3.) update the Q table
    qupdate;
    
    % 4.) update the food states:  The food replaced equals the food eaten, randomly distributed.
    if (eaten>MaxFoodSize),
      foodtest=rand(1);
      newfood;
    end
    
    % 5.) update the step and states. 
    STEP=STEP+1;
    state1=state2;
  end
end
