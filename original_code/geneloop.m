
% ----------------------------------------
% MAIN LOOP (GENETIC VERSION)
% ----------------------------------------

while (~stopped),
  if (paused==0),
    set(StepBox,'string',['Step=',num2str(STEP)]);
    BETA=(1/STEP)^BETAEXP;
    RANDOM=qrand(STEP,RANDOM);
        
    % 1.) Kill off ants with no remaining energy:  if an ant dies, he should be replaced, so the
    %           population can continue to survive.
    
    dead=find(state1<100);
    DEADDATA=[DEADDATA;length(dead)];
    delete(ants(6,dead));       %just deletes the graphics, not the record of the ants.
    [ants,BEST]=newants(ants,dead,BEST,ATTR,WorldSize,MASKSIZE,NEnergyStates,NTotalStates,NActions,animated);
    state1=getstate(ants,NEnergyStates,SAMEPLACE);

    if (rem(STEP,250)==0),
      set(StatusBar,'string',['Now evaluating STEP number ',num2str(STEP)]);
      save qttemp ants BEST STEP
      axes(METRAX);
      set(DLINE,'ydata',DEADDATA,'xdata',[1:length(DEADDATA)]);
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
    genepick;
    geneacts;
    set(RandomBox,'string','Using genetic actions.');
    
    % 3.) perform ant actions; record food consumed in "eaten."
    geneacts;
    ants(10,:)=sameplac(ants,food,MASKSIZE);    
    eaten=eaten+eatentemp;
    foodeaten=[foodeaten,eatentemp];

    % 4.) determine what state transitions occurred, among the live ants.
    state2=getstate(ants,NEnergyStates,SAMEPLACE);
        % NOTE:  since new ants can be created, they will not have experienced
        % "transitions."  Therefore, only the first NAnts columns are treated.
    s1=state1(1:NAnts);s2=state2(1:NAnts);
    
    % 5.) update the food states:  The food replaced equals the food eaten, randomly distributed.
    if (eaten>MaxFoodSize),
      foodtest=rand(1);
      newfood;
    end
    
    % 6.) update the step and states. 
    STEP=STEP+1;
    state1=state2;
  end
end
