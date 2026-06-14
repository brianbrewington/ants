% ----------------------------------------
% MAIN LOOP
% ----------------------------------------

TotalFood=full(sum(sum(food)));
while (~stopped),
  if (paused==0),
    set(StepBox,'string',['Step=',num2str(STEP)]);
    BETA=(1/STEP)^BETAEXP;

    % 1.) Kill off ants with no remaining energy:  if an ant dies, he should be replaced, so the
    %           population can continue to survive.

    dead=find(state1<1000);
    if (length(dead)~=0),
        dd=dd+length(dead);
        delete(ants(6,dead));       %just deletes the graphics, not the record of the ants.
        ants=qnewants(ants,dead,food,WorldSize,MASKSIZE,NEnergyStates,NTotalStates,NActions,animated);
    end

    if (rem(STEP,UPDATE)==0),
      set(StatusBar,'string',['Now evaluating STEP number ',num2str(STEP)]);
      DEADDATA=[DEADDATA;dd/NAnts];
      CODATA=[CODATA;colocation/UPDATE];
      OPTDATA=[OPTDATA;optimal/UPDATE];
      FOODDATA=[FOODDATA;foodeaten/UPDATE];
      DEADTIME=[DEADTIME;STEP];
      sum(sum(food)),
      axes(METRAX);
      set(DLINE,'ydata',DEADDATA/max(DEADDATA),'xdata',DEADTIME);
      set(CLINE,'ydata',CODATA,'xdata',DEADTIME);
      set(OLINE,'ydata',OPTDATA,'xdata',DEADTIME);
      set(FLINE,'ydata',FOODDATA/max(FOODDATA),'xdata',DEADTIME);
                        
      dd=0;optimal=0;colocation=0;foodeaten=0;
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

    % 2.) select ant actions, based on the QTable or randomizer, after getting the ant states.
    state1=getstate(ants,NEnergyStates,SAMEPLACE);
    acts=zeros(1,NAnts);
    set(RandomBox,'string','Using mixed actions.');
    pickacts;

    % 3.) perform ant actions; record food consumed in "eaten."
    doacts;
    ants(SAMEP,:)=sameplac(ants,food,MASKSIZE,MASK,WorldSize);

    eaten=eaten+eatentemp;   
    foodeaten=foodeaten+eatentemp;

    % 4.) determine state transitions occurred, and update the QTable accordingly
    state2=getstate(ants,NEnergyStates,SAMEPLACE);
    transitions=[state1;state2];
    qupdate;
    
    % 5.) update the food states:  The food replaced equals the food eaten, randomly distributed.

    if (eaten>MaxFoodSize),
      newfood;
    end
    
    % 6.) update the step and the monitor metrics
    STEP=STEP+1;
    colocation=colocation+mean(ants(SAMEP,:));
    optimal=optimal+(1-mean(ants(RANDA,:)));
  end
end
