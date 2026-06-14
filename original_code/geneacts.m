% GENEACTS Perform the actions in vector acts.
%

% NOTE that all energy costs are multiples of ENCOST.

actcosts=zeros(1,length(acts));
eating=find(acts==1);
comm=find(acts==2);
doingnothing=find(acts==3);
moving=find((acts==4)|(acts==6));
listen=find(acts==5);
turning=find(acts==6);
set(ants(6,doingnothing),'color',NOTHING_COLOR);

ants(4,turning)=2*pi*rand(1,length(turning))-pi*ones(1,length(turning));
ants(9,turning)=floor(MASKSIZE*rand(1,length(turning)))+1;

eatentemp=0;
eatants=sparse(WorldSize,WorldSize);
for j=1:length(eating),
  k=eating(j);
  eatants(ants(1,k),ants(2,k))=eatants(ants(1,k),ants(2,k))+1;
end

%if (max(max(eatants))>1),
%  ta=ants(:,eating);
%end

for j=1:length(eating),
  k=eating(j);
  x=ants(1,k);
  y=ants(2,k);
  Available=NEnergyStates-ants(3,k);
  BiteSize=min([BASEFOOD+(BONUS+ENCOST)*(eatants(x,y)-1),Available]);
  % eat 1 unit of food if available. Otherwise, the competition got it first!
  % important bug fix:  if there's no longer any food, need to update the state.
  if (food(x,y)>=BiteSize),
      % at least BiteSize food remains...eat it!
      ants(3,k)=ants(3,k)+BiteSize;
      food(x,y)=food(x,y)-BiteSize;
      eatentemp=eatentemp+BiteSize;
      actcosts(k)=-BiteSize;
  else
    % less than BiteSize remains.  Eat it all.
    actcosts(k)=-food(x,y);
    ants(3,k)=ants(3,k)+food(x,y);
    food(x,y)=0;
   end
  
  % next, if the food here has been exhausted, we need to modify the state.
  if (food(x,y)<=0),
    ants(5,k)=1;
  end
end
set(ants(6,eating),'color',EATING_COLOR);

for j=1:length(moving),
  k=moving(j);
  x=ants(7,k);
  y=ants(8,k);
  % move to the preferred location.
    % first, update actual position, stored in elements 7 and 8 of an ant record.
    moveang=ants(4,k);
    radius=ants(9,k);
    xmove=radius*cos(moveang);
    ymove=radius*sin(moveang);
    newx=x+xmove;
    newy=y+ymove;
    % check new x coordinate for validity.
    if (newx>WorldSize),
      % we walked off the world.  Wrap around.
      ants(7,k)=1+newx-WorldSize;
    else   
      if (newx<1),
        ants(7,k)=WorldSize-(1-newx);
      else
        ants(7,k)=newx;
      end
    end
    
    % check new y coordinate.
    if (newy>WorldSize),
      % we walked off the world.  Wrap around.
      ants(8,k)=1+newy-WorldSize;
    else
      if (newy<1),
        ants(8,k)=WorldSize-(1-newy);
      else
        ants(8,k)=newy;
      end
    end

    % set new grid position.
    ants(1,k)=round(ants(7,k));
    ants(2,k)=round(ants(8,k));
    
    % update plot position data.
    set(ants(6,k),'xdata',ants(7,k),'ydata',ants(8,k),'color',MOVING_COLOR);

    if(food(ants(1,k),ants(2,k))),
      % there is food at the new location
      ants(5,k)=2;
    else
      % no food here!
      ants(5,k)=1;
    end
end

% deduct the (extra) cost for communication
ants(3,comm)=ants(3,comm)-ENCOST*ones(1,length(comm));
set(ants(6,comm),'color',TALKING_COLOR);
set(ants(6,listen),'color',LISTEN_COLOR);

if (length(commlines)>0),
  if (length(find(get(ANTAX,'children')==commlines(1)))),
    delete(commlines);
  end
end

commlines=[];

% perform all communications.
docomm;

%deduct standard energy cost for all ants.
ants(3,:)=ants(3,:)-ENCOST*ones(1,length(state1));

%increment the age of all ants.
ants(ATTR,:)=ants(ATTR,:)+1;


