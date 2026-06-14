% DOACTS Perform the actions in vector acts.
%

% NOTE that all energy costs are multiples of ENCOST.

axes(ANTAX);
eating=find(acts==1);
comm=find(acts==2);
doingnothing=find(acts==3);
moving=find((acts==4)|(acts==6));
listen=find(acts==5);
turning=find(acts==6);
set(ants(HANDL,doingnothing),'color',NOTHING_COLOR);

rad=floor(MASKSIZE*0.05*rand(1,length(turning)))+1;
dir=floor(2*pi*rand(1,length(turning)))-pi;
newx=ants(REALX,turning)+cos(dir).*rad;
newy=ants(REALY,turning)+sin(dir).*rad;
for k=1:length(turning),
  if (newx(k)>WorldSize),
    newx(k)=newx(k)-WorldSize+1;
  elseif (newx(k)<1),
    newx(k)=WorldSize-(1-newx(k));
  end
  if (newy(k)>WorldSize),
    newy(k)=newy(k)-WorldSize+1;  
  elseif (newy(k)<1),
    newy(k)=WorldSize-(1-newy(k));  
  end
end

ants(XDEST,turning)=newx;
ants(YDEST,turning)=newy;

eatentemp=0;

for j=1:length(eating),
  k=eating(j);
  x=ants(XGRID,k);
  y=ants(YGRID,k);
  Available=NEnergyStates-ants(ENERG,k);
  BiteSize=min([BASEFOOD,Available]);

  if (food(x,y)>BiteSize),
      % at least BiteSize food remains...eat it!
      ants(ENERG,k)=ants(ENERG,k)+BiteSize;
      food(x,y)=food(x,y)-BiteSize;
      eatentemp=eatentemp+BiteSize;
  else
    % less than BiteSize remains.  Eat it all.
    ants(ENERG,k)=ants(ENERG,k)+food(x,y);
    eatentemp=eatentemp+food(x,y);
    food(x,y)=0;
   end
  
  % next, if the food here has been exhausted, we need to modify the state.
  if (food(x,y)<=0),
    ants(ONFOO,k)=1;
  end
end


set(ants(HANDL,eating),'color',EATING_COLOR);

for j=1:length(moving),
  k=moving(j);
  % move to the preferred location.
    % first, update actual position, stored in elements 7 and 8 of an ant record.
    ants(REALX,k)=ants(XDEST,k);
    ants(REALY,k)=ants(YDEST,k);

    % set new grid position.
    ants(XGRID,k)=round(ants(REALX,k));
    ants(YGRID,k)=round(ants(REALY,k));
    
    % update plot position data and color
    set(ants(HANDL,k),'xdata',ants(REALX,k),'ydata',ants(REALY,k),'color',MOVING_COLOR);

    if(food(ants(XGRID,k),ants(YGRID,k))),
      % there is food at the new location
      ants(ONFOO,k)=2;
    else
      % no food here!
      ants(ONFOO,k)=1;
    end
end

% No longer deduct the (extra) cost for communication ...
% ants(ENERG,comm)=ants(ENERG,comm)-ENCOST*ones(1,length(comm));

set(ants(HANDL,comm),'color',TALKING_COLOR);
set(ants(HANDL,listen),'color',LISTEN_COLOR);

if (length(commlines)>0),
  if (length(find(get(ANTAX,'children')==commlines(1)))),
    delete(commlines);
  end
end

commlines=[];

% perform all communications.
docomm;

%deduct standard energy cost for all ants.
ants(ENERG,:)=ants(ENERG,:)-ENCOST;

