
set(StatusBar,'string','All ants are now dead.  Restarting population.');

DEADTIME=STEP-ALLDEAD;
DEADDATA=[DEADDATA,DEADTIME];
ALLDEAD=STEP;
axes(METRAX);
xd=get(DLINE,'xdata');
set(DLINE,'ydata',DEADDATA,'xdata',[xd,length(xd)+1]);
drawnow;
NAnts=InitAnts;
      
axes(ANTAX);
cla;
      
food=makefood(FoodDensity,MaxFoodSize,WorldSize);
[xfood,yfood]=find(food);
FOODLINE=line(xfood,yfood,'linestyle','+','color',[1 1 0],...
'erasemode','xor','markersize',4,'visible',animated);
     
% locations and preferences are initially random.
ants(XGRID,:)=floor(WorldSize*rand(1,NAnts))+1;
ants(YGRID,:)=floor(WorldSize*rand(1,NAnts))+1;
ants(ENERG,:)=NEnergyStates*ones(1,NAnts);
ants(XDEST,:)=floor(WorldSize*rand(1,NAnts))+1;
ants(ONFOO,:)=ones(1,NAnts);
ants(HANDL,:)=zeros(1,NAnts);
ants(REALX:REALY,:)=ants(1:2,:);
ants(YDEST,:)=floor(WorldSize*rand(1,NAnts))+1;
ants(SAMEP,:)=ones(1,NAnts);ants(SAMEP,:)=sameplac(ants,food,MASKSIZE,MASK,WorldSize);
 
for k=1:NAnts,
  ants(HANDL,k)=line(ants(XGRID,k),ants(YGRID,k),'linestyle','*','color',[1 0 1],...
            'erasemode','xor','visible','on','markersize',5,...
            'visible',animated);
  if food(ants(XGRID,k),ants(YGRID,k)),
    ants(ONFOO,k)=2;
  end
end
