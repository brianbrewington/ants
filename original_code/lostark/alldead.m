
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
ants(1,:)=floor(WorldSize*rand(1,NAnts))+1;
ants(2,:)=floor(WorldSize*rand(1,NAnts))+1;
ants(3,:)=NEnergyStates*ones(1,NAnts);
ants(4,:)=(2*pi*rand(1,NAnts))-pi*ones(1,NAnts);
ants(5,:)=ones(1,NAnts);
ants(6,:)=zeros(1,NAnts);
ants(7:8,:)=ants(1:2,:);
ants(9,:)=zeros(1,NAnts);
ants(10,:)=ones(1,NAnts);
ants(10,:)=sameplac(ants);
 
for k=1:NAnts,
  ants(6,k)=line(ants(1,k),ants(2,k),'linestyle','*','color',[1 0 1],...
            'erasemode','xor','visible','on','markersize',5,...
            'visible',animated);
  if food(ants(1,k),ants(2,k)),
    ants(5,k)=2;
  end
end
