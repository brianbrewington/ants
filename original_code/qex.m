AVERAGING_LENGTH=50;

[rq,cq]=size(QTable);
[ract,cact]=size(ActionIndex);
[y,i]=min(QTable');
zots=find(y==0);
i(zots)=(ract)*ones(size(zots));
[m,o,p]=find(Qindex);

optacts=reshape(ActionIndex(i,:)',cact*rq/(NEnergyStates+1),NEnergyStates+1)',
dd=conv(ones(1,AVERAGING_LENGTH)./AVERAGING_LENGTH,DEADDATA/max(DEADDATA))';
dd=dd(AVERAGING_LENGTH:length(dd)-AVERAGING_LENGTH);
cd=conv(ones(1,AVERAGING_LENGTH)./AVERAGING_LENGTH,CODATA)';
cd=cd(AVERAGING_LENGTH:length(cd)-AVERAGING_LENGTH);
od=conv(ones(1,AVERAGING_LENGTH)./AVERAGING_LENGTH,OPTDATA)';
od=od(AVERAGING_LENGTH:length(od)-AVERAGING_LENGTH);
fd=conv(ones(1,AVERAGING_LENGTH)./AVERAGING_LENGTH,FOODDATA/max(FOODDATA))';
fd=fd(AVERAGING_LENGTH:length(fd)-AVERAGING_LENGTH);
xvec=[1:length(dd)];
plot(xvec,dd,'b',xvec,fd,'y',xvec,od,'r',xvec,cd,'g');
legend('b-','Death rate','y-','Food eaten','r-','Optimal %','g-','Colocation %');
now=clock;
datestamp=[num2str(now(4)),':',num2str(now(5)),':',num2str(round(now(6))),'  ',date],

clear rq,clear cq,clear ract,clear cact, clear y, clear i, clear m, clear o, clear p,
clear mi, clear mj,clear mv, clear r, clear c,
save qttemp ants datestamp NEnergyStates QTable Qindex NActions BETAEXP GAMMA NAnts FoodDensity ENCOST WorldSize MaxFoodSize FOODDATA DEADDATA OPTDATA CODATA DEADTIME


