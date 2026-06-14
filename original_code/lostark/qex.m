AVERAGING_LENGTH=50;

[rq,cq]=size(QTable);
[ract,cact]=size(ActionIndex);
[y,i]=min(QTable');
[m,o,p]=find(Qindex);
%[mi,mj,mv]=find(markov);
%[r,c]=size(markov);
%mr=[1:r];mc=[1:c];
%mr(mi)=[];mc(mj)=[];
%markov(mr,:)=[];
%markov(:,mc)=[];
optacts=reshape(ActionIndex(i,:)',cact*rq/(NEnergyStates+1),NEnergyStates+1)',
dd=conv(ones(1,AVERAGING_LENGTH)./AVERAGING_LENGTH,DEADDATA)';
dd=dd(AVERAGING_LENGTH:length(dd)-AVERAGING_LENGTH);
plot(dd),
now=clock;
datestamp=[num2str(now(4)),':',num2str(now(5)),':',num2str(round(now(6))),'  ',date],
clear rq,clear cq,clear ract,clear cact, clear y, clear i, clear m, clear o, clear p,
clear mi, clear mj,clear mv, clear r, clear c,
save qttemp ants datestamp NEnergyStates QTable Qindex markov NActions BETAEXP GAMMA NAnts FoodDensity ENCOST WorldSize MaxFoodSize


