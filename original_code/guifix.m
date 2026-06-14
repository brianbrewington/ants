function NT=guifix(Sliders,Lifetime,QTsize,NActions,ECres);

% GUIFIX updates the information boxes when relevant parameters are altered.
%
%       USAGE: guifix(sliders,Lifetime,QTsize,NActions,ECres);


NE=round(get(Sliders(8,1),'value'));
SL=round(get(Sliders(12,1),'value'));
EC=ECres*round(get(Sliders(2,1),'value')/ECres);
NT=(NE+1)*(SL+1)*2*2;

set(QTsize,'string',['QTable size:  ',...
num2str(NT),' X ',num2str(NActions)]);

set(Lifetime,'string',['Base lifespan:  ',...
num2str(round((NE-0.5)/EC)),' cycles']);

