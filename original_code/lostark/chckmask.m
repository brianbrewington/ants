function MASKSIZE=chckmask(Sliders,StatusBar);

% CHCKMASK checks the communications mask if the WorldSize is changed.
%
%       USAGE:  MASKSIZE=chckmask(Sliders,StatusBar);

WorldSize=get(Sliders(5,1),'value');
reduced=ceil(sqrt(2)*WorldSize/2);
maxmask=get(Sliders(9,1),'max');
MASKSIZE=get(Sliders(9,1),'value');

if (reduced<MASKSIZE),
  set(Sliders(9,1),'value',reduced,'max',reduced);
  set(Sliders(9,2),'string',num2str(reduced));
  MASKSIZE=reduced;
  str=get(StatusBar,'string');
  set(StatusBar,'string',[str,'; changed MASKSIZE to ',num2str(MASKSIZE)]);
end

set(Sliders(9,1),'max',reduced);




