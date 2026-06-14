function animated=toggle(anim,button);

% TOGGLE toggles the radio buttons in the ant GUI.
%
%       USAGE:  animated=toggle(anim);
%

anval=get(anim(1),'value');
hideval=get(anim(2),'value');

if (button==1),
  % changed the value of "ANIMATED" button
  if (anval),
    animated='on';
    set(anim(2),'value',0);
  else
    animated='off';
    set(anim(2),'value',1);
  end
end

if(button==2),
  % changed the value of "INVISIBLE" button
  if (hideval),
    animated='off';
    set(anim(1),'value',0);
  else
    animated='on';
    set(anim(1),'value',1);
  end
end
