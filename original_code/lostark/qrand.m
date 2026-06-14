function RANDOM=qrand(st,RANDOM);

% RANDOM=qrand(STEP,RANDOM)
%

EXITPROB=0.001;
if (RANDOM),
  % already in a period of random actions.  Test to see if we tunnel out.
  if (rand(1)<EXITPROB),
    RANDOM=0;
  end
else
  % operating on optimal policy
  if (rand(1)<(1/st^0.6)),
    RANDOM=1;
  end
end
