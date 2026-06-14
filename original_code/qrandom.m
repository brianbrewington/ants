function [RANDOM,RSTEP]=qrandom(step,RSTEP);

% QRANDOM select whether the actions from a given time
%         step are to be random or not.
%
%         USAGE: r=qrandom(step);

GROWTH_EXP=0.15;
OFFSET=100;
RAND_RUN_LENGTH=500;

%transitions occur at integer boundaries in the function.
if (floor((step+OFFSET+1)^GROWTH_EXP)-floor((step+OFFSET)^GROWTH_EXP)~=0),
  % this is a boundary; we return RANDOM=1 and reset RSTEP.
  RANDOM=1;
  RSTEP=0;
else
  % this is not a boundary; we increment RSTEP and return RANDOM=0
  % if we're past the RANDOM_RUN_LENGTH
  RSTEP=RSTEP+1;
  if (RSTEP>RAND_RUN_LENGTH),
    RANDOM=0;
  else
    RANDOM=1;
  end
end

