% RANDACTS Create a vector of random, legal actions.
%

energy=round(state1/1000);
onfood=rem(state1,10);

for k=1:length(state1),
  legal;
  acts(k)=allowed(floor(length(allowed)*rand(1))+1);
end

