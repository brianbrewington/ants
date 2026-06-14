function p=policy(BEST,ND,ATTR,NActions);

% POLICY chooses new optimal policies for the newly created ants
%
%       USAGE:  p=policy(BEST,NUMBER_DEAD,ATTR,NActions);
%

MUTATION_PROB=0.1;
[r,c]=size(BEST);
mates=floor(rand(2,ND)*c)+1;
spliced=floor(rand(1,ND)*(r-ATTR))+ATTR+1;
mutation=(rand(1,ND)<MUTATION_PROB).*(floor(NActions*(rand(1,ND))+1));

for k=1:ND,
  p(:,k)=[BEST(ATTR+1:spliced(k),mates(1,k));BEST(spliced(k)+1:r,mates(2,k))];
  if (mutation(k)~=0),
    p(floor(c*rand(1))+1,k)=mutation(k);
  end
end


