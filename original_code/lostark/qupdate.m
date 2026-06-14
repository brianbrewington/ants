% QUPDATE Update the QTable.
%

[r,c1]=size(QTable);
for k=1:length(acts),
  % for each action, we update a value in the QTable.
  s1=Qindex(transitions(1,k));
  s2=Qindex(transitions(2,k));
  a=acts(k);
  x=ants(1,k);y=ants(2,k);
  c=actcosts(k);
  qv2=QTable(s2,:);
  QTable(s1,a)=(1-BETA)*QTable(s1,a)+(BETA*(c+GAMMA*min(qv2)));
end

  
