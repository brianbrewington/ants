function d=distan(m1,m2);

[r1,c1]=size(m1);
[r2,c2]=size(m2);
for k=1:c1,
  for j=1:c2,
    d(j,k)=norm(m1(:,k)-m2(:,j));
  end
end
