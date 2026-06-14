% QUANT100.M tests "quantize.m" with 100 points.  Set NDirections beforehand.

angles=2*pi*rand(1,100)-pi*ones(1,100);
fig=gcf;
clf;
plot(exp(i*angles),'y+');
axis('image');
hold on;
for k=1:100,
  q=quantize(angles(k),NDirections);
  v=exp(i*angles(k));
  x=real(1.01*v);
  y=imag(1.01*v);
  text(x,y,num2str(q));
end

increment=2*pi/NDirections;
binends=exp(i.*[-pi+increment/2:increment:pi-increment/2]);
len=length(binends);
binpoints=zeros(2,2*len);
for k=1:len,
  binpoints(1,2*k)=real(binends(k));
  binpoints(2,2*k)=imag(binends(k));
end

l=line(binpoints(1,:),binpoints(2,:),'linestyle',':','color',[0 0 1]);
