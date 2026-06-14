% DOCOMM.M perform all ant-to-ant communications
%

if (length(listen)~=0)&(length(comm)~=0),
  % at least one ant is communicating, and at least one is listening.
  basecomm=ants(REALX:REALY,comm);
  comm9=[];
  for k=1:9,
    comm9=[comm9,comm];
  end
  baselist=ants(REALX:REALY,listen);
  shift=WorldSize*ones(1,length(comm));
  commshift=[];
  for k=-1:1,
    for j=-1:1,
      commshift=[commshift,[basecomm(1,:)+k*shift;basecomm(2,:)+j*shift]];
    end
  end

  for k=1:length(listen),
    x=ants(REALX,listen(k));
    y=ants(REALY,listen(k));
    test=[];
    test(1,:)=(commshift(1,:)-x.*ones(1,9*length(comm))).^2;
    test(2,:)=(commshift(2,:)-y.*ones(1,9*length(comm))).^2;
    test=sqrt(sum(test));
    inrange=find(test<MASKSIZE);
    % only perform the updates for communications which were in range.
    if (length(inrange)~=0),
      [distance,indx]=min(test(inrange));
      indx=inrange(indx);
      commindx=comm9(indx);   
      ants(XDEST,listen(k))=ants(REALX,commindx);
      ants(YDEST,listen(k))=ants(REALY,commindx);
      ants(TOFOO,listen(k))=ants(ONFOO,commindx);
      commx=ants(REALX,commindx);commy=ants(REALY,commindx);
      commlines=[commlines,line([commx,x],[commy,y],'color',[0 0.5 0],...
                   'erasemode','xor','linestyle',':','visible',animated)];
    end
  end
end  

