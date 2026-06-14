% DOCOMM.M perform all ant-to-ant communications
%

if (length(listen)~=0)&(length(comm)~=0),
  % at least one ant is communicating, and at least one is listening.
  basecomm=ants(7:8,comm);
  comm9=[comm, comm, comm, comm, comm, comm, comm, comm, comm];
  baselist=ants(7:8,listen);
  shift=WorldSize*ones(1,length(comm));
  commshift=[];
  for k=-1:1,
    for j=-1:1,
      commshift=[commshift,[basecomm(1,:)+k*shift;basecomm(2,:)+j*shift]];
    end
  end

  for k=1:length(listen),
    x=ants(7,listen(k));
    y=ants(8,listen(k));
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
      xdiff=commshift(1,indx)-x;
      ydiff=commshift(2,indx)-y;
      if (xdiff~=0)|(ydiff~=0),
        theangle=atan2(ydiff,xdiff);
      end
      ants(4,listen(k))=theangle;
      ants(9,listen(k))=distance;
      commx=ants(7,commindx);commy=ants(8,commindx);
      commlines=[commlines,line([commx,x],[commy,y],'color',[0 0.5 0],...
                   'erasemode','xor','visible','on','linestyle',':','visible',animated)];
    end
  end
end  

