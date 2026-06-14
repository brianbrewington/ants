function m=oriented(a,b);

% ORIENT Find the relative orientations of sets of points in a plane.
%        Given matrices of Cartesian column vectors in a plane, this 
%        function returns the matrix of relative orientations (in radians)
%        of columns a(:,i) and b(:,j) in a matrix element m(j,i).  All
%        orientations are in the interval (-Pi,Pi).  When elements are the
%        same, the function flags them with the value "99" rather than
%        arbitrarily assigning an angle.
%
%        USAGE:  orientations=orient(a,b);
%  
%        NOTE:  matrices a and b must be 2-by-n, since they contain points
%               in the plane.
%

[m1,n1]=size(a);
[m2,n2]=size(b);

for i=1:n1,
  for j=1:n2,
    y=b(2,j)-a(2,i);
    x=b(1,j)-a(1,i);
    if (x==0)&(y==0),
      m(j,i)=99;
    else
      m(j,i)=atan2(y,x);
    end
  end
end
