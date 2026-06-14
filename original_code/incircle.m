function m=incircle(N);

% INCIRCLE returns points enclosed in a circle
%
%	   Given a square matrix of side length 2N+1, we inscribe a circle
% 	   of radius N centered at element N+1,N+1 (with indices starting
%	   at 1). If the circle encloses any part of a matrix element, we
%	   return a 1 at that location, and if not, it is a 0.
%
%	   USAGE:  m=incircle(N);
%

s=2*N+1;
x=[-N:N];
tm=meshgrid(x);
tm=(tm.^2+(tm').^2).^0.5;
m=(tm<=N);
