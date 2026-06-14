function t=toroid(m);

% TOROID makes a 2d toroidal map by surrounding the original matrix m
%	with 8 copies of itself, like so:
%
%	m	m	m
%	m	m	m
%	m	m	m
%
%	USAGE: t=toroid(m);
%

t=[m,m,m;m,m,m;m,m,m];
