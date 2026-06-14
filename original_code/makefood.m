function food=makefood(FoodDensity,MaxFoodSize,WorldSize);

% MAKEFOOD Create food for the artificial ant problem.
%            
%               USAGE:  food=makefood(density,MaxFoodSize,WorldSize);
%

N=floor(FoodDensity*WorldSize^2);
foodsizes=round(MaxFoodSize*rand(1,N))+1;
foodx=floor(rand(1,N)*WorldSize)+1;
foody=floor(rand(1,N)*WorldSize)+1;
food=sparse(foodx,foody,foodsizes,WorldSize,WorldSize);
