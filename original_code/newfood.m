% NEWFOOD Introduce a quantity of new food into the environment.
%

while((rand(1)<FoodDensity)&(eaten>MaxFoodSize)),
  x=floor(WorldSize*rand(1))+1;
  y=floor(WorldSize*rand(1))+1;
  newf=MaxFoodSize*rand(1);
  food(x,y)=food(x,y)+newf;
  set(StatusBar,'string',['Adding ',num2str(newf),...
    ' units of food at location (',num2str(x),',',num2str(y),').']);
  eaten=eaten-newf;
end


