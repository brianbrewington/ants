% NEWFOOD Introduce a quantity of new food into the environment.
%

while((foodtest<FoodDensity)&(eaten>MaxFoodSize)),
  x=floor(WorldSize*rand(1))+1;
  y=floor(WorldSize*rand(1))+1;
  newf=min([floor(MaxFoodSize*rand(1))+1 eaten]);
  food(x,y)=food(x,y)+newf;
  if (newf>0),
    set(StatusBar,'string',['Adding ',num2str(newf),...
    ' units of food at location (',num2str(x),',',num2str(y),').']);
  end
  eaten=eaten-newf;
  foodtest=rand(1);
end


