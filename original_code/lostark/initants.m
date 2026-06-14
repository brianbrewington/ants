NActions=6;                             % number of actions
NAnts=10;                               % number of ants per round
NEnergyStates=50;                       % number of energy states in QTable
ENCOST=0.5;                             % standard energy cost per round
GAMMA=0.995;                            % future cost discount factor
BETAEXP=0.75;                           % exponent reducing BETA each round
WorldSize=20;                           % side length for the world
MaxFoodSize=5;                          % largest food size at a location
FoodDensity=0.01;                       % probability a location has food
paused=0;                               % simulation paused or not?                               
ECres=0.1;                              % increment of change in ENCOST
MASKSIZE=ceil(sqrt(2)*WorldSize/2);     % communication range
BASEFOOD=0.1;                           % allowed food consumption if only one ant is present
SAMEPLACE=1;                            % number of other ants at the same location of which an ant is aware
BONUS=0.1;                              % additional food consumption allowed per extra ant on a food site
animated='on';				% by default, the animation is turned on.
graphical='on';				% may be used in the future...

