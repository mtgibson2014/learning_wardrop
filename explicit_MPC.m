clear all 

% From the example
%
% See following links
%   How to check for region and compute opt control: 
%       https://www.mathworks.com/help/mpc/ug/design-workflow.html#buj6dbz
%   Examples: 
%       https://www.mathworks.com/help/mpc/ug/explicit-mpc-control-of-an-aircraft-with-unstable-poles.html#d120e19058
%   Functions:    
%       https://www.mathworks.com/help/mpc/referencelist.html?type=function&category=explicit-mpc-design
%   Properties of Explicit MPC Object:
%       https://www.mathworks.com/help/mpc/ref/mpc.generateexplicitmpc.html#buikta6-1-EMPCobj
%
%


A = [1/3 1/3 1/3;
     1/3 1/5 1/3;
     1/3 1/3 1/3];
B = eye(3);
C = eye(3);
D = zeros(3);
plant = ss(A,B,C,D);
x0 = zeros(3,1);



Ts = 0.05; % Sample time
% p = 10;             % Prediction horizon
% m = 1;              % Control horizon
% MV = struct('Min',{-25,-25},'Max',{25,25},'ScaleFactor',{50,50});
% OV = struct('Min',{[-0.5;-Inf],[-100;-Inf]},'Max',{[0.5;Inf],[100;Inf]},'ScaleFactor',{1,200});
% Weights = struct('MV',[0 0 0],'MVRate',[0 0 0],'OV',[1 1 1]);
% mpcobj = mpc(plant,Ts,p,m,Weights,MV,OV);
mpcobj = mpc(plant,Ts, 10, 1);

range = generateExplicitRange(mpcobj);

range.State.Min(:) = 0;
range.State.Max(:) = 1;

% range.Reference.Min(:) = 0;
% range.Reference.Max(:) = 0;

range.ManipulatedVariable.Min(:) = -1;
range.ManipulatedVariable.Max(:) =  1;

mpcobjExplicit = generateExplicitMPC(mpcobj, range);
mpcobjExplicitSimplified = simplify(mpcobjExplicit, 'exact');
display(mpcobjExplicitSimplified)

params = generatePlotParameters(mpcobjExplicitSimplified);
params.State.Index = [1 2 3];
params.State.Value = [0 0 0];
% params.Reference.Index = 1;
% params.Reference.Value = 0;
params.ManipulatedVariable.Index = [1 2 3];
params.ManipulatedVariable.Value = [0 0 0];

plotSection(mpcobjExplicitSimplified, params);
axis([-1 1 -1 1])
grid
% xlabel('?')
% ylabel('??')

% How to add the simplex constraint to this thing? 

disp("Here's piecewise affine solution")
mpcobjExplicitSimplified.PiecewiseAffineSolution
mpcobjExplicitSimplified.PiecewiseAffineSolution.F
mpcobjExplicitSimplified.PiecewiseAffineSolution.G
mpcobjExplicitSimplified.PiecewiseAffineSolution.H
mpcobjExplicitSimplified.PiecewiseAffineSolution.K




% 
% 
% 
% 
% 
% % Note: The "manipulated variable" is the control variable.
% 
% A = eye(3);
% B = eye(3);
% C = eye(3);
% D = zeros(3);
% sys = ss(A, B, C, D, 1);
% total_flow = 100;
% 
% mpcobj = mpc(sys); 
% 
% % setconstraint(mpcobj,[1 -1 1],[1 -1  3],1);
% 
% range = generateExplicitRange(mpcobj);
% 
% range.State.Min(:) = [0;0;0];
% range.State.Max(:) = [1; 1; 1];
% 
% range.ManipulatedVariable.Min = [-1;-1;-1] ;
% range.ManipulatedVariable.Max = [1;1;1] ;
% 
% 
% mpcobjExplicit = generateExplicitMPC(mpcobj, range); 
% mpcobjExplicitSimplified = simplify(mpcobjExplicit, 'exact');
% 
% display(mpcobjExplicitSimplified)
% 
% % Plot piecewise affine partition
% params = generatePlotParameters(mpcobjExplicitSimplified)
% 
% params.State.Index = [];
% params.State.Value = [];
% params.Reference.Index = 1;
% params.Reference.Value = 0;
% params.ManipulatedVariable.Index = [];
% params.ManipulatedVariable.Value = [];
% 
% plotSection(mpcobjExplicitSimplified, params);
% axis([0 1 0 1]);
% grid
% xlabel('State #1');
% ylabel('State #2');
% zlabel('State #3')

