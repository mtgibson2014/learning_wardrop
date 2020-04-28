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

%% Set up
Q = eye(3);
R = zeros(3);
P = eye(3);
N = 4 ; 
A = eye(3);
B = eye(3);
gamma = 0.5;

% Routing Game settings
x0 = [0.30; 0.05; 0.65];
x_eq = [1/3; 1/3; 1/3];
simSteps = N;
% Describe latencies of network
% latencyi(xi) = li*xi + bi
% L = [l1  0  0;
%       0 l2  0;
%       0  0  l3];
% b = [b1; b2; b3];
L = eye(3);
b = [0; 0; 0];

% Setting params
nX = size(A,2);
nU = size(B,2);
A = gamma*A;
B = (1-gamma)*B;

% Set constraints
x_mass = 1;
u_mass = 1;
u_min = zeros(nU, 1);
u_max = ones(nU, 1);
x_min = zeros(nX, 1);
x_max = ones(nX, 1);
ep = 0.0001;
Au = [ones(1, nU);
      -ones(1, nU)] ;
bu = [u_mass+ep;
     -u_mass+ep] ; 
Ax = [ones(1, nX);
      -ones(1, nX)] ; 
bx = [x_mass+ep;
     -x_mass+ep] ;
% u_min = zeros(nU, 1);
% u_max = ones(nU, 1);
% x_min = zeros(nX, 1);
% x_max = ones(nX, 1);
% Au = [-eye(nU);
%        eye(nU);
%        ones(1, nU);
%       -ones(1, nU)] ;
% bu = [u_min;
%       u_max;
%       1;
%       -1] ; 
% Ax = [-eye(nX);
%        eye(nX);
%        ones(1, nX);
%       -ones(1, nX)] ; 
% bx = [x_min;
%       x_max;
%       1;
%       -1] ; 


% % Similar example as used in tests
% gamma = 0.5 ; 
% A = [1 1;
%      1 1];
% B = [0;
%      1];
% nX = size(A,2);
% nU = size(B,2);
% Q = eye(nX);
% P = 2*eye(nX);
% R = eye(nU);
% N = 2;
% Au = 1;
% bu = -1 ; 
% Ax = [1 1] ; 
% bx = 1;

%% Make necessary inputs to MPQP
% Objective function -----------------
[Q_mp, F_mp, c_mp, Y_mp] = makeMPQPobjective(Q,R,P,N,A,B,nX,nU) ; 

% Constraints -----------------
[G,W,S] = makeMPQPconstraints(Au,bu,Ax,bx,A,B,nX,nU,N) ;

% thmin=[xmin;refmin;umin;mdmin];
% thmax=[xmax;refmax;umax;mdmax];
verbose = 1; % prints out computational information
% options.zerotol = 1.0000e-08 ;  
% options.removetol = 1.0000e-04 ; 
% options.flattol = 1.0000e-05 ; 
% options.normalizetol = 0.0100 ;
% options.maxiterNNLS = 500 ;
% options.maxiterQP = 200 ;
% options.maxiterBS = 100 ;
% options.polyreduction = 2 ;
envelope = 1;
thmin = x_min ; 
thmax = x_max ; 

%% Solve mpQP
% mpqpsol = computeMPQP(Q_mp,F_mp,c_mp,G,W,S,[0;0;0],[1;1;1],verbose,options);
%  DIMENSIONS:
%  A(qxn), b(qx1), S(qxm), C(qxm), Q(nxn), where q are the number of constraints,
%  n=number of x-variables; m=number of th-parameters
% mpqpsol=mpqp(Q,     C, A,b,S,thmin,thmax,verbose,qpsolver,lpsolver,envelope)
mpqpsol = mpqp(Q_mp,F_mp,G,W,S,thmin,thmax,verbose,'quadprog','linprog', envelope) ; 

%% Let's plot the partition space
h=pwaplot(mpqpsol);

%% Now let's simulate the routing game
step = 1;
nr = mpqpsol.nr;
ut = zeros(nU, simSteps);
xt = zeros(nX, simSteps+1);
xt(:, 1) = x0;
while step <= simSteps
    j = -1;
    % Choose control
    % if {x: H(i)*x<=K(i)} then u=F(i)*x+G(i)
    for r = 1:nr
        if all(mpqpsol.H(mpqpsol.i1(r):mpqpsol.i2(r), :)*xt(:, step) <= mpqpsol.K(mpqpsol.i1(r):mpqpsol.i2(r),:))
            j = r;
            break
        end
    end
    if j == -1
        error("Control not found! Game is ending");
        break
    else
        ut(:, step) = mpqpsol.F((j-1)*nU+1:j*nU,:)*xt(:, step) + mpqpsol.G((j-1)*nU+1:j*nU);
        % Step env
        xt(:, step+1) = A*xt(:, step) + B*ut(:, step); 
        step = step + 1 ;
    end
end

%% Plot the results
latencies = zeros(nX, simSteps);

for i=1:simSteps+1
    latencies(:, i) = L*xt(:, i) + b;
end

h1 = subplot(2,1,1);
hold on
t = 0:simSteps;
plot(t, xt(1, :))
plot(t, xt(2, :))
plot(t, xt(3, :))
axis([0 simSteps 0 1])
legend("path p1 (e1)", "path p2 (e2)", "path p3 (e3)")
ylabel("F_t")
xlabel("t")
title ("flow")
hold off

h2 = subplot(2,1,2); 
hold on
t = 0:simSteps;
plot(t,latencies(1, :))
plot(t,latencies(2, :))
plot(t,latencies(3, :))
axis([0 simSteps 0 inf])
legend("path p1 (e1)", "path p2 (e2)", "path p3 (e3)")
ylabel("l(f_t(i))")
xlabel("t")
title ("latency")
hold off

%% Old example
% A = [1/3 1/3 1/3;
%      1/3 1/3 1/3;
%      1/3 1/3 1/3];
% B = eye(3);
% C = eye(3);
% D = zeros(3);
% plant = ss(A,B,C,D);
% x0 = zeros(3,1);
% 
% 
% 
% Ts = 0.05; % Sample time
% p = 10;             % Prediction horizon
% m = 1;              % Control horizon
% MV = struct('Min',{-25,-25},'Max',{25,25},'ScaleFactor',{50,50});
% OV = struct('Min',{[-0.5;-Inf],[-100;-Inf]},'Max',{[0.5;Inf],[100;Inf]},'ScaleFactor',{1,200});
% Weights = struct('MV',[0 0 0],'MVRate',[0 0 0],'OV',[1 1 1]);
% mpcobj = mpc(plant,Ts,p,m,Weights,MV,OV);
% % mpcobj = mpc(plant,Ts, 10, 1);
% 
% range = generateExplicitRange(mpcobj);
% 
% range.State.Min = [0; 0; 0; 0];
% range.State.Max = [1; 1; 1; 1];
% 
% % range.Reference.Min(:) = 0;
% % range.Reference.Max(:) = 0;
% 
% range.ManipulatedVariable.Min(:) = -1;
% range.ManipulatedVariable.Max(:) =  1;
% 
% mpcobjExplicit = generateExplicitMPC(mpcobj, range);
% mpcobjExplicitSimplified = simplify(mpcobjExplicit, 'exact');
% display(mpcobjExplicitSimplified)
% 
% params = generatePlotParameters(mpcobjExplicitSimplified);
% params.State.Index = [2 3];
% params.State.Value = [0 0];
% % params.Reference.Index = 1;
% % params.Reference.Value = 0;
% params.ManipulatedVariable.Index = [1 2 3];
% params.ManipulatedVariable.Value = [0 0 0];
% 
% plotSection(mpcobjExplicitSimplified, params);
% axis([-1 1 -1 1])
% grid
% % xlabel('?')
% % ylabel('??')
% 
% % How to add the simplex constraint to this thing? 
% 
% disp("Here's piecewise affine solution")
% mpcobjExplicitSimplified.PiecewiseAffineSolution
% mpcobjExplicitSimplified.PiecewiseAffineSolution.F
% mpcobjExplicitSimplified.PiecewiseAffineSolution.G
% mpcobjExplicitSimplified.PiecewiseAffineSolution.H
% mpcobjExplicitSimplified.PiecewiseAffineSolution.K
% 
% % u = Fx + G
% % {x: H*x<=K} and u=F*x+G
