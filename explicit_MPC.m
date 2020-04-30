clear all 

% Question: Why does the game converge on one-shot even with the
% constraints on U and the matrices setting being what they are?

% Question 2: Why is the state space only defined with one control
% solution? When will we ever have more than 1 partitions?

% Seems like having a nonzero R makes it an iterative game. Once R passes
% 0.15 * eye(3) solution is reached in 3 steps 

% I just inserted a nonnegative constraint for the input and it gives a
% nonuniform partition space for the explicit MPC solution.

%% Set up
% Describe latencies of network
% latencyi(xi) = li*xi + bi
% L = [l1  0  0;
%       0 l2  0;
%       0  0  l3];
% b = [b1; b2; b3];
l1 = 1;
l2 = 1;
l3 = 1;
L = diag([l1; l2; l3]);
b = [0; 0; 0];

% Describe cost in terms of latencies -- *TODO: Yiling*
Q = L;
% Q = eye(3);
R = [0 0  0;
     0 0.25  0;
     0 0  0];
P = Q;
N = 20 ; 
A = [1  0  0;
     0  1  0;
     0  0  1];
B = [1  0  0;
     0  1  0;
     0  0  1];
gamma = 0.5;

% Routing Game settings
x0 = [0.30; 0.5; 0.2];
x_eq = [1/3; 1/3; 1/3];
simSteps = N;

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
Au = [-eye(nU);
      ones(1, nU);
      -ones(1, nU)] ;
bu = [u_min;
      u_mass+ep;
     -u_mass+ep] ;
Ax = [-eye(nX);
       ones(1, nX);
      -ones(1, nX)] ; 
bx = [x_min;
      x_mass+ep;
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
%  G(qxn), W(qx1), S(qxm), F_mp(nxm), Q_mp(nxn), where q are the number of constraints,
%  n=N*nU (number of u-variables); m=nX (number of x-parameters)
mpqpsol = mpqp(Q_mp,F_mp,G,W,S,thmin,thmax,verbose,'quadprog','linprog', envelope) ; 

%% Let's plot the partition space
h=pwaplot(mpqpsol);

%% Now let's simulate the routing game
step = 1;
nr = mpqpsol.nr;
ut = zeros(nU, simSteps);
xt = zeros(nX, simSteps+1);
xt(:, 1) = x0;
lqr_costs = -1*ones(1,simSteps+1);
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
        %break
    else
        disp("Control is: ")
        ut(:, step) = mpqpsol.F((j-1)*nU+1:j*nU,:)*xt(:, step) + mpqpsol.G((j-1)*nU+1:j*nU);
        ut(:, step)
        disp("------")
        if ~(u_mass - ep <= ones(nU, 1)'*ut(:, step) <= u_mass + ep)
            error("Input constraints are violated! Game is ending")
            break
        end
        lqr_costs(step) = xt(:, step)'*Q*xt(:, step) + ut(:, step)'*R*ut(:, step);
        % Step env
        xt(:, step+1) = A*xt(:, step) + B*ut(:, step); 
        if ~(x_mass-ep<=ones(nX,1)'*xt(:, step+1)<=x_mass+ep)
            error("State constraints are violated! Game is ending")
            break
        end
        step = step + 1 ;
    end
end
lqr_costs(simSteps+1) = xt(:, simSteps+1)'*P*xt(:, simSteps+1);

% disp("Control is: ")
% mpqpsol.F(1:3,:)*x0 + mpqpsol.G(1:3)
% disp("------")

%% Let's plot the partition space
h=pwaplot(mpqpsol);
figure(h);
hold on;
plot3(x0(1),x0(2),x0(3), "-o");
for i=2:simSteps+1
    %xt(1, i), xt(2, i), xt(3, i)
    %plot3(xt(1, i), xt(2, i), xt(3, i), "-o");
    plot3([xt(1, i-1) xt(1, i)], [xt(2, i-1) xt(2, i)], [xt(3, i-1) xt(3, i)],"-o");
    pause(0.5)
end
hold off;

%% Plot the results
latencies = zeros(nX, simSteps);

for i=1:simSteps+1
    latencies(:, i) = L*xt(:, i) + b;
end

h1 = subplot(2,2,1:2);
hold on
t = 0:simSteps;
plot(t, xt(1, :))
plot(t, xt(2, :))
plot(t, xt(3, :))
axis([0 simSteps 0 inf])
legend("path p1 (e1)", "path p2 (e2)", "path p3 (e3)")
ylabel("F_t")
xlabel("t")
title ("flow")
hold off

h2 = subplot(2,2,3); 
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

h3 = subplot(2,2,4); 
hold on
t = 0:simSteps;
plot(t, lqr_costs)
axis([0 simSteps -inf inf])
ylabel("cost_val")
xlabel("t")
title ("lqr_costs")

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
