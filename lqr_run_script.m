%% Game Parameters: Network, Linear Cost functions, Dynamics of Game
% 
% % Network params
a = [4/3 4/3 4/3];
b = [2/3 2/3 2/3];
% nx = 4;
% nu = 4;
% mu = 100;
% 
% Alpha = diag(a);
% beta_bar = b';
% 
% % Dynamics of our game = A, B. 
% gamma = 0.5;
% A = gamma*[1/2  0   1/2  0;
%            1/2 1/2   0   0;
%            0   1/2  1/2  0;
%            0    0    0   1];
% % A = gamma*[1  0   1/2  0;
% %            0  1   0   0;
% %            0   1/2  1/2  0;
% %            0    0    0   1];
% B = (1-gamma)*eye(4); 

%% Solve for the Nash and set x^*
co = [1;
      b(2)-b(1);
      b(3)-b(2)]; 
map = [1     1     1;
       a(1) -a(2)  0;
       0     a(2) -a(3)];

x_nash = map\co

%% Calculate Rosenthal potential and set gamma, A, B, Q, R, P.
gamma = 0.5;
A = gamma*[1/2  0   1/2;
           1/2 1/2   0 ;
           0   1/2  1/2];
% A = gamma*[1  0   1/2  0;
%            0  1   0   0;
%            0   1/2  1/2  0;
%            0    0    0   1];
B = (1-gamma)*eye(3); 

% Old Rosenthal potential
Q =     [2/3  0  0;
         0  2/3  0;
         0   0  2/3];

% % Rosenthal with x penalty 
% Q = [0.5*Alpha + mu*ones(3)  0.5*beta_bar; 
%      0.5*beta_bar'           -mu*d^2]    ; 

R = zeros(3);
% A = gamma*[1/3 1/3 1/3  0;
%            1/3 1/3 1/3  0;
%            1/3 1/3 1/3  0;
%            0   0   0   1] ;

%% Solve for P matrix
P = Q ; 
steps = 0 ;
max_steps = 100000 ; 
err = 0.00000003 ; 

while steps < max_steps
    newP = Q + A'*P*A - A'*P*B*((B'*P*B + R)\B'*P*A) ; 
    if all(abs(P-newP) <= err)
        disp("P has converged!")
        break
    end
    P = newP ; 
    steps = steps + 1;
end


%% Solve for control
K_inf = (B'*P*B + R)\(B'*P*A) ;   % where u = -K*x_t

%% Simulate
flow_init = [0.30 0.05 0.65] ;  

% Initialize data stores
xt = [flow_init]' ; 

steps = 0;
while steps < 5
    % Choose action
    ut = -K_inf*(xt) ; 
    % Step env
    xt = A*(xt) + B*ut;
     % x = [x, xt + x_nash] 
    if all(abs(xt - x_nash) <= err)
        disp("Flow has converged to Nash!")
        break
    end
    steps = steps + 1 ; 
end

disp("Final x")
xt

%%%%%%%%%%%%
% If A, B, Q is doubly stochastic, constraint on u should be satisfied
% We picked coefficients of cost functions such that Q is doubly stochastic
% We subtract [x_nash 1] from our state. Gives us the nash in no time. 
%%%%%%%%%%%

%% Graphing results
disp("Begin plotting")

figure(1) 
hold on
subplot(2,1,1)
plot(x(1, :),'-o', 'MarkerFaceColor',[.49 1 .63],'MarkerSize', 7)
plot(x(2, :),'-o', 'MarkerFaceColor',[.49 1 .63],'MarkerSize', 7)
plot(x(3, :),'-o', 'MarkerFaceColor',[.49 1 .63],'MarkerSize', 7)
title("Path Flows")
legend("Path A", "Path B", "Path C")

% subplot(2,1,1)
% plot();
% title("Latencies")
% legend()
hold off
