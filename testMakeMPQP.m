%% Test MPQP Functions

%% Test makeMPQPobjective
A = [1 1;
     1 1];
B = [0;
     1];
nX = size(A,2);
nU = size(B,2);
Q = eye(nX);
P = 2*eye(nX);
R = eye(nU);
N = 2;
[Q_mp, F_mp, c_mp, Y_mp] = makeMPQPobjective(Q,R,P,N,A,B,nX,nU) ; 
check_Qmp = isequal(Q_mp, [12 4; 4 6]) ; 
check_Fmp = isequal(F_mp, [18 18; 8 8]) ; 
check_cmp = isequal(c_mp, 0) ; 
check_Ymp = isequal(Y_mp, [38 36; 36 38]) ; 

disp("Objective: Test 1 --- ")
fprintf("Correct Q_mp: %u\n", check_Qmp)
fprintf("Correct F_mp: %u\n", check_Fmp)
fprintf("Correct c_mp: %u\n", check_cmp)
fprintf("Correct Y_mp: %u\n\n", check_Ymp)

%% Test makeMPQPconstraints
nX = 2;
nU = 1;
N = 2;
A = eye(nX);
B = [0;
     1] ; 
Au = 1;
bu = -1 ; 
Ax = [1 1] ; 
bx = 1;

[G,W,S] = makeMPQPconstraints(Au,bu,Ax,bx,A,B,nX,nU,N);
check_G = isequal(G, [1 0; 0 1; 0 0; 1 0; 1 1]);
check_W = isequal(W, [-1; -1; 1; 1; 1]);
check_S = isequal(S, [0 0; 0 0; -1 -1; -1 -1; -1 -1]);

disp("Constraints: Test 1 --- ")
fprintf("Correct G: %u\n", check_G)
fprintf("Correct W: %u\n", check_W)
fprintf("Correct S: %u\n\n", check_S)




