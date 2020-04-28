function [Q_mp, F_mp, c_mp, Y_mp] = makeMPQPobjective(Q,R,P,N,A,B,nX,nU)
% Makes the matrices for objective function in mpQP
Q_bar = blkdiag(kron(eye(N),Q),P); 
R_bar = kron(eye(N),R) ; 
Sx = zeros(nX*(N+1),nX);
Su = zeros(nX*(N+1),nU*N);
Sx(1:nX,:) = eye(nX);
for i=1:N
 Sx(nX*i+1:nX*(i+1),:) = A*Sx(nX*(i-1)+1:nX*i,:);
 Su(nX*i+1:nX*(i+1),1:i*nU) = [A*Su(nX*(i-1)+1:nX*i,1:(i-1)*nU) B];
end
% Make Batch params for conversion
H = Su'*Q_bar*Su + R_bar ; 
F = Sx'*Q_bar*Su ; 
Y_batch  = Sx'*Q_bar*Sx ; 
% Batch --> mpQP params
Q_mp = 2*H ; 
F_mp = 2*F';
c_mp = zeros(size(F_mp,1),1) ; 
Y_mp = 2*Y_batch ; 
end