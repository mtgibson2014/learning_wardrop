function [G,W,S] = makeMPQPconstraints(Au,bu,Ax,bx,A,B,nX,nU,N)
% Make the matrices that forms the constraints for MPQP
Su = zeros(nX*(N+1),nU*N);
for i=1:N
 Su(nX*i+1:nX*(i+1),1:i*nU) = [A*Su(nX*(i-1)+1:nX*i,1:(i-1)*nU) B];
end
nBU = size(bu,1);
nBX = size(bx,1);


Su_cnstr = zeros((N+1)*nBX,nU*N);
for i=1:N
    for j=1:N
        Su_cnstr(nBX*i+1:nBX*(i+1), (j-1)*nU+1:j*nU) = Ax*Su(nX*i+1:nX*(i+1), (j-1)*nU+1:j*nU);
    end
end

G = [blkdiag(kron(eye(N),Au));
     Su_cnstr];
W = zeros(N*nBU+(N+1)*nBX, 1);
S = zeros(N*nBU+(N+1)*nBX, nX);
for i=1:N
    W((i-1)*nBU+1:i*nBU, :) = bu;
end
for i=N+1:N+N+1
    W((i-1)*nBX+1:i*nBX, :) = bx;
    if i == N+1
        S((i-1)*nBX+1:i*nBX, :) = -Ax;
    else
        S((i-1)*nBX+1:i*nBX, :) = S((i-2)*nBX+1:(i-1)*nBX, :)*A;
    end
end
end

