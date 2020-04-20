% Simulate  Routing Game as infinite-horizon LQR


function cost = ros(edgeCostFuns, edgeFlows)
% edgeCosts: a list of cost functions c( ) for each edge
%                   c = @(flow) (...)
%            length = # of edges
% 
% edgeFlows: a list of edge flows induced by a routing distribution
%            Ex. [.2, .3, ...] --- each number should be 
%            length = # of edges
    nE = length(edgeFlows) ; 
    cost = 0 ; 
    for i=1:nE
        cost = cost + integral(edgeCostFuns(i), 0, edgeFlows(i)) ; 
    end
end

function cost = soc(edgeCostFuns, edgeFlows)
    nE = length(edgeFlows) ; 
    cost = 0 ; 
    for i=1:nE
        c = edgeCostFuns(i) ; 
        cost = cost + edgeFlows(i)*c(edgeFlows(i)) ; 
    end
end