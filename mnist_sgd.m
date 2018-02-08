function [net_parameter, net_d_parameter, solverState] = mnist_sgd(net_parameter, net_d_parameter, solverState)
thisDecay = 0.0005;
thisLR = 0.001;
batchSize = 256;
momentum = 0.9;

for i = 1:numel(net_parameter)
    parDer = net_d_parameter{i}  ;
    parDer = vl_taccum(1/batchSize, parDer, thisDecay, net_parameter{i}) ;
    
    solverState{i} = ...
        vl_taccum(momentum, solverState{i}, -1, parDer) ;
    
    delta = solverState{i} ;
    
    % Update parameters.
    net_parameter{i} = vl_taccum(1, net_parameter{i}, thisLR, delta) ;
end

end

