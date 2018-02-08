classdef Softmaxloss < handle
    
    methods
        function self = Softmaxloss()
        end
        
        function output = forward(self, input, label)
            output = vl_nnsoftmaxloss(input, label) ;
        end
        
        function grad_input = backward(self, input, label, grad_output)
            grad_input = vl_nnsoftmaxloss(input, label, grad_output) ;
        end
    end
end

