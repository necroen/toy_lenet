classdef ReLU < handle
    properties
        leak;
    end
    
    methods
        function self = ReLU(leak)
            self.leak = leak;
        end
        
        function output = forward(self, input)
            output = vl_nnrelu(input, [], self.leak{:}) ;
        end
        
        function grad_input = backward(self, input, grad_output)
            grad_input = vl_nnrelu(input, grad_output, self.leak{:}) ;
        end

    end
end

