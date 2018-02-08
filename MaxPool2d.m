classdef MaxPool2d < handle
    properties
        pool
        stride
        pad
    end
    
    methods
        function self = MaxPool2d(pool, stride, pad)
            self.pool = pool;
            self.stride = stride;
            self.pad = pad;
        end
        
        function output = forward(self, input)
            output = vl_nnpool(input, self.pool, ...
                'pad', self.pad, 'stride', self.stride, 'method', 'max') ;
        end
        
        function grad_input = backward(self, input, grad_output)
            grad_input = vl_nnpool(input, self.pool, grad_output, ...
                'pad', self.pad, 'stride', self.stride, 'method', 'max');
        end

    end
end

