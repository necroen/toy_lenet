classdef Conv2d < handle
    properties
        w
        b
        dw
        db
        pad = 0
        stride = 1
        dilate = 1
    end
    
    methods
        function self = Conv2d(w, b)
            self.w = w;
            self.b = b;
            self.dw = zeros(size(w),'single');
            self.db = zeros(size(b),'single');
        end
        
        function output = forward(self,input)
            output = vl_nnconv(input, self.w, self.b,  ...
                'pad', self.pad, 'stride', self.stride, 'dilate', self.dilate) ;
        end
        
        function grad_input = backward(self, input, grad_output)
            [grad_input dw db] = vl_nnconv(input, self.w, self.b, grad_output, ...
                'pad', self.pad, 'stride', self.stride, 'dilate', self.dilate);
            self.dw = dw;
            self.db = db;
        end
        
    end
end

