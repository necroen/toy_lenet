function [net_d_parameter dx] = mnist_backward(net, loss, x, labels)

dx = cell( size(x) ); 
dx{end} = loss;

for i = ( numel(net)-1 ):-1:0
    if isa(net{i+1}, 'Softmaxloss')
        dx{i+1} = net{i+1}.backward(x{i+1}, labels, dx{i+2}); 
    else
        dx{i+1} = net{i+1}.backward(x{i+1}, dx{i+2});  
    end
end

net_d_parameter = {};
for i = 1:numel(net)
    if isa(net{i}, 'Conv2d') 
        net_d_parameter{end+1} = net{i}.dw;
        net_d_parameter{end+1} = net{i}.db;
    end
end

end

