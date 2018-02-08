function x = mnist_forward(net, input, labels)
    var = input;
    x = {input};
    
    for i = 1:numel(net)
        if isa(net{i}, 'Softmaxloss')
            var = net{i}.forward(var, labels);
        else
            var = net{i}.forward(var);
        end

        x{end+1} = var;
    end
end

