function [net net_parameter] = model2net(varargin)
    net = {};  
    for i = 1:nargin
        net{end+1} = varargin{i};
    end
    
    net_parameter = {}; % 从 net 中抽取出待训练参数得到 net_parameter
    for i = 1:numel(net)
        if isa(net{i}, 'Conv2d') % 这里只有conv中有待训练的 w 和 b，其他模块类型都没有
            net_parameter{end+1} = net{i}.w;
            net_parameter{end+1} = net{i}.b;
        end
    end
    
end

