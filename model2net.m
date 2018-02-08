function [net net_parameter] = model2net(varargin)
    net = {};  
    for i = 1:nargin
        net{end+1} = varargin{i};
    end
    
    net_parameter = {}; % �� net �г�ȡ����ѵ�������õ� net_parameter
    for i = 1:numel(net)
        if isa(net{i}, 'Conv2d') % ����ֻ��conv���д�ѵ���� w �� b������ģ�����Ͷ�û��
            net_parameter{end+1} = net{i}.w;
            net_parameter{end+1} = net{i}.b;
        end
    end
    
end

