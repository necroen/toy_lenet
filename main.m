% 这个repo就是写着玩的，好多参数被我定死了
clear all;clc;

run('vl_setupnn.m');
clc;

% matconvnet 版本为 beta25
% imdb.mat 是 getMnistImdb 产生的 mat 文件
% 包含70000个样本, mat 文件大概 107 MB
imdb_path = 'D:\mat_file\matconvnet25\data\mnist-baseline-simplenn\imdb.mat';
imdb = load(imdb_path) ;

clear imdb_path;
%%
rng('default');
rng(0) ;
f=1/100 ;

% 将模块组成网络，这些模块在 net 中是有序的，mnist_forward 函数
% 会顺序执行这些模块的 forward 方法。
% 从 net 中抽取出待训练参数得到 net_parameter
% 
%  'conv','pool','conv','pool','conv','relu','conv','softmaxloss'
%     1      2     3       4      5      6      7         8   
%     w1           w3            w5            w7
%     b1           b3            b5            b7
%  x1    x2     x3     x4     x5     x6    x7       x8        x9
% dx1   dx2    dx3    dx4    dx5    dx6   dx7      dx8       dx9 

[net net_parameter] = model2net(...
    Conv2d(f*randn(5,5,1,20, 'single'), zeros(1, 20, 'single')), ...
    MaxPool2d([2 2], 2, 0), ...
    Conv2d(f*randn(5,5,20,50, 'single'), zeros(1, 50, 'single')),...
    MaxPool2d([2 2], 2, 0),...
    Conv2d(f*randn(4,4,50,500, 'single'), zeros(1, 500, 'single')),...
    ReLU({}),...
    Conv2d(f*randn(1,1,500,10, 'single'), zeros(1, 10, 'single')),... % 这个conv当做linear用了
    Softmaxloss()...
);
clear f;

% 在pytorch中需要在 model 外定义 loss，这里
% 相当于定义了最后一层为 softmax， loss 定义为 cross entropy loss
% 因为 softmaxloss = softmax + cross entropy loss
%%
solverState = []; % sgd solver state
num = 0 ;
error = [];
errorLabels = {'top1err', 'top5err'} ;
stats = [];
t_table = [];

for epoch = 1:20
    rng(epoch) ;
    
    if isempty(solverState)
        for i = 1:numel(net_parameter)
            solverState{i} = 0 ;
        end
    end
    
    tic;
    for t = 1:256:60000 % batchSize = 256，被调用的函数中好几处还保留着256
        fprintf('epoch:%2d t:%3d\t', epoch, (t-1)/256 );
        
        batchSize = min(256, 60000 - t + 1); % 这个是求余用的，因为最后一个batch可能不足256
        batchStart = t;
        batchEnd = min(t+256-1, 60000) ;
        batch = (batchStart : 1 : batchEnd) ;
        
        num = num + numel(batch) ;
        
        im = imdb.images.data(:,:,:,batch) ;
        labels = imdb.images.labels(1,batch) ;
        
        loss = 1; % 手动给了个loss
        
        x = mnist_forward(net, im, labels);
        % x = forward()
        
        [net_d_parameter dx] = mnist_backward(net, loss, x, labels);
        % [dw dx] = backward()
        
        top_err = error_class(x, labels);
        % accumulate errors
        error = sum(...
        [   error, ...
            [ sum( double( x{end} ) ) ;
             reshape(   top_err  ,[],1) ; ]
        ], 2 ) ;
    
        % sgd 更新参数
        [net_parameter, net_d_parameter, solverState] = ...
            mnist_sgd(net_parameter, net_d_parameter, solverState);
        
        % 参数回填到net中，这里比较简单，只有conv层有待训练参数要回填
        for i = 1:numel(net)
            if isa(net{i}, 'Conv2d')
                %           i = 1   3   5   7
                % net_parameter 1 2 3 4 5 6 7 8
                net{i}.w = net_parameter{i};
                net{i}.b = net_parameter{i+1};
            end
        end
        
        tmp_error = error/num;
        stats.objective = tmp_error(1) ;
        for i = 1:numel(errorLabels)
            stats.(errorLabels{i}) = tmp_error(i+1) ;
        end
        fprintf('\t top1err: %f\n', stats.top1err);
        
    end

    t = toc
    t_table = [t_table, t];
    
end
%%
clear batch* dx epoch error*  i im labels loss num solverState stats t top_err;
%%
% validate
batch = (60001 : 1 : 60010) ;

im_valid = imdb.images.data(:,:,:,batch) ;
labels_valid = imdb.images.labels(1,batch) ;

x_valid = mnist_forward(net, im_valid, labels_valid);

predictions = gather(x_valid{end-1}) ;
[~,predictions] = sort(predictions, 3, 'descend') ;
predictions = squeeze(predictions);

prob = vl_nnsoftmax(x_valid{end-1});
prob = squeeze(prob);