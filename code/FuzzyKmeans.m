
clear;clc;

tif=double(imread('whu.tif'));
tif_size=size(tif);
N=tif_size(1)*tif_size(2);      % 像素数
features=tif_size(3);           % 特征维数
tif=reshape(tif,N,features);    % 特性指标矩阵
num_type=4;                     % 分类的类别数
e=0.01;
m=2;
t=0; % 迭代次数



tic
[center, U, obj_fcn, e]=FCM(tif,4);
toc
[~,type]=max(U,[],1);



% 按照最终分类结果构建结果图像
colors_R=[0,255,0,160,0];
colors_G=[255,0,0,32,0];
colors_B=[0,0,255,240,0];
type=reshape(type,tif_size(1),tif_size(2));
new_tif=cat(3,colors_R(type),colors_G(type),colors_B(type));
new_tif=uint8(new_tif);

% 绘制结果
figure,imshow(new_tif)
figure,plot(e,'LineWidth',2,'Marker','*')




function [center, U, obj_fcn, e] = FCM(data, cluster_n, options)

    % 输入：
    % data ---- n*m矩阵,表示n个样本,每个样本具有m维特征值
    % cluster_n ---- 标量,表示聚合中心数目,即类别数
    % options ---- 4*1列向量，其中
    % options(1): 隶属度矩阵U的指数，>1(缺省值: 2.0)
    % options(2): 最大迭代次数(缺省值: 100)
    % options(3): 隶属度最小变化量,迭代终止条件(缺省值: 1e-5)
    % options(4): 每次迭代是否输出信息标志(缺省值: 0)
    % 输出：
    % center ---- 聚类中心
    % U ---- 隶属度矩阵
    % obj_fcn ---- 目标函数值

    % 初始化initialization
    % 输入参数数量检测
    
    if nargin ~= 2 && nargin ~= 3 %判断输入参数个数只能是2个或3个
        error('Too many or too few input arguments!');
    end
    
    data_n = size(data, 1); % 求出data的第一维(rows)数,即样本个数
    data_m = size(data, 2); % 求出data的第二维(columns)数，即特征属性个数
    
    % 设置默认操作参数
    default_options = ...
    [2; % 隶属度矩阵U的指数
    100; % 最大迭代次数 
    1e-3; % 隶属度最小变化量,迭代终止条件
    1]; % 每次迭代是否输出信息标志
    
    if nargin == 2
        % 如果输入参数个数是二那么就调用默认的option
        options = default_options;
    else
        % 如果用户给的opition数少于4个那么其他用默认值
        if length(options) < 4
            tmp = default_options;
            tmp(1:length(options)) = options;
            options = tmp;
        end
        % 检测options中是否有nan值
        nan_index = find(isnan(options)==1);
        % 将denfault_options中对应位置的参数赋值给options中不是数的位置.
        options(nan_index) = default_options(nan_index);
        % 如果模糊矩阵的指数小于等于1，给出报错
        if options(1) <= 1,
            error('The exponent should be greater than 1!');
        end
    end
    
    % 将options中的分量分别赋值给四个变量
    expo = options(1); % 隶属度矩阵U的指数
    max_iter = options(2); % 最大迭代次数
    min_impro = options(3); % 隶属度最小变化量,迭代终止条件
    display = options(4); % 每次迭代是否输出信息标志
    obj_fcn = zeros(max_iter, 1); % 初始化输出参数obj_fcn
    U = initfcm(cluster_n, data_n); % 初始化模糊分配矩阵,使U满足列上相加为1
    

    % Main loop 主要循环
    e=[];
    for i = 1:max_iter
        % 在第k步循环中改变聚类中心ceneter,和分配函数U的隶属度值;
        [U, center, obj_fcn(i)] = stepfcm(data, U, cluster_n, expo);
        if display
            fprintf('FCM:Iteration count = %d, obj.fcn = %f\n', i, obj_fcn(i));
        end
    
        % 终止条件判别
        if i > 1 && abs(obj_fcn(i) - obj_fcn(i-1)) <= min_impro
            break;
        end
        if i>1
            e=cat(1,e,abs(obj_fcn(i) - obj_fcn(i-1)));
        end
    end
    
    iter_n = i; % 实际迭代次数
    obj_fcn(iter_n+1:max_iter) = [];
end


%% initfcm子函数
function U = initfcm(cluster_n, data_n)
    % 初始化fcm的隶属度函数矩阵
    % 输入:
    % cluster_n ---- 聚类中心个数
    % data_n ---- 样本点数
    % 输出：
    % U ---- 初始化的隶属度矩阵
    U = rand(cluster_n, data_n);
    col_sum = sum(U);
    U = U./col_sum(ones(cluster_n, 1), :);
end

%% stepfcm子函数
function [U_new, center, obj_fcn] = stepfcm(data, U, cluster_n, expo)
    % 模糊C均值聚类时迭代的一步
    % 输入：
    % data ---- n*m矩阵,表示n个样本,每个样本具有m维特征值
    % U ---- 隶属度矩阵
    % cluster_n ---- 标量,表示聚合中心数目,即类别数
    % expo ---- 隶属度矩阵U的指数
    % 输出：
    % U_new ---- 迭代计算出的新的隶属度矩阵
    % center ---- 迭代计算出的新的聚类中心
    % obj_fcn ---- 目标函数值
    mf = U.^expo; % 隶属度矩阵进行指数运算结果
    center = mf*data./((ones(size(data, 2), 1)*sum(mf'))'); % 新聚类中心
    dist = distfcm(center, data); % 计算距离矩阵
    obj_fcn = sum(sum((dist.^2).*mf)); % 计算目标函数值
    tmp = dist.^(-2/(expo-1));
    U_new = tmp./(ones(cluster_n, 1)*sum(tmp)); % 计算新的隶属度矩阵
end

%% distfcm子函数
function out = distfcm(center, data)
    % 计算样本点距离聚类中心的距离
    % 输入：
    % center ---- 聚类中心
    % data ---- 样本点
    % 输出：
    % out ---- 距离
    out = zeros(size(center, 1), size(data, 1));
    for k = 1:size(center, 1) % 对每一个聚类中心
        % 每一次循环求得所有样本点到一个聚类中心的距离
        out(k, :) = sqrt(sum(((data-repmat(center(k,:),size(data,1),1)).^2)',1));
    end
end

















