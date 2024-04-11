
clear;clc;

tif=double(imread('whu.tif'));
tif_size=size(tif);
N=tif_size(1)*tif_size(2);      % 像素数
features=tif_size(3);           % 特征维数
tif=reshape(tif,N,features);    % 特性指标矩阵
num_type=4;                     % 分类的类别数

tic
% 随机选取num_type个起始点
center = tif(randperm(N,num_type)',:);
% 记录迭代次数和误差
t=0;
e=[];
while true
    old_center=center;
    % 分类
    type=classify(tif,center);
    % 计算新的聚类中心
    center=new_center(tif, type, num_type);
    % 计算新的聚类中心与上次聚类中心的误差
    error=norm(center-old_center,'fro');
    e=cat(2,e,error);
    t=t+1;
    if error < 1e-3 || t>100
        break;
    end
end
toc

fprintf('迭代次数：%d\n',t)

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 定义的函数

% 按照最小距离进行分类
function rst=classify(tif, center)
    dist=[];
    for i=1:size(center,1)
        center_i=repmat(center(i,:),size(tif,1),1);
        dpix=tif-center_i;
        dist=cat(2,dist,sum(dpix.^2,2));
    end
    [~,rst]=min(dist,[],2);  
end

% 计算新的聚类中心
function rst=new_center(tif, classify, num_type)
    rst=[];
    for i=1:num_type
        rst=cat(1,rst,mean(tif(classify==i,:)));
    end
end








