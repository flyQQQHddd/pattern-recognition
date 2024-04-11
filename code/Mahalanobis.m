
% 脚本名：Mahalanobis.m
% 描述：mashijul
% 编码：utf-8
% 测试环境：MATLAB R2022b
% 作者：曲浩栋
% 学号：2021302131044
% 单位：武汉大学遥感信息工程学院-空间信息与数字技术
% 课程名：计算机视觉与模式识别（模式识别部分）
% 最后修订时间：2023-05-28

clear;clc;
load("whu.mat")

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 划分训练集和测试集
test=[];
training=cell(1,NumOfROIs);
for i=1:NumOfROIs
    test=cat(1,test,cat(2,data{i}(1:100,:),ones(100,1)*i));
    training{i}=data{i}(101:end,:);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 使用训练集求各类别的均值向量以及协方差矩阵
means=cell(1,NumOfROIs);
covs=cell(1,NumOfROIs);
for i=1:NumOfROIs
    means{i}=mean(training{i})';
    covs{i}=cov(training{i});
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 计算测试集的马氏距离
r2=zeros(400,NumOfROIs);
for i=1:NumOfROIs
    du=(test(:,1:n)-repmat(means{i}',100*NumOfROIs,1))';
    inv_cov=pinv(covs{i});
    r2(:,i)=sum((du'*inv_cov.*du'),2);
end
[~,classify]=min(r2,[],2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 使用混淆矩阵进行精度评定

ConfusionMatrix=zeros(NumOfROIs,NumOfROIs);
for i=1:100*NumOfROIs
    ConfusionMatrix(test(i,end),classify(i))=ConfusionMatrix(test(i,end),classify(i))+1;
end
precision=sum(diag(ConfusionMatrix))/(100*NumOfROIs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 对整张图像进行分类

% 读取原始影像
tif=imread("whu.tif");
[M,N,bands]=size(tif);

% 假彩色增强显示原始图像
imshow(cat(3,tif(:,:,6),tif(:,:,4),tif(:,:,3)))
tif=double(reshape(tif, M*N,[]));

% 计算各类别马氏距离并分类
r2_s=[];
for i=1:NumOfROIs
    dpix=tif-means{i}';
    r2=sum((dpix/covs{i}).*dpix,2);
    r2_s=cat(2,r2_s,r2);
end

% 根据类别赋予不同的颜色
[~,type]=min(r2_s,[],2);
classify=reshape(type,M,N);

colors_R=[0,255,0,160];
colors_G=[255,0,0,32];
colors_B=[0,0,255,240];

new_tif=cat(3,colors_R(classify),colors_G(classify),colors_B(classify));
new_tif=uint8(new_tif);
figure,imshow(new_tif)



clear begin_index end_index i ans


