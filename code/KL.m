
% 脚本名：KL.m
% 描述：KL变换
% 编码：utf-8
% 测试环境：MATLAB R2022b
% 作者：曲浩栋
% 学号：2021302131044
% 单位：武汉大学遥感信息工程学院-空间信息与数字技术
% 课程名：计算机视觉与模式识别（模式识别部分）
% 最后修订时间：2023-05-28


clear;clc;

% 读取图像
tif=double(imread("whu.tif"));

% 图像格式处理
% 处理前：1024*1024*6
% 处理后：1048576*6
[M,N,bands]=size(tif);
tif=reshape(tif, [], bands);

% 标准化
tif=(tif-mean(tif))./std(tif);

% 求解协方差矩阵
cov_matrix=cov(tif);

% 求解特征值D和特征向量X
[X,D]=eig(cov_matrix);
D=diag(D);

% 变换
new_tif=tif*X;
new_tif=reshape(new_tif,M,N,[]);

% 展示变化后的6个分量

figure
for i=1:bands
    subplot(2,3,i)
    imshow(new_tif(:,:,bands-i+1),[])
    title("特征值："+num2str(D(bands-i+1)))
end


% 取前K组最大的特征值对应的特征向量
k=3;
[~,index]=sort(D,'descend');
P=X(:,index(1:k));

% 旋转变换
new_tif=tif*P;

% 绘制特征值曲线
figure
hold on
xlabel("特征值序号")
ylabel("特征值（标准化）")
plot(D(index), ...
    LineStyle="-", ...
    Color="r", ...
    LineWidth=2, ...
    Marker="o")
hold off

% 将KL变换得到的三个分量进行假彩色增强输出
figure
imshow(reshape(new_tif,M,N,[]),[])
% plot(new_tif(:,1),new_tif(:,2),"*")






