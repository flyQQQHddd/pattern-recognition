
% 脚本名：ANN.m
% 描述：人工神经网络
% 编码：utf-8
% 测试环境：MATLAB R2022b
% 作者：曲浩栋
% 学号：2021302131044
% 单位：武汉大学遥感信息工程学院-空间信息与数字技术
% 课程名：计算机视觉与模式识别（模式识别部分）
% 最后修订时间：2023-05-28


clear;clc;

roi=load("whu.mat");

roidata=[];
for i=1:roi.NumOfROIs
    roidata=[roidata;roi.data{i},i*ones(roi.NumOfPerROIs(i),1)];
end

% 注：至此处理过后的roidata为(M*N)行(bands+1)列的二维矩阵
%     bands为波段数
%     roidata前6列为6个波段的值，第七列为类别标签（1~4）

% 定义常数
N=size(roidata,1);        % 样本个数
bands=size(roidata,2)-1;  % 波段数
class=4;                  % 类别数
cell1=10;                 % 第一层神经元个数
cell2=15;                 % 第二层神经元个数
learning_rate=0.001;      % 学习率
numEpoch=100;             % 训练次数
e_line=zeros(numEpoch);   % 交叉熵总和曲线

% 数据归一化
[roidata(:,1:bands),means,stds]=zscore(roidata(:,1:bands));

% 将数据打乱
randIndex=randperm(N);
roidata=roidata(randIndex,:);

% 将数据输入向量与标签分开（标签处理成one-hot向量）
roidata_x=roidata(:,1:6);
roidata_y=zeros(N,class);
for i=1:N
    roidata_y(i,roidata(i,bands+1))=1;
end

% 划分训练集、验证集，测试集
ratioTraining=0.7;      % 训练数据集比例
redioValidation=0.15;   % 验证数据集比例
radioTesting=0.15;      % 测试数据集比例

numTraining=int32(N*ratioTraining);
numValidation=int32(N*redioValidation);
numTesting=N-numTraining-numValidation;

dataset_train_x=roidata_x(1:numTraining,:);
dataset_train_y=roidata_y(1:numTraining,:);

dataset_validation_x=roidata_x(numTraining+1:numTraining+numValidation,:);
dataset_validation_y=roidata_y(numTraining+1:numTraining+numValidation,:);

dataset_test_x=roidata_x(numTraining+numValidation+1:end,:);
dataset_test_y=roidata_y(numTraining+numValidation+1:end,:);

% 两层隐含层

W1 = 2*rand(cell1,bands)-1;
W2 = 2*rand(cell2,cell1)-1;
W3 = 2*rand(class,cell2)-1;
for epoch=1:numEpoch

    % 使用训练集进行训练
    for i=1:numTraining
        x=dataset_train_x(i,:)';
        d=dataset_train_y(i,:)';
        % 前向传播
        % 隐含层第一层
        v1=W1*x;
        y1=sigmoid(v1);
        % 隐含层第二层
        v2=W2*y1;
        y2=sigmoid(v2);
        % 输出层
        v=W3*y2;
        y=softmax(v);

        % 反向传播（梯度下降法）
        % 输出层
        e=y-d;
        delta=e;
        % 隐含层第二层
        e2=W3'*delta;
        delta2=sigmoid_derivative(v2).*e2;
        % 隐含层第一层
        e1=W2'*delta2;
        delta1=sigmoid_derivative(v1).*e1;

        % 改正W
        W1=W1-learning_rate*delta1*x';
        W2=W2-learning_rate*delta2*y1';
        W3=W3-learning_rate*delta*y2';
    end

    % 使用验证集进行验证
    sum_e=0;
    for i=1:numValidation
        % 输入及标签
        x=dataset_validation_x(i,:)';
        d=dataset_validation_y(i,:)';
        % 隐含层第一层
        v1=W1*x;
        y1=sigmoid(v1);
        % 隐含层第二层
        v2=W2*y1;
        y2=sigmoid(v2);
        % 输出层
        v=W3*y2;
        y=softmax(v);
        % 计算误差
        sum_e=sum_e+error(y,d);
    end
    % 输出信息
    fprintf("epoch=%d\n",epoch)
    fprintf("error=%d\n",sum_e)
    % 打乱数据
    randIndex=randperm(numTraining);
    dataset_train_x=dataset_train_x(randIndex,:);
    dataset_train_y=dataset_train_y(randIndex,:);
    % 记录误差熵总和
    e_line(epoch)=sum_e;
end

% 绘制验证集的交叉熵总和曲线
hold on
plot(e_line)
ylabel("验证集上的交叉熵总和")
xlabel("训练次数")
hold off

% 使用测试集对模型进行评估
confusionMatrix=zeros(class,class);
for i=1:numTesting
    x=dataset_test_x(i,:)';
    d=dataset_test_y(i,:)';
    % 隐含层第一层
    v1=W1*x;
    y1=sigmoid(v1);
    % 隐含层第二层
    v2=W2*y1;
    y2=sigmoid(v2);
    % 输出层
    v=W3*y2;
    y=softmax(v);
    [~,rst]=max(y);
    [~,real]=max(d);
    confusionMatrix(rst,real)=confusionMatrix(rst,real)+1;
end

% 输出测试集测试结果
precision=sum(diag(confusionMatrix))/sum(confusionMatrix,"all");
fprintf("混淆矩阵：\n")
disp(confusionMatrix)
fprintf("整体精度：%f\n",precision)

% 对TM影像进行分类
tif=imread("whu.tif");
[M,N,bands]=size(tif);
tif=double(reshape(tif,M*N,bands));
tif=(tif-means)./stds;
class=zeros(M*N,1);

for i=1:M*N
    % 输入层
    x=tif(i,:)';
    % 隐含层第一层
    v1=W1*x;
    y1=sigmoid(v1);
    % 隐含层第二层
    v2=W2*y1;
    y2=sigmoid(v2);
    % 输出层
    v=W3*y2;
    y=softmax(v);
    % 记录分类结果
    [~,class(i)]=max(y);
end
class=reshape(class,M,N);

% 按照最终分类结果构建结果图像
colors_R=[0,255,0,160,0];
colors_G=[255,0,0,32,0];
colors_B=[0,0,255,240,0];
new_tif=cat(3,colors_R(class),colors_G(class),colors_B(class));
new_tif=uint8(new_tif);
figure,imshow(new_tif)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 以下是定义的函数

% sigmoid激活函数
function y = sigmoid(x)
    y=1./(1+ exp(-x));
end

% sigmoid激活函数的导数
function y = sigmoid_derivative(x)
    y = sigmoid(x).*(1-sigmoid(x));
end

% ReLU激活函数（没有用）
function y=ReLU(v)
    y=max(0,v);
end

% 交叉熵
function j=error(y,d)
    j=sum(-d.*log(y));
end

% softmax
function y=softmax(v)
   v=v-max(v);
   y=exp(v)./sum(exp(v)); 
end

% 标准化函数
function [rst,means,stds]=zscore(x)
    means=mean(x);
    stds=std(x);
    rst=(x-repmat(means,size(x,1),1))./repmat(stds,size(x,1),1);
end