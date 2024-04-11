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

precision=sum(diag(ConfusionMatrix))/(100*NumOfROIs)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 对整张图像进行分类

tif=imread("whu.tif");
Info=imfinfo("whu.tif");

test_size=1024;
tif=tif(1:test_size,1:test_size,:);
Size=[test_size,test_size];

imshow(cat(3,tif(:,:,6),tif(:,:,4),tif(:,:,3)))
figure,imshow(imread("envi_r2.tif"))
 
tif_2=double(reshape(tif(:),Size(1)*Size(2),n));


tif_r2=[];
for i=1:NumOfROIs
    mean_2=repmat(means{i}',Size(1)*Size(2),1);

    dpix=tif_2-mean_2;

    type_r2=sum((dpix/covs{i}).*dpix,2);

    tif_r2=cat(2,tif_r2,type_r2);
end


[~,type]=min(tif_r2,[],2);
classify=reshape(type,Size(1),Size(2));

colors_R=[0,255,0,160];
colors_G=[255,0,0,32];
colors_B=[0,0,255,240];

new_tif=cat(3,colors_R(classify),colors_G(classify),colors_B(classify));
new_tif=uint8(new_tif);
figure,imshow(new_tif)



clear begin_index end_index i ans


