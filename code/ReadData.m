clear;clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 设置

% ENVI导出的文件名
filename="whu_roi.txt";
% 波段数
n=6;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 处理ENVI导出的文件

% 打开文件
fid=fopen(filename);
% 读取头行
fgetl(fid);
% 读取分类数
NumOfROIs_line=fgetl(fid);
NumOfROIs=str2double(NumOfROIs_line(19));
% 读取图像大小
size_line=fgetl(fid);
Size(1)=str2double(size_line(19:22));
Size(2)=str2double(size_line(26:29));
% 读取每个类别像素的数量
NumOfPerROIs=zeros(NumOfROIs,1);
for i=1:NumOfROIs
    fgetl(fid);fgetl(fid);fgetl(fid);
    line=fgetl(fid);
    NumOfPerROIs(i)=str2double(line(13:length(line)));
end
% 关闭文件
fclose(fid);
% 读取矩阵数据
data=readmatrix(filename);
clear i line size_line fid filename NumOfROIs_line
end_index=0;
new_data=cell(1,NumOfROIs);
for i=1:NumOfROIs
    begin_index=end_index+1;
    end_index=begin_index+NumOfPerROIs(i)-1;
    new_data{i}=data(begin_index:end_index,8:(8+n-1));
end
data=new_data;
clear begin_index end_index i ans new_data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 保存结果
save("whu.mat")
