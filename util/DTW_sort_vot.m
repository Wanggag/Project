clc
clear
%-----------------------------------------------------------
% ����˵�����ȸ��ݻ�ȡ�������ߣ����Զ������ȡ������������ѡ�����Ը�ǿ��
% ��ȡ���������������������
% ͶƱ�ķ���
% ȥ�����2�����6
%-----------------------------------------------------------
load ('./feature_DANN/HI_DANN.mat','HI');
load('./feature_DANN/y2_1.mat','y2_1');
load('./feature_DANN/y2_4.mat','y2_4');
load('./feature_DANN/y2_6.mat','y2_6');
load('./Data/Sample_B2.mat');

VoteNum=size(y2_1,2); %�õ�Fea_1�����ᣨ41ά������
FeaDim=VoteNum;

MaxLen=floor(max(X1)); %y=floor(x)������x��Ԫ��ȡ����ֵyΪ�����ڱ�����������
MLData=interp1(X1,Y1,1:MaxLen,'linear');
%��ֵ���� y=interp1(x,y,xi,method)��֪����������x,y,��xi���ĺ���ֵyi����ֵ������method��

ChangePoint=[59,45,87,84,85,82,91];
% RUL_GetSet.T_FTP=[871,743,1943,736,2297,684,221];

Data1{1}=double(y2_1(ChangePoint(1):end,:));
Data1{2}=double(y2_4(ChangePoint(2):end,:));
Data1{3}=double(y2_6(ChangePoint(3):end,:));
% Data1{4}=double(Fea_4(ChangePoint(4):end,:));
% Data1{5}=double(Fea_5(ChangePoint(5):end,:));
% Data1{6}=double(Fea_6(ChangePoint(6):end,:));
% Data1{7}=double(Fea_7(ChangePoint(7):end,:));

%Data1=Fea_Smooth(Data1);

clear y2_1 y2_4 y2_6

Fea_Val=zeros(3,FeaDim);

MLData=datanorm(MLData',MLData');
MLData=MLData(7:end);
for Num1=1:3
    Data=datanorm(Data1{Num1},Data1{Num1});
    for Num2=1:FeaDim
        SM = simmx(MLData',Data1{Num1}(:,Num2));
        [p,q,C] = dpfast(SM);
        Fea_Val(Num1,Num2)=C(size(C,1),size(C,2));
    end
end

[AA,Fea_Tag]=sort(Fea_Val,2);
%% ����ͶƱѡ��
MatchNum=zeros(1,FeaDim);% weight
Fea_Ind=Fea_Tag(:,1:VoteNum);

for Num1=1:FeaDim%��������ѭ�� �α�
    
    for Num2=1:3%7����е�ѭ�� �� 
        for Num3=1:VoteNum%ͶƱ����ѭ�� λ��������
            if Num1==Fea_Ind(Num2,Num3)% Ѱ������Ϊ��һά��Ѱ����ȵ�ֵ
                Weight=FeaDim-Num3;%Num3Խ��ǰ��Num3ԽС��Ȩ��Խ��
                MatchNum(Num1)=MatchNum(Num1)+Weight;
                break;
            end
        end
    end
    
end
% for Num1=1:FeaDim%��������ѭ�� �α�
%     for Num2=1:7%5����е�ѭ��
%         for Num3=1:VoteNum%ͶƱ����ѭ��
%             if Num1==Fea_Ind(Num2,Num3)%Ѱ����ȵ�ֵ
%                 Weight=FeaDim-Num3;%Num3Խ��ǰ��Num3ԽС��Ȩ��Խ��
%                 MatchNum(Num1)=MatchNum(Num1)+Weight;
%             end
%         end
%     end
% end
[Val,FeaInd]=sort(MatchNum,'descend');

% FeaInd
bar(MatchNum)

% save('./Data/FeaInd_B2_46','FeaInd','VoteNum','Val');
save FeaInd_B1_13 FeaInd VoteNum Val

