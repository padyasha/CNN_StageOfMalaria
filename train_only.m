function deepnet=train_only(per)
for i=1:round((125/100)*per)
    path=strcat(pwd,'\Healthy\',num2str(i),'.png');
    data_gray{i}=imread(path);
    class{i}='Normal';
end 
for j=1:round((131/100)*per)
    path=strcat(pwd,'\Not_Healthy\',num2str(j),'.png');
    data_gray{i+j}=imread(path);
    class{i+j}='Abnormal';
end

r=50;c=50;
final_data_re=zeros(r*c+1,size(class,2));
for i=1:size(class,2)
    if(strcmp(class{i},'Normal'))
        final_data_re(1,i)=0;
        im=imresize(data_gray{i},[r c]);
        final_data_re(2:end,i)=im2double(im(:));
    else
        final_data_re(1,i)=1;
        im=imresize(data_gray{i},[r c]);
        final_data_re(2:end,i)=im2double(im(:));
    end
end    

X=final_data_re(2:end,:);
T=zeros(2,size(final_data_re,2));
T(1,:)=final_data_re(1,:);
T(2,:)=~T(1,:);
size(T)
hiddenSize = 100;
autoenc1 = trainAutoencoder(X,hiddenSize,'L2WeightRegularization',0.001,'SparsityRegularization',4,'SparsityProportion',0.05,'DecoderTransferFunction','purelin');
features1 = encode(autoenc1,X);
hiddenSize = 50;
autoenc2 = trainAutoencoder(features1,hiddenSize,'L2WeightRegularization',0.001,'SparsityRegularization',4,'SparsityProportion',0.05,'DecoderTransferFunction','purelin','ScaleData',false);
features2 = encode(autoenc2,features1);
softnet = trainSoftmaxLayer(features2,T,'LossFunction','crossentropy');
deepnet = stack(autoenc1,autoenc2,softnet);
deepnet = train(deepnet,X,T);
