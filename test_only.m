function accuracy=test_only(per,deepnet)
ii=0;
for i=125:-1:round((125/100)*per)
    ii=ii+1;
    path=strcat(pwd,'\Healthy\',num2str(i),'.png');
    data_gray{ii}=imread(path);
    class{ii}='Normal';
end 
jj=0;
for j=131:-1:round((131/100)*per)
    jj=jj+1;
    path=strcat(pwd,'\Not_Healthy\',num2str(j),'.png');
    data_gray{ii+jj}=imread(path);
    class{ii+jj}='Abnormal';
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
im_type = deepnet(X);
figure,plotconfusion(T,im_type);
figure,plotroc(T,im_type);
for i=1:size(im_type,2)
    if(round(im_type(1,i))==0 && round(im_type(2,i))==1)
        detected{i}='Normal';
    else
        detected{i}='Abnormal';
    end
end
M=confusionmat(detected,class);
accuracy=(sum(sum(M.*eye(size(M))))/size(class,2))*100;