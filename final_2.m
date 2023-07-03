clear
close all
clc

per=70;
deepnet=train_only(per);
save deepnet deepnet
sample_image_folder=pwd;
[filename,user_canceled] = imgetfile('InitialPath',sample_image_folder);
test_data=im2double(imresize(imread(filename),[50 50]));
X(:,1)=test_data(:);
im_type = deepnet(X);
if(max(im_type)==1)
    disp('Healthy');
else
    disp('Infected');
end
