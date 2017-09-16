setConstants
setMap
gpFile = fopen(groundPlaneEqFile);
GPs = cell2mat(textscan(gpFile, '%f %f %f %f'));
fclose(gpFile);
frame = 6;
frameLeft = imread(strcat(dirImages,'/','left_', num2str(frame),'.png'));
frameDisp = imread(strcat(dirDisparity,'/','disp_', num2str(frame),'.png'));
lImg = imread(strcat(dirLabels, '/','label_', num2str(frame),'.png'));
[heightImg, widthImg,~] = size(frameLeft); %size of image
frameDisp(frameDisp < 5) = 0;
labelsImg = lImg;
%segment image in superpixels using slicmex function
numberOfSuperpixels = fix(heightImg*widthImg/100)+1;
[labelImg, numberOfSuperpixels] = slicmex(frameLeft,numberOfSuperpixels,20);
sup_img = labelImg;
sup_img(sup_img==0) = max(max(sup_img));
%compute pointcloud
depthMap = (f*B)*1./double(frameDisp);
V = meshgrid(1:1:heightImg, 1:1:widthImg)';
U = meshgrid(1:1:widthImg, 1:1:heightImg);
Y = (V - c_v).*depthMap/f;
X = (U - c_u).*depthMap/f;
pcloud = [reshape(X', widthImg*heightImg, 1) reshape(Y', widthImg*heightImg, 1) ...
    reshape(depthMap', widthImg*heightImg, 1)];
%compute features
[nalpha, heightAbGr, localPl, neighborPl, cameraPath, isValid] = ...
    getFeatures(frameLeft, sup_img, numberOfSuperpixels, pcloud, GPs(frame+1,:));
%feature matrix
X_train = [nalpha, heightAbGr, localPl, neighborPl];
%normalize feature
nX = sqrt(sum(X_train.^2,2));
nX(nX == 0) = 1;
nX_train = bsxfun( @rdivide, X_train, nX );


load modelVE.mat;
[prediction, v, p] = classRF_predict(nX_train, model);
for i=1:numberOfSuperpixels
   labelImg(labelImg == i) = -prediction(i);
end
labelImg = labelImg * -1;
accuracy(lImg, labelImg);