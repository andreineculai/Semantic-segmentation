setConstants
setMap
% read groundplane equation from text file
gpFile = fopen(groundPlaneEqFile);
GPs = cell2mat(textscan(gpFile, '%f %f %f %f'));
fclose(gpFile);
frame = 22;
frameLeft = imread(strcat(dirImages,'/','image (', num2str(frame),').png'));
frameDisp = imread(strcat(dirDisparity,'/','disp (', num2str(frame),').png'));
lImg = imread(strcat(dirLabels, '/','gt (', num2str(frame),').png'));
[heightImg, widthImg,~] = size(frameLeft); %size of image
frameDisp(frameDisp < 5) = 0;
labelsImg = lImg;
%segment image in superpixels using slicmex function
    numberOfSuperpixels = fix(heightImg*widthImg/100)+1;
    [labelImg, numberOfSuperpixels] = slicmex(frameLeft,numberOfSuperpixels,20);
% sizeSP = 10;
% numberOfSuperpixels = ceil(widthImg / sizeSP) * ceil(heightImg / sizeSP);
% labelImg = zeros(heightImg, widthImg);
% for h=1:heightImg
%     for w=1:widthImg
%         labelImg(h, w) = floor((h-1)/sizeSP) * ceil(widthImg / sizeSP) + ceil(w / sizeSP);
%     end
% end
sup_img = labelImg;
sup_img(sup_img==0) = max(max(sup_img));
%compute pointcloud
depthMap = (f*B)*1./double(frameDisp);
V = meshgrid(1:1:heightImg, 1:1:widthImg)';
U = meshgrid(1:1:widthImg, 1:1:heightImg);
Y = (V - c_v).*depthMap/f;
X = (U - c_u).*depthMap/f;
%CHANGE BACK!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%CHANGE BACK!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%CHANGE BACK!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%CHANGE BACK!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%CHANGE BACK!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%CHANGE BACK!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%CHANGE BACK!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%CHANGE BACK!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%CHANGE BACK!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%CHANGE BACK!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%CHANGE BACK!!!!!!!!!!!!!!!!!!!!!!!!!!!!

pcloud = [reshape(X', widthImg*heightImg, 1) reshape(Y', widthImg*heightImg, 1) ...
    reshape(depthMap', widthImg*heightImg, 1)];
%compute features
[nalpha, heightAbGr, localPl, neighborPl, cameraPath, isValid] = ...
    getFeatures(frameLeft, sup_img, numberOfSuperpixels, pcloud, GPs(frame,:));
%feature matrix
X_train = [nalpha, heightAbGr, localPl, neighborPl cameraPath];
%normalize feature
nX = sqrt(sum(X_train.^2,2));
nX(nX == 0) = 1;
nX_train = bsxfun( @rdivide, X_train, nX );


load modelVERealwithLabelsTest.mat;
[prediction, v, p] = classRF_predict(nX_train, model);
prediction;
v;
p;
for i=1:numberOfSuperpixels
    labelImg(labelImg == i) = -prediction(i);
end
labelImg = labelImg * -1;
lImg = rgb2ind(lImg, MAP);
accuracy(lImg, labelImg);