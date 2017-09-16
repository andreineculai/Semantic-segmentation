clear all;
setConstants
setMap
% read groundplane equation from text file
gpFile = fopen(groundPlaneEqFile);
GPs = cell2mat(textscan(gpFile, '%f %f %f %f'));
fclose(gpFile);
totalSP = 0;
%features vectors
for frame=22:22
    frameLeft = imread(strcat(dirImages,'/','image (', num2str(frame),').png'));
    frameDisp = imread(strcat(dirDisparity,'/','disp (', num2str(frame),').png'));
    lImg = imread(strcat(dirLabels, '/','gt (', num2str(frame),').png'));
    [heightImg, widthImg,~] = size(frameLeft); %size of image
    frameDisp(frameDisp < 5) = 0;
    labelsImg = lImg;
    %segment image in superpixels using slicmex function
    
    numberOfSuperpixels = fix(heightImg*widthImg/100)+1;
    [labelImg, numberOfSuperpixels] = slicmex(frameLeft,numberOfSuperpixels,20);
%     sizeSP = 10;
%     numberOfSuperpixels = ceil(widthImg / sizeSP) * ceil(heightImg / sizeSP);
%     labelImg = zeros(heightImg, widthImg);
%     for h=1:heightImg
%         for w=1:widthImg
%             labelImg(h, w) = floor((h-1)/sizeSP) * ceil(widthImg / sizeSP) + ceil(w / sizeSP);
%         end
%     end
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
    labels = zeros(numberOfSuperpixels,1,'uint8');
    %assign labels from labelled image
    labelsImg = rgb2ind(labelsImg, MAP);
    for i=1:numberOfSuperpixels
        indices = labelImg == i;
        pixels = labelsImg(indices);
        pixels = pixels(pixels ~= 11);
        if ~isempty(pixels)
            labels(i) = mode(pixels);
        else
            labels(i) = 11;
        end
    end
    isValid (labels == 11) = 0;
    totalNormal(totalSP+1:totalSP+numberOfSuperpixels,1) = nalpha;
    totalGPdist(totalSP+1:totalSP+numberOfSuperpixels,1) = heightAbGr;
    totallp(totalSP+1:totalSP+numberOfSuperpixels,1) = localPl;
    totalnp(totalSP+1:totalSP+numberOfSuperpixels,1) = neighborPl;
    totalcp(totalSP+1:totalSP+numberOfSuperpixels,1) = cameraPath;
    totalLabels(totalSP+1:totalSP+numberOfSuperpixels,1) = labels;
    totalValid(totalSP+1:totalSP+numberOfSuperpixels,1) = isValid;
    totalSP = totalSP + numberOfSuperpixels;
    frame
end
% compute model
totalNormal = totalNormal(logical(totalValid));
totalGPdist = totalGPdist(logical(totalValid));
totallp = totallp(logical(totalValid));
totalnp = totalnp(logical(totalValid));
totalcp = totalcp(logical(totalValid));
totalLabels = totalLabels(logical(totalValid));
numberOfValidentries = sum(totalValid);
%feature matrix
X_train = [totalNormal totalGPdist totallp totalnp totalcp];
%normalize feature
nX = sqrt(sum(X_train.^2,2));
nX(nX == 0) = 1;
nX_train = bsxfun( @rdivide, X_train, nX );
options = struct('DEBUG_ON', 1);
model = classRF_train(nX_train, double(totalLabels), 80, floor(sqrt(size(nX_train,2))), options);
save -v7.3 modelVERealwithLabelsTest.mat model;