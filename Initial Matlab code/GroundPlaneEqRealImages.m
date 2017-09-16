%ground plane equation
clc;
clear;
close all;
%Virtual Environment
setConstants
fileID = fopen(groundPlaneEqFile,'w');
for frame = 1:58
    frameLeft = imread(strcat(dirImages,'/','image (', num2str(frame),').png'));
    frameDisp = imread(strcat(dirDisparity,'/','disp (', num2str(frame),').png'));
    gpImg = imread(strcat(dirGP, '/', 'gp (', num2str(frame), ').png'));
    gpImg=rgb2gray(gpImg);
    [heightImg, widthImg,~] = size(frameLeft); %size of image
    %compute pointcloud
    depthMap = (f*B)*1./double(frameDisp);
    V = meshgrid(1:1:heightImg, 1:1:widthImg)';
    U = meshgrid(1:1:widthImg, 1:1:heightImg);
    Y = (V - c_v).*depthMap/f;
    X = (U - c_u).*depthMap/f;
    pcloud = [reshape(double(X'), widthImg*heightImg, 1) reshape(double(Y'), ...
        widthImg*heightImg, 1) reshape(depthMap', widthImg*heightImg, 1) reshape(double(gpImg'), widthImg*heightImg, 1)];
    gplane = pcloud(pcloud(:,4) == 255,:);
    points = gplane(:,1:3);
    points = points(points(:,3)~= Inf & points(:,3) ~= -Inf,:);%& points(:,3) < 10
    [normal,~,point] = affine_fit(points);
    %planeEq = fitplane(points');
    d = dot(normal,-point);
    planeEq = [normal;d]';
    if planeEq(4) < 0
        planeEq(:,:) = planeEq(:,:).*(-1);
    end
    planeEq
    fprintf(fileID,'%f %f %f %f\n',planeEq(1),planeEq(2),planeEq(3),planeEq(4));
end
fclose(fileID);