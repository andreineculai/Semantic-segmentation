%% returns features for each superpixel
%% surfaceNorm - surface normal
%% nalpha - cos of the angle between surface normal and ground plane normal
%% heightAbGr - height above ground, the average distance to ground within the corresponding 3D patch
%% localPl - surface local planarity, sum of square distances from points to the plane
%% neighborPl - neighboring planarity, avearage difference of a 3D patch's normal with respect to its neighbors' surface normals
%% cameraPath - distance to camera path (yoz plane)
function [nalpha, heightAbGr, localPl, neighborPl, cameraPath, isValid] = getFeatures(image, labels, superpixelsNumber, pointCloud, groundPlane)
surfaceNorm = zeros(3,superpixelsNumber,'double'); %surface normal feature
nalpha = zeros(superpixelsNumber,1,'double');
heightAbGr = zeros(superpixelsNumber,1,'double'); %height above ground feature
localPl = zeros(superpixelsNumber,1,'double'); %surface local planarity feature
neighborPl = zeros(superpixelsNumber,1,'double');
cameraPath = zeros(superpixeldisparityRange = [-6 10];
disparityMap = disparity(rgb2gray(I1),rgb2gray(I2),'BlockSize',...
    15,'DisparityRange',disparityRange);sNumber, 1,'double');
isValid = ones(superpixelsNumber,1,'uint8');
GPnormal = groundPlane(1:3);
GPnou = groundPlane;
for i = 1:superpixelsNumber
    superpixelIndices = labels == i; 
    superpixelPoints = pointCloud(superpixelIndices,1:3);
    superpixelPoints = superpixelPoints(superpixelPoints(:,3)~= Inf & superpixelPoints(:,3) ~= -Inf ,:);
    [validSize,~] = size(superpixelPoints);
    if validSize > 3
        fittedPlane = fitplane(superpixelPoints');
        surfaceNorm(:,i) = fittedPlane(1:3);
        nalpha(i) = acos(dot(GPnormal,surfaceNorm(:,i))); %cos of the angle between surface normal and ground plane normal
        heightAbGr(i) = dist2GP(superpixelPoints, GPnou); %height above ground feature
        localPl(i) = distance2plane(superpixelPoints,fittedPlane);
        sCenter = sum(superpixelPoints)/validSize;
        cameraPath(i) = abs(sCenter(1)); %distance to plane yoz
    end
end
% Compute RAG
[~, ad] = imRAG(labels);
for i=1:superpixelsNumber
    if sqrt(sum(surfaceNorm(:,i).*surfaceNorm(:,i))) ~= 0
        neighbors1 = ad(ad(:,1) == i, 2);
        neighbors2 = ad(ad(:,2) == i, 1);
        neighbors = [neighbors1; neighbors2];
        [nrNeighb,~] = size(neighbors);
        if nrNeighb > 0
            neighSN = surfaceNorm(:,neighbors(:,1));
            neighSN = neighSN(:, sqrt(sum(neighSN.*neighSN)) ~= 0);
            [~, numOfValidNeigh] = size(neighSN);
            if numOfValidNeigh > 0
                nmat = repmat(surfaceNorm(:,i),1,numOfValidNeigh);
                prod = cross(nmat, neighSN);
                neighborPl(i) = sum(sqrt(sum(prod.^2,1)))/numOfValidNeigh;
            end
        end
    end
end
isValid(sqrt(sum(surfaceNorm.*surfaceNorm))==0) = 0;
end