function [ distance ] = distance2plane( superpixelPoints, plane )
distanceSum = 0;
A = plane(1);
B = plane(2);
C = plane(3);
D = plane(4);
numberOfPoints = size(superpixelPoints, 1);
for pointNumber = 1:numberOfPoints
    x = superpixelPoints(pointNumber, 1);
    y = superpixelPoints(pointNumber, 2);
    z = superpixelPoints(pointNumber, 3);
    currentDistance = abs(A * x + B * y + C * z + D) / sqrt(A^2 + B^2 + C^2);
    distanceSum  = distanceSum + currentDistance;
end
distance = distanceSum / numberOfPoints;
end
