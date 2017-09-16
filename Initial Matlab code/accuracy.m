function acc = accuracy(originalLabelledImg, labelledImg)
setMap

origIm = ind2rgb(originalLabelledImg, MAP);
predIm = ind2rgb(labelledImg, MAP);
subplot(3,1,1);
imshow(origIm);
subplot(3,1,2);
imshow(predIm);
[heightImg, widthImg, ~] = size(originalLabelledImg); %size of image 
labelsImgO = originalLabelledImg;
labelsImgL = labelledImg;
[correctLabelled,~] = size(labelsImgO(labelsImgO==labelsImgL)); %global accuracy
correctLabelled = correctLabelled/(heightImg*widthImg) *100;
[pavement,~] = size(labelsImgO(labelsImgO== 0 & labelsImgO==labelsImgL));
%pavement
[px,~] = size(labelsImgO(labelsImgO== 0));
if px == 0
    pavement = -1;
else
    pavement = pavement/px *100;
end
[car,~] = size(labelsImgO(labelsImgO== 1 & labelsImgO==labelsImgL)); %car
[px,~] = size(labelsImgO(labelsImgO== 1));
if px == 0
    car = -1;
else
    car = car/px *100;
end
[sky,~] = size(labelsImgO(labelsImgO== 2 & labelsImgO==labelsImgL)); %sky
[px,~] = size(labelsImgO(labelsImgO== 2));
if px == 0
    sky = -1;
else
    sky = sky/px*100;
end
[poles,~] = size(labelsImgO(labelsImgO== 3 & labelsImgO==labelsImgL)); %poles
[px,~] = size(labelsImgO(labelsImgO== 3));
if px == 0
    poles = -1;
else
    poles = poles/px*100;
end
[vegetation,~] = size(labelsImgO(labelsImgO== 4 & labelsImgO==labelsImgL));
%vegetation
[px,~] = size(labelsImgO(labelsImgO== 4));
if px == 0
    vegetation = -1;
else
    vegetation = vegetation/px*100;
end
[pedestrian,~] = size(labelsImgO(labelsImgO== 5 & labelsImgO==labelsImgL));
%pedestrian
[px,~] = size(labelsImgO(labelsImgO== 5));
if px == 0
    pedestrian = -1;
else
    pedestrian = pedestrian/px *100;
end
[signance,~] = size(labelsImgO(labelsImgO== 6 & labelsImgO==labelsImgL));
%signance
[px,~] = size(labelsImgO(labelsImgO== 6));
if px == 0
    signance = -1;
else
    signance = signance/px*100;
end
[fence,~] = size(labelsImgO(labelsImgO== 7 & labelsImgO==labelsImgL)); %fence
[px,~] = size(labelsImgO(labelsImgO== 7));
if px == 0
    fence = -1;
else
    fence = fence/px*100;
end
[building,~] = size(labelsImgO(labelsImgO== 8 & labelsImgO==labelsImgL));
%building
[px,~] = size(labelsImgO(labelsImgO== 8));
if px == 0
    building = -1;
else
    building = building/px*100;
end
[road,~] = size(labelsImgO(labelsImgO== 9 & labelsImgO==labelsImgL)); %road
[px,~] = size(labelsImgO(labelsImgO== 9));
if px == 0
    road = -1;
else
    road = road/px*100;
end
[black,~] = size(labelsImgO(labelsImgO== 10 & labelsImgO==labelsImgL)); %black
[px,~] = size(labelsImgO(labelsImgO== 10));
if px == 0
    black = -1;
else
    black = black/px*100;
end
fprintf(1,'pavement: %f\ncar: %f\nsky: %f\npoles: %f\nvegetation: %f\npedestrian: %f\nsignance: %f\nfence: %f\nbuilding: %f\nroad: %f\nblack: %f\nglobal: %f\n',pavement, car, sky, poles, vegetation, pedestrian, signance, fence, building, road, black, correctLabelled);
acc = [pavement, car, sky, poles, vegetation, pedestrian, signance, fence, building, road, black, correctLabelled];
end