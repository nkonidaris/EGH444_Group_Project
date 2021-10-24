% EGH444 - Group Project 
% by Nicholas Konidaris & Thomas Cotter

% Clear all
clear all; close all, clc
%% Importing

% Load pretrained: GoogLeNet
net = googlenet;

% Load images into datastore
imds = imageDatastore('Training_Data\All\', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

imds.ReadFcn = @customReadDatastoreImage;

%% Manual image resizing on image reading

 % Read image 
% I = readimage(imds,3);
% s = size(I);
% fprintf("Size of original images vary, e.g. First indexed image size: %d by %d\n",s(1),s(2));
 
 % Resize images for use in pretrained model
% inputSize = net.Layers(1).InputSize;
% imds.ReadFcn = @(loc)imresize(imread(loc),inputSize(1:2));
 
 % Read resized image 
% I_new = readimage(imds,3);
% imshowpair(I, I_new,'montage');
% s = size(I_new);
% fprintf("Size of new images: %d by %d\n",s(1),s(2));

%% 70/30 split for training / validation, ramdonized

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

%% Reworking model

% 244 by 224 by 3 (244x244 RGB Image)
inputSize = net.Layers(1).InputSize;

% Extract layer graph
lgraph = layerGraph(net); 

% Get number training classes
numClasses = numel(categories(imdsTrain.Labels));

newLearnableLayer = fullyConnectedLayer(numClasses, ...
    'Name','FC Bridge Layer', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);
    
% Replace fully connected layer for Bridge learning
lgraph = replaceLayer(lgraph,'loss3-classifier',newLearnableLayer);

% Leave softmax layer for classification output

% Replace classifcation layer for new outputs
newClassLayer = classificationLayer('Name','Classification Bridge');
lgraph = replaceLayer(lgraph,'output',newClassLayer);

%% Visulise Network

%deepNetworkDesigner(net);
%plot(lgraph);


%% Image Augmentation, increase image dataset

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandYReflection',true, ...
    'RandRotation', [-45 45], ...
    'RandScale', [1 2], ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Custom read function used to add random noise to images

%% Train model

miniBatchSize  = 10;
validationFrequency = floor(numel(imdsTrain.Labels)/miniBatchSize);

options = trainingOptions('adam', ...
    'miniBatchSize',10, ...
    'MaxEpochs',15, ...
    'InitialLearnRate',1e-4, ...
    'LearnRateSchedule','piecewise', ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',validationFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'ExecutionEnvironment', 'parallel');

netTransfer = trainNetwork(augimdsTrain,lgraph,options);

%% Save model for CNN

save('netTransfer', 'netTransfer', 'inputSize');
%% Load model of CNN

if exist('netTransfer', 'var') == 0
    load('netTransfer.mat','netTransfer', 'inputSize');
end
%% Training Accuracy

% imdsTesting = imageDatastore('Training_Data\All\', ...
%     'IncludeSubfolders',true, ...
%     'LabelSource','foldernames');

 imdsTesting = imageDatastore('Testing_Data\', ...
     'IncludeSubfolders',true, ...
     'LabelSource','foldernames');

augimdsTesting = augmentedImageDatastore(inputSize(1:2),imdsTesting);

[YPred,scores] = classify(netTransfer,augimdsTesting);

idx = randperm(numel(imdsTesting.Files),9);
figure
for i = 1:9
    subplot(3,3,i)
    I = readimage(imdsTesting,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

YTesting = imdsTesting.Labels;
accuracy = mean(YPred == YTesting);
fprintf("Final validation accruacy of model: %f %%\n",accuracy*100);

figure
confusionchart(YPred, YTesting);

%% Testing detect_landmark function

img = imread('Training_Data\hard_classification\Harbour Bridge\photo-1585978426586-a806812f07c2.jpg');
img = imresize(img, [224 224]);
Predicted = detect_landmark(img);

%% Custom functions used on script

