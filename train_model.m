% EGH444 - Group Project 
% by Nicholas Konidaris & Thomas Cotter

% Clear all
clear variables; close all; clc;
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

%% 64/20/16 split for Training / Validation / Testing, ramdonized

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomized');
[imdsTrain,imdsTesting] = splitEachLabel(imdsTrain,0.8,'randomized');

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

% Replace classification layer for new outputs
newClassLayer = classificationLayer('Name','Classification Bridge');
lgraph = replaceLayer(lgraph,'output',newClassLayer);

%% Visualise Network

%deepNetworkDesigner(net);
%plot(lgraph);
%disp(net.Layers)


%% Image Augmentation, increase image dataset

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandYReflection',true, ...
    'RandRotation', [-60 60], ...
    'RandScale', [1 2], ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsTesting = augmentedImageDatastore(inputSize(1:2),imdsTesting);

% Custom read function used to add random noise to images

%% Train model

% miniBatchSize = 10
% Training accuracy of model: 97.560976 %
% Validation accuracy of model: 87.272727 %
% Testing accuracy of model: 86.274510 %

% miniBatchSize = 20
% Training accuracy of model: 97.446809 %
% Validation accuracy of model: 90.410959 %
% Testing accuracy of model: 87.931034 %

% miniBatchSize = 20, All data
% Training accuracy of model: 92.150171 %
% Validation accuracy of model: 93.150685 %

% Note: miniBatchSize = 30 performed poorly

miniBatchSize  = 20;
validationFrequency = floor(numel(imdsTrain.Labels)/miniBatchSize);

options = trainingOptions('adam', ...
    'miniBatchSize',miniBatchSize, ...
    'MaxEpochs',20, ...
    'InitialLearnRate',1e-4, ...
    'LearnRateSchedule','piecewise', ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',validationFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'OutputNetwork', 'best-validation-loss', ...
    'ExecutionEnvironment', 'auto');

netTransfer = trainNetwork(augimdsTrain,lgraph,options);

%% Save model for CNN

save('netTransfer_Full_Dataset', 'netTransfer', 'inputSize');
%save('netTransfer', 'netTransfer', 'inputSize');
%% Load model of CNN

if exist('netTransfer', 'var') == 0
    load('netTransfer.mat','netTransfer', 'inputSize');
end

%% Training Accuracy

YPred = classify(netTransfer, augimdsTrain);

idx = randperm(numel(imdsTrain.Files),9);
figure

for i = 1:9
    subplot(3,3,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

YTrain = imdsTrain.Labels;
TrainError = mean(YPred == YTrain);
fprintf("Training accruacy of model: %f %%\n",TrainError*100);
t = sprintf('Training accuracy: %f %%\n', TrainError*100);
sgtitle(t);

figure
cmtrain = confusionchart(YPred, YTrain);
cmtrain.Title = 'Confusion Matrix for Training Data';
cmtrain.ColumnSummary = 'column-normalized';
cmtrain.RowSummary = 'row-normalized';

%% Validation Accuracy

YPred = classify(netTransfer,augimdsValidation);

idx = randperm(numel(imdsValidation.Files),9);
figure

for i = 1:9
    subplot(3,3,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

YValidation = imdsValidation.Labels;
ValError = mean(YPred == YValidation);
fprintf("Validation accruacy of model: %f %%\n",ValError*100);
t = sprintf('Validation accuracy: %f %%\n', ValError*100);
sgtitle(t);

figure
cmVal = confusionchart(YPred, YValidation);
cmVal.Title = 'Confusion Matrix for Validation Data';
cmVal.ColumnSummary = 'column-normalized';
cmVal.RowSummary = 'row-normalized';

%% Testing Accuracy

% External Testing datastore
%  imdsTesting = imageDatastore('Testing_Data\', ...
%      'IncludeSubfolders',true, ...
%      'LabelSource','foldernames');

%augimdsTesting = augmentedImageDatastore(inputSize(1:2),imdsTesting);

YPred = classify(netTransfer,augimdsTesting);

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
testError = mean(YPred == YTesting);
fprintf("Testing accruacy of model: %f %%\n",testError*100);
t = sprintf('Testing accuracy: %f %%\n', testError*100);
sgtitle(t);

figure
cmTest = confusionchart(YPred, YTesting);
cmTest.Title = 'Confusion Matrix for Testing Data';
cmTest.ColumnSummary = 'column-normalized';
cmTest.RowSummary = 'row-normalized';

%% Testing detect_landmark function

img = imread('Training_Data\hard_classification\Harbour Bridge\photo-1585978426586-a806812f07c2.jpg');
img = imresize(img, [224 224]);
Predicted = detect_landmark(img);

%% Custom functions used on script

