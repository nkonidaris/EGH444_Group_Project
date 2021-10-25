%% 
% EGH444 - Group Project 
% by Nicholas Konidaris & Thomas Cotter

% Clear all
% clear all; close all, clc

%%

% Combine img datastores 
%   * script needs commenting and cleaning to pair with 
% unzip('C:/Users/Tom/Documents/MATLAB/EGH444/ASSESSMENT/OG/Level 11.zip')
% unzip('C:/Users/Tom/Documents/MATLAB/EGH444/ASSESSMENT/OG/Level 2.zip')
% unzip('C:/Users/Tom/Documents/MATLAB/EGH444/ASSESSMENT/OG/Level 3.zip')

imds1 = imageDatastore('Training_Data/Level 1', ...
    'IncludeSubfolders', true, ...,
    'LabelSource', 'foldernames');

[imds1, imdsTest1] = splitEachLabel(imds1, 0.9, 'randomized'); 
[imdsTrain1, imdsValidation1] = splitEachLabel(imds1, 0.7, 'randomized');

imds2 = imageDatastore('Training_Data/Level 2', ...
    'IncludeSubfolders', true, ...,
    'LabelSource', 'foldernames');

[imds2, imdsTest2] = splitEachLabel(imds2, 0.9, 'randomized'); 
[imdsTrain2, imdsValidation2] = splitEachLabel(imds2, 0.7, 'randomized');

imds3 = imageDatastore('Training_Data/Level 3', ...
    'IncludeSubfolders', true, ...,
    'LabelSource', 'foldernames');

[imds3, imdsTest3] = splitEachLabel(imds3, 0.9, 'randomized'); 
[imds3, imdsValidation3] = splitEachLabel(imds3, 0.7, 'randomized');

% imds4 = imageDatastore('OTHER', ...
%     'IncludeSubfolders', true, ...,
%     'LabelSource', 'foldernames');
% 
imds4 = imageDatastore('Training_Data/New', ...
    'IncludeSubfolders', true, ...,
    'LabelSource', 'foldernames');

[imds4, imdsTest4] = splitEachLabel(imds4, 0.85, 'randomized'); 
[imds4, imdsValidation4] = splitEachLabel(imds4, 0.8, 'randomized');

imdsTrain = imageDatastore(cat(1, imds1.Files, imds2.Files, imds3.Files, imds4.Files));
imdsTrain.Labels = cat(1, imds1.Labels, imds2.Labels, imds3.Labels, imds4.Labels); 

imdsValidation = imageDatastore(cat(1, imdsValidation1.Files, imdsValidation2.Files, imdsValidation3.Files, ...
    imdsValidation4.Files));
imdsValidation.Labels = cat(1, imdsValidation1.Labels, imdsValidation2.Labels, imdsValidation3.Labels, ...
    imdsValidation4.Labels); 

imdsTest = imageDatastore(cat(1, imdsTest1.Files, imdsTest2.Files, imdsTest3.Files, imdsTest4.Files));
imdsTest.Labels = cat(1, imdsTest1.Labels, imdsTest2.Labels, imdsTest3.Labels, imdsTest4.Labels); 

imdsTrain.ReadFcn = @customReadDatastoreImage;
imdsValidation.ReadFcn = @customReadDatastoreImage;
imdsTest.ReadFcn = @customReadDatastoreImage;

%% Importing

% Load pretrained: GoogLeNet
net = googlenet;

% Load images into datastore
imds = imageDatastore('Training_Data\Level 1\', ...
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
%deepNetworkDesigner(net)

plot(lgraph);


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
    'miniBatchSize',15, ...
    'MaxEpochs',50, ...
    'InitialLearnRate',1e-4, ...
    'LearnRateSchedule','piecewise', ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',10, ...
    'Verbose',false, ...
    'Plots','training-progress');
%     'ExecutionEnvironment', 'parallel');

netTransfer7 = trainNetwork(augimdsTrain,lgraph,options);

%% Save model for CNN

save netTransfer;
%% Load model of CNN

load netTransfer.mat;
%% Training Accuracy

imdsTesting = imageDatastore('Training_Data\All\', ...
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

%% .............................NOT USED.................................





%%
% Test: Test datasets 

imdsHardValidation = imageDatastore('Training_Data/hard_classification', ...
    'IncludeSubfolders', true, ...,
    'LabelSource', 'foldernames');

imdsHardValidation2 =  augmentedImageDatastore(inputSize(1:2),imdsHardValidation);
% augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsHardValidation);
[YPred,probs] = classify(netTransfer6, imdsHardValidation2);
accuracy_hard = mean(YPred == imdsHardValidation.Labels)

confusionchart(YPred, imdsHardValidation.Labels)

%% 

netTransferTest = netTransfer7;

augimdsTest =  augmentedImageDatastore(inputSize(1:2),imdsTest);
[YPred,probs] = classify(netTransferTest, augimdsTest);
accuracy_test = mean(YPred == imdsTest.Labels)

confusionchart(YPred, imdsTest.Labels)
% horzcat(YPred, imdsTest.Labels)

% Changed minibatches to 10 from 5 then to 7
% -> learning rate to 5e5 
% changing learning rate (halved) slowed learning too much and results were
% sporadic: stopped early (netTransfer4) or 5 

%%

%% Custom functions used on script

function data = customReadDatastoreImage(filename)

data = imread(filename);

% Half images get gaussian noise
if rand > 0.5
data = imnoise(data, 'gaussian');
end

end

% augmentedImageDatastore(inputSize(1:2),imdsHardValidation);

