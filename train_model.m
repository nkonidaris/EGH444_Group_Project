% EGH444 - Group Project 
% by Nicholas Konidaris & Thomas Cotter

% Clear all
clear all; close all, clc
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
validationFrequency = floor(numel(y_train)/miniBatchSize);

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

save netTransfer;
%% Load model of CNN

load netTransfer.mat;
%% Training Accuracy

imdsTesting = imageDatastore('Training_Data\All\', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

[YPred,scores] = classify(netTransfer,imdsTesting);

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

confusionchart(YPred, YTesting);

%% .............................NOT USED.................................




newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% plot(lgraph)
% ylim([0,10])

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

pixelRange = [-30 30];
scaleRange = [0.9 1.1];

imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);

options = trainingOptions('adam', ...
    'GradientDecayFactor', 0.9, ...
    'SquaredGradientDecayFactor', 0.99, ...
    'verbose', 1, ...
    'MiniBatchSize',10, ...
    'MaxEpochs',100, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',8, ...
    'Verbose',false, ...
    'Plots','training-progress');

% options = trainingOptions('sgdm', ...
%     'MiniBatchSize',miniBatchSize, ...
%     'MaxEpochs',50, ...
%     'InitialLearnRate',1e-3, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',augimdsValidation, ...
%     'ValidationFrequency',10, ...
%     'Verbose',false, ...
%     'Plots','training-progress');


neuralnet6 = trainNetwork(augimdsTrain, lgraph, options);

%%

[YPred,probs] = classify(neuralnet5,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)

idx = randperm(numel(imdsValidation.Files), 49);
figure
for i = 1:49
    subplot(7,7,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end


%%


layersTransfer = net.Layers(1:end-3);

numClasses = numel(categories(imdsTrain.Labels));

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',2,'BiasLearnRateFactor',2)
    softmaxLayer
    classificationLayer
    ];

pixelRange = [-35 35];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter, ...
    'ColorPreprocessing', 'gray2rgb');

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation, 'ColorPreprocessing', 'gray2rgb');


%%
% options = trainingOptions('sgdm', ...
%     'Momentum', 0.85, ...
%     'LearningRateDropFactor', 0.2, ...
%     'LearningRateDropPeriod', 2, ...
%     'MiniBatchSize',10, ...
%     'MaxEpochs',25, ...
%     'InitialLearnRate',1e-4, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',augimdsValidation, ...
%     'ValidationFrequency',3, ...
%     'Verbose',false, ...
%     'Plots','training-progress');

options = trainingOptions('adam', ...
    'GradientDecayFactor', 0.9, ...
    'SquaredGradientDecayFactor', 0.99, ...
    'verbose', 1, ...
    'MiniBatchSize',8, ...
    'MaxEpochs',300, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',5, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer3 = trainNetwork(augimdsTrain,layers,options);

%% Custom functions used on script

function data = customReadDatastoreImage(filename)

data = imread(filename);

% Half images get gaussian noise
if rand > 0.5
data = imnoise(data, 'gaussian');
end

end

