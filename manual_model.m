% EGH444 - Group Project 
% by Nicholas Konidaris & Thomas Cotter

% Clear all
clear variables; close all; clc;
%% Importing

% Load images into datastore
imds = imageDatastore('Training_Data\All\', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

imds.ReadFcn = @customReadDatastoreImage;
%% 64/20/16 split for Training / Validation / Testing, ramdonized

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomized');
[imdsTrain,imdsTesting] = splitEachLabel(imdsTrain,0.8,'randomized');

%%
netWidth = 4;
layers = [
    imageInputLayer([244 244 3],'Name','input')
    convolution2dLayer(3,netWidth,'Padding','same','Name','convInp')
    batchNormalizationLayer('Name','BNInp')
    reluLayer('Name','reluInp')

    convolutionalUnit(netWidth,1,'S1U1')
    additionLayer(2,'Name','add11')
    reluLayer('Name','relu11')
    convolutionalUnit(netWidth,1,'S1U2')
    additionLayer(2,'Name','add12')
    reluLayer('Name','relu12')

    convolutionalUnit(2*netWidth,2,'S2U1')
    additionLayer(2,'Name','add21')
    reluLayer('Name','relu21')
    convolutionalUnit(2*netWidth,1,'S2U2')
    additionLayer(2,'Name','add22')
    reluLayer('Name','relu22')
    
    convolutionalUnit(4*netWidth,2,'S3U1')
    additionLayer(2,'Name','add31')
    reluLayer('Name','relu31')
    convolutionalUnit(4*netWidth,1,'S3U2')
    additionLayer(2,'Name','add32')
    reluLayer('Name','relu32')

    averagePooling2dLayer(7,'Name','globalPool')
    fullyConnectedLayer(3,'Name','fcFinal')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
    ];

lgraph = layerGraph(layers);

lgraph = connectLayers(lgraph,'reluInp','add11/in2');
lgraph = connectLayers(lgraph,'relu11','add12/in2');
skip1 = [
    convolution2dLayer(1,2*netWidth,'Stride',2,'Name','skipConv1')
    batchNormalizationLayer('Name','skipBN1')];
lgraph = addLayers(lgraph,skip1);
lgraph = connectLayers(lgraph,'relu12','skipConv1');
lgraph = connectLayers(lgraph,'skipBN1','add21/in2');
lgraph = connectLayers(lgraph,'relu21','add22/in2');
skip2 = [
    convolution2dLayer(1,4*netWidth,'Stride',2,'Name','skipConv2')
    batchNormalizationLayer('Name','skipBN2')];
lgraph = addLayers(lgraph,skip2);
lgraph = connectLayers(lgraph,'relu22','skipConv2');
lgraph = connectLayers(lgraph,'skipBN2','add31/in2');
lgraph = connectLayers(lgraph,'relu31','add32/in2');

%% Image Augmentation, increase image dataset
inputSize = [244 244];

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandYReflection',true, ...
    'RandRotation', [-60 60], ...
    'RandScale', [0.5 2], ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsTesting = augmentedImageDatastore(inputSize(1:2),imdsTesting);

% Custom read function used to add random noise to images

%% Train model

miniBatchSize  = 20;
validationFrequency = floor(numel(imdsTrain.Labels)/miniBatchSize);

if strcmp(version('-release'),'2021b')
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
		'ExecutionEnvironment', 'gpu');
else
	options = trainingOptions('adam', ...
	'miniBatchSize',miniBatchSize, ...
	'MaxEpochs',12, ...
	'InitialLearnRate',1e-4, ...
	'LearnRateSchedule','piecewise', ...
	'Shuffle','every-epoch', ...
	'ValidationData',augimdsValidation, ...
	'ValidationFrequency',validationFrequency, ...
	'Verbose',false, ...
	'Plots','training-progress',...
	'ExecutionEnvironment', 'gpu');	
end
	

netTransfer = trainNetwork(augimdsTrain,lgraph,options);

%% Save model for CNN

save('Manual_net', 'netTransfer');
%% Load model of CNN

if exist('netTransfer', 'var') == 0
    load('Manual_net.mat','netTransfer');
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
    title(string(label), 'FontSize', 24);
end

YTrain = imdsTrain.Labels;
TrainError = mean(YPred == YTrain);
fprintf("Training accruacy of model: %f %%\n",TrainError*100);
t = sprintf('Training accuracy: %f %%', TrainError*100);
sgtitle(t, 'FontSize', 30);

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
    title(string(label), 'FontSize', 24);
end

YValidation = imdsValidation.Labels;
ValError = mean(YPred == YValidation);
fprintf("Validation accruacy of model: %f %%\n",ValError*100);
t = sprintf('Validation accuracy: %f %%', ValError*100);
sgtitle(t, 'FontSize', 30);

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
    title(string(label), 'FontSize', 24);
end

YTesting = imdsTesting.Labels;
testError = mean(YPred == YTesting);
fprintf("Testing accruacy of model: %f %%\n",testError*100);
t = sprintf('Testing accuracy: %f %%', testError*100);
sgtitle(t, 'FontSize', 30);

figure
cmTest = confusionchart(YPred, YTesting);
cmTest.Title = 'Confusion Matrix for Testing Data';
cmTest.ColumnSummary = 'column-normalized';
cmTest.RowSummary = 'row-normalized';

%% Helper Functions

function layers = convolutionalUnit(numF,stride,tag)
layers = [
    convolution2dLayer(3,numF,'Padding','same','Stride',stride,'Name',[tag,'conv1'])
    batchNormalizationLayer('Name',[tag,'BN1'])
    reluLayer('Name',[tag,'relu1'])
    convolution2dLayer(3,numF,'Padding','same','Name',[tag,'conv2'])
    batchNormalizationLayer('Name',[tag,'BN2'])];
end