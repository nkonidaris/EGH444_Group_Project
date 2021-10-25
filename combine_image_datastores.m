% Combine img datastores 
%   * script needs commenting and cleaning to pair with 
%% 
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

imdsHardValidation = imageDatastore('Training_Data/hard_classification', ...
    'IncludeSubfolders', true, ...,
    'LabelSource', 'foldernames');


% [imds5, imdsTest5] = splitEachLabel(imds5, 0.9, 'randomized'); 
% [imds5, imdsValidation5] = splitEachLabel(imds5, 0.7, 'randomized');

% imds6 = imageDatastore('C:/Users/Tom/Documents/MATLAB/EGH444/ASSESSMENT/OG/OTHER', ...
%     'IncludeSubfolders', true, ...,
%     'LabelSource', 'foldernames');

% [imds6, imdsTest6] = splitEachLabel(imds6, 0.9, 'randomized'); 
% [imds6, imdsValidation6] = splitEachLabel(imds6, 0.7, 'randomized');

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



