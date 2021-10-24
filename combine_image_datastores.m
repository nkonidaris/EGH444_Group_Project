% Combine img datastores 
%   * script needs commenting and cleaning to pair with 
%% 
unzip('C:/Users/Tom/Documents/MATLAB/EGH444/ASSESSMENT/OG/Level 11.zip')
unzip('C:/Users/Tom/Documents/MATLAB/EGH444/ASSESSMENT/OG/Level 2.zip')
unzip('C:/Users/Tom/Documents/MATLAB/EGH444/ASSESSMENT/OG/Level 3.zip')

imds1 = imageDatastore('Level 11', ...
    'IncludeSubfolders', true, ...,
    'LabelSource', 'foldernames');

[imds1, imdsTest1] = splitEachLabel(imds1, 0.9, 'randomized'); 
[imdsTrain1, imdsValidation1] = splitEachLabel(imds1, 0.7, 'randomized');

imds2 = imageDatastore('Level 2', ...
    'IncludeSubfolders', true, ...,
    'LabelSource', 'foldernames');

[imds2, imdsTest2] = splitEachLabel(imds2, 0.9, 'randomized'); 
[imdsTrain2, imdsValidation2] = splitEachLabel(imds2, 0.7, 'randomized');

imds3 = imageDatastore('Level 3', ...
    'IncludeSubfolders', true, ...,
    'LabelSource', 'foldernames');

[imds3, imdsTest3] = splitEachLabel(imds3, 0.9, 'randomized'); 
[imds3, imdsValidation3] = splitEachLabel(imds3, 0.7, 'randomized');

% imds4 = imageDatastore('OTHER', ...
%     'IncludeSubfolders', true, ...,
%     'LabelSource', 'foldernames');
% 
% [imds4, imds4Test] = splitEachLabel(imds4, 0.9, 'randomized'); 
% [imds4, imds4Validation] = splitEachLabel(imds4, 0.7, 'randomized');

imds5 = imageDatastore('C:/Users/Tom/Documents/MATLAB/EGH444/ASSESSMENT/OG/OTHER_BRIDGES', ...
    'IncludeSubfolders', true, ...,
    'LabelSource', 'foldernames');

[imds5, imdsTest5] = splitEachLabel(imds5, 0.9, 'randomized'); 
[imds5, imdsValidation5] = splitEachLabel(imds5, 0.7, 'randomized');

imds6 = imageDatastore('C:/Users/Tom/Documents/MATLAB/EGH444/ASSESSMENT/OG/OTHER', ...
    'IncludeSubfolders', true, ...,
    'LabelSource', 'foldernames');

[imds6, imdsTest6] = splitEachLabel(imds6, 0.9, 'randomized'); 
[imds6, imdsValidation6] = splitEachLabel(imds6, 0.7, 'randomized');

imdsTrain = imageDatastore(cat(1, imds1.Files, imds2.Files, imds3.Files, imds5.Files, imds6.Files));
imdsTrain.Labels = cat(1, imds1.Labels, imds2.Labels, imds3.Labels, imds5.Labels, imds6.Labels); 

imdsValidation = imageDatastore(cat(1, imdsValidation1.Files, imdsValidation2.Files, imdsValidation3.Files, ...
    imdsValidation5.Files, imdsValidation6.Files));
imdsValidation.Labels = cat(1, imdsValidation1.Labels, imdsValidation2.Labels, imdsValidation3.Labels, ...
    imdsValidation5.Labels, imdsValidation6.Labels); 

imdsTest = imageDatastore(cat(1, imdsTest1.Files, imdsTest2.Files, imdsTest3.Files, imdsTest5.Files, imdsTest6.Files));
imdsTest.Labels = cat(1, imdsTest1.Labels, imdsTest2.Labels, imdsTest3.Labels, imdsTest5.Labels, imdsTest6.Labels); 
