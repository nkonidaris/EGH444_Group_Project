% EGH444 - Group 24 Project 
% by Nicholas Konidaris & Thomas Cotter

function landmark = detect_landmark(img)

    % Loads CNN model, if not already present
    if exist('netTransfer', 'var') == 0
        load('netTransfer_Presentation.mat','netTransfer');
    end

    % Resizes image to 224 by 224
    img = imresize(img, [224 224]);

    % Classifies image into categorical 
    YPred = classify(netTransfer,img);

    % Returns outcome as uint8
    switch YPred
       case 'Harbour Bridge'
           landmark = uint8(1);
       case 'Story Bridge'
           landmark = uint8(2);
        otherwise
            landmark = uint8(0);
    end 


end
