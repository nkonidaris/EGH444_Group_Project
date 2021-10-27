% EGH444 - Group 24 Project 
% by Nicholas Konidaris & Thomas Cotter

function landmark = detect_landmark(img)

if exist('netTransfer', 'var') == 0
    load('netTransfer_Presentation.mat','netTransfer');
end

    YPred = classify(netTransfer,img);

    switch YPred
       case 'Harbour Bridge'
           landmark = uint8(1);
       case 'Story Bridge'
           landmark = uint8(2);
        otherwise
            landmark = uint8(0);
    end 


end
