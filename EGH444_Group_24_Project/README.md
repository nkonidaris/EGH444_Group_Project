# EGH444 Group 24 Project

Folder structure:
	detect_landmark.m - API for classification of bridges.
	netTransfer_Presentation.mat - Model of trained CNN, automatically loaded by detect_landmark.m, used for presentation results.
	train_model.m - Script used to train the CNN models and evaluate results.
	customReadDatastoreImage.m - Function used by train_model.m to add noise to images during training.

Use:
	Simply run "detect_landmark(img)" with img
	Return will be an uint8 representing Story Bridge (2), Harbour Bridge (1), or Other (0)