function data = customReadDatastoreImage(filename)

data = imread(filename);

% Half images get gaussian noise
if rand > 0.5
data = imnoise(data, 'gaussian');
end

end