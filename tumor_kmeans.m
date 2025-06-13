clc;
clear;
close all;

% Path to dataset (current folder)
datasetPath = pwd;
folders = {'yes', 'no'}; % Subfolders containing images

for f = 1:length(folders)
    folderName = folders{f};
    folderPath = fullfile(datasetPath, folderName);
    imageFiles = dir(fullfile(folderPath, '*.jpg')); % Change to '*.png' or '*.jpeg' if needed

    for i = 1:length(imageFiles)
        % Load image
        imgName = imageFiles(i).name;
        imgPath = fullfile(folderPath, imgName);
        img = imread(imgPath);

        % Convert to grayscale if RGB
        if size(img, 3) == 3
            grayImg = rgb2gray(img);
        else
            grayImg = img;
        end

        % Apply median filtering
        filteredImg = medfilt2(grayImg, [3 3]);

        % Reshape image for K-means
        pixelValues = double(filteredImg(:));
        k = 2;
        [idx, C] = kmeans(pixelValues, k, 'MaxIter', 200, 'Replicates', 3);
        clusteredImg = reshape(idx, size(filteredImg));

        % Identify tumor cluster
        [~, tumorCluster] = max(C);
        tumorMask = clusteredImg == tumorCluster;

        % Clean mask
        cleanMask = bwareaopen(tumorMask, 100);
        cleanMask = imfill(cleanMask, 'holes');

        % Tumor detection threshold
        tumorArea = sum(cleanMask(:));
        tumorThreshold = 500;

        % Decide overlay image and label
        if tumorArea > tumorThreshold
            overlayImg = imoverlay(grayImg, cleanMask, [1 0 0]);
            detectionStatus = 'Tumor Detected';
        else
            cleanMask = false(size(grayImg)); % Empty mask
            overlayImg = grayImg;             % No overlay
            detectionStatus = 'No Tumor Detected';
        end

        % Show images
        figure('Name', ['Result: ' imgName], 'NumberTitle', 'off');
        subplot(1,3,1);
        imshow(grayImg);
        title('Original Image');

        subplot(1,3,2);
        imshow(cleanMask);
        title('Tumor Mask');

        subplot(1,3,3);
        imshow(overlayImg);
        title(detectionStatus);

        % Save results
        imwrite(cleanMask, fullfile(folderPath, ['mask_' imgName]));
        imwrite(overlayImg, fullfile(folderPath, ['overlay_' imgName]));

        fprintf('%s in %s/%s\n', detectionStatus, folderName, imgName);

        % Pause to allow viewing; press any key to continue
        disp('Press any key to continue to next image...');
        pause;
        close all;
    end
end





