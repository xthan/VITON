% Xintong Han 2017
% Perform TPS transformation using the masks output by the first stage
% Code for shape context matching (under shape_context folder) are modified from:
% https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/sc_digits.html

addpath('shape_context')
DATA_ROOT='data/women_top/';
MASK_DIR='results/stage1/tps/00015000_';

% Check if using MATLAB or Octave
isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if(isOctave)
  % Load image package for resizing images
  pkg load image;
  % Turn off warning
  warning('off', 'Octave:possible-matlab-short-circuit-operator');
end

[image1, image2] = textread('data/viton_test_pairs.txt', '%s %s\n');
% using a smaller height and width for the shape context matching
% can save time without hurting the perform too much.
h = 256/4;
w = 192/4;
% we use 10x10 control_points
n_control = 10;
for i = 1:1 %length(image1) % only run over 1 image (for now)
    image_name1 = image1{i};
    image_name2 = image2{i};
    if exist([MASK_DIR, image_name1, '_', image_name2, '_tps.mat'])
        continue
    end
    V1 = imread([DATA_ROOT, image_name2]);
    [h0, w0, ~] = size(V1);
    orig_im = imresize(im2double(V1), [h,w]);
    % extract fashion item masks
    V1 = V1(:,:,1) ~= 255 & V1(:,:,2) ~= 255 & V1(:,:,3) ~= 255;
    V1 = imresize(double(V1), [h,w], 'nearest');
    V1 = imfill(V1);
    V1 = medfilt2(V1);
    % Load product mask of image.
    V2 = load([MASK_DIR, image_name1, '_', image_name2, '_mask.mat']);
    V2 = imresize(double(V2.mask), [h,w]);
    % TPS transformation
    try
        tic;[keypoints1, keypoints2, warp_points0, warp_im] = tps_main(V1, V2, n_control, orig_im, 0);toc;
    catch
        % when there is not enough keypoints for estimating the TPS
        % transformation
        'not enough keypoints'
        continue
    end
    keypoints1 = [keypoints1(:,2) / h, keypoints1(:,1) / w];
    keypoints2 = [keypoints2(:,2) / h, keypoints2(:,1) / w];
    warp_points = [warp_points0(1,:); warp_points0(2,:)];
    [gx, gy] = meshgrid(linspace(1,w,n_control), linspace(1,h,n_control));
    gx = gx(:); gy = gy(:);
    [x,y] = meshgrid(linspace(-2*w,2*w,4*w+1),linspace(-2*h,2*h,4*h+1));
    x=x(:); y=y(:);
    % nn of each point
    point_distance = dist2([gx,gy], warp_points');
    [~, point_index] = min(point_distance, [], 2);
    control_points = [x(point_index), y(point_index)]';
    control_points = [control_points(1,:) / w; control_points(2,:) / h];
    control_points = reshape(control_points, [2,n_control,n_control]);
    save([MASK_DIR, image_name1, '_', image_name2, '_tps.mat'], 'control_points', '-v6');
end
