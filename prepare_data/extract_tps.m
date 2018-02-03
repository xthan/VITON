clear all;
addpath('../shape_context/')
FEATURE_DIR='../data/tps/';
DATA_ROOT='../data/women_top/';
SEG_DIR='../data/segment/';


all_images = textread('../data/viton_train_images.txt', '%s', 'delimiter', ' ');
n_control = 10;
h = 256/2
w = 192/2
% Calculate TPS cost time. You may consider to run it in parallel.
for i = 1:length(all_images)/2
    2*i-1
    image_name = all_images{2*i-1};
    if exist([FEATURE_DIR, image_name(1:end-4), '.mat'], 'file')
        'skipping...'
        continue
    end
    V1 = imread([DATA_ROOT, image_name(1:end-5), '1.jpg']);
    [h0,w0,~] = size(V1);
    orig_im = imresize(im2double(V1), [h,w]);
    V1 = imresize(double(V1(:,:,1) ~= 255 & V1(:,:,2) ~= 255 & V1(:,:,3) ~= 255), [h,w], 'nearest');
    V1 = imfill(V1);
    V1 = medfilt2(V1);
    % Load product mask of image.
    orig_im2 = imread([DATA_ROOT, image_name]);
    V2 = load([SEG_DIR, image_name(1:end-4)]);
    V2 = V2.segment';
    % preprocess segmentation map
    if h0 > w0
      V2 = V2(:, 1: round(641 * w0 / h0));
    else
      V2 = V2(1:round(641 * h0 / w0), :);
    end
    V2 = imresize(V2, [h, w], 'nearest');
    V2 = double(V2 == 5);
    
    % TPS transformation
    try
        [keypoints1, keypoints2, warp_points0, warp_im] = tps_main(V1, V2, n_control, orig_im, 0);
    catch
        'not enough'
        continue
    end
    keypoints1 = [keypoints1(:,2) / h, keypoints1(:,1) / w];
    keypoints2 = [keypoints2(:,2) / h, keypoints2(:,1) / w];
    
    warp_points = [warp_points0(1,:); warp_points0(2,:)];
    
    % I am now using the nearest neighbor instead of an inverse transformation to get the control points
    % There should be a better way to do so.
    % Actually using the warp_im for stage2 should also work, but I calcualte the control_points
    % since I initially wanted to use a small network to predict these control points,
    % so we can speedup the test process.
    [gx,gy]=meshgrid(linspace(1,w,n_control),linspace(1,h,n_control));
    gx=gx(:); gy=gy(:);
    [x,y]=meshgrid(linspace(-2*w,2*w,4*w+1),linspace(-2*h,2*h,4*h+1));
    x=x(:); y=y(:);
    % nn of each point
    point_distance = dist2([gx,gy], warp_points');
    [~, point_index] = min(point_distance, [], 2);
    control_points = [x(point_index), y(point_index)]';
    control_points = [control_points(1,:) / w; control_points(2,:) / h];
    control_points = reshape(control_points, [2,n_control,n_control]);
    save([FEATURE_DIR, image_name(1:end-4), '.mat'], 'keypoints1', 'keypoints2', 'control_points');
end