function im(X,vec);
% im(X) or im(X,[minval maxval])
%
% Handy function for showing a grayscale image with a colorbar and
% interactive pixel value tool
%
% Serge Belongie
% 20-Aug-2000
% sjb@eecs.berkeley.edu
 
if nargin==1
   imagesc(X)
else
   imagesc(X,vec)
end
% pixval on
title(inputname(1))
colormap(gray)
colorbar
axis('image')

