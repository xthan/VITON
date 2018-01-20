function W=gaussker(N);
% W=gaussker(N);
% returns NxN Gaussian kernel

g=2^(1-N)*diag(fliplr(pascal(N)));
W=g*g';
