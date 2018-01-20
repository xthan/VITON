function [cx,cy,E,L]=bookstein(X,Y,beta_k);
% [cx,cy,E,L]=bookstein(X,Y,beta_k);
%
% Bookstein PAMI89

N=size(X,1);
Nb=size(Y,1);

if N~=Nb
   error('number of landmarks must be equal')
end

% compute distances between left points
r2=dist2(X,X);

K=r2.*log(r2+eye(N,N)); % add identity matrix to make K zero on the diagonal
P=[ones(N,1) X];
L=[K  P
   P' zeros(3,3)];
V=[Y' zeros(2,3)];
if nargin>2
   % regularization
   L(1:N,1:N)=L(1:N,1:N)+beta_k*eye(N,N);
end
invL=inv(L);

c=invL*V';
cx=c(:,1);
cy=c(:,2);

if nargout>2
   % compute bending energy (w/o regularization)
   Q=c(1:N,:)'*K*c(1:N,:);
   E=mean(diag(Q));
end