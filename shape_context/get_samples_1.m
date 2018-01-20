function [xi,yi,ti]=get_samples_1(x,y,t,nsamp);
% [xi,yi,ti]=get_samples_1(x,y,t,nsamp);
%
% uses Jitendra's sampling method

N=length(x);
k=3;
Nstart=min(k*nsamp,N);

ind0=randperm(N);
ind0=ind0(1:Nstart);

xi=x(ind0);
yi=y(ind0);
ti=t(ind0);
xi=xi(:);
yi=yi(:);
ti=ti(:);

d2=dist2([xi yi],[xi yi]);
d2=d2+diag(Inf*ones(Nstart,1));

s=1;
while s
   % find closest pair
   [a,b]=min(d2);
   [c,d]=min(a);
   I=b(d);
   J=d;
   % remove one of the points
   xi(J)=[];
   yi(J)=[];
   ti(J)=[];
   d2(:,J)=[];
   d2(J,:)=[];
   if size(d2,1)==nsamp
      s=0;
   end
end

      