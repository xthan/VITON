function [x,y,t,c]=bdry_extract_3(V);
% [x,y,t,c]=bdry_extract_3(V);
% compute (r,theta) histograms for points along boundary of single 
% region in binary image V 

% extract a set of boundary points w/oriented tangents
%sig=1;
%Vg=gauss_sig(V,sig);
Vg=V; % if no smoothing is needed
c=contourc(Vg,[.5 .5]);
[G1,G2]=gradient(Vg);

% need to deal with multiple contours (for objects with holes)
fz=c(1,:)~=0.5;
c(:,find(~fz))=NaN;
B=c(:,find(fz));

npts=size(B,2);
t=zeros(npts,1);
for n=1:npts
   x0=round(B(1,n));
   y0=round(B(2,n));
   t(n)=atan2(G2(y0,x0),G1(y0,x0))+pi/2;
end

x=B(1,:)';
y=B(2,:)';

