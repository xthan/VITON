% script for doing shape-context based matching with alternating steps
% of estimating correspondences and estimating the regularized TPS
% transformation

% initialize transformed version of model pointset
Xk=X; 
% initialize counter
k=1;
s=1;
% out_vec_{1,2} are indicator vectors for keeping track of estimated
% outliers on each iteration
out_vec_1=zeros(1,nsamp1); 
out_vec_2=zeros(1,nsamp2);
while s
   disp(['iter=' int2str(k)])
   
   % compute shape contexts for (transformed) model
   [BH1,mean_dist_1]=sc_compute(Xk',zeros(1,nsamp1),mean_dist_global,nbins_theta,nbins_r,r_inner,r_outer,out_vec_1);

   % compute shape contexts for target, using the scale estimate from
   % the warped model
   % Note: this is necessary only because out_vec_2 can change on each
   % iteration, which affects the shape contexts.  Otherwise, Y does
   % not change.
   [BH2,mean_dist_2]=sc_compute(Y',zeros(1,nsamp2),mean_dist_1,nbins_theta,nbins_r,r_inner,r_outer,out_vec_2);

   % compute regularization parameter
   beta_k=(mean_dist_1^2)*beta_init*r^(k-1);

   % compute pairwise cost between all shape contexts
   costmat=hist_cost_2(BH1,BH2);
   % pad the cost matrix with costs for dummies
   nptsd=nsamp1+ndum1;
   costmat2=eps_dum*ones(nptsd,nptsd);
   costmat2(1:nsamp1,1:nsamp2)=costmat;
   disp('running hungarian alg.')
   cvec=hungarian(costmat2);
%   cvec=hungarian_fast(costmat2);
   disp('done.')

   % update outlier indicator vectors
   [a,cvec2]=sort(cvec);
   out_vec_1=cvec2(1:nsamp1)>nsamp2;
   out_vec_2=cvec(1:nsamp2)>nsamp1;

   % format versions of Xk and Y that can be plotted with outliers'
   % correspondences missing
   X2=NaN*ones(nptsd,2);
   X2(1:nsamp1,:)=Xk;
   X2=X2(cvec,:);
   X2b=NaN*ones(nptsd,2);
   X2b(1:nsamp1,:)=X;
   X2b=X2b(cvec,:);
   Y2=NaN*ones(nptsd,2);
   Y2(1:nsamp2,:)=Y;

   % extract coordinates of non-dummy correspondences and use them
   % to estimate transformation
   ind_good=find(~isnan(X2b(1:nsamp1,1)));
   % NOTE: Gianluca said he had to change nsamp1 to nsamp2 in the
   % preceding line to get it to work properly when nsamp1~=nsamp2 and
   % both sides have outliers...
   n_good=length(ind_good);
   X3b=X2b(ind_good,:);
   Y3=Y2(ind_good,:);

   if display_flag
      figure(2)
      plot(X2(:,1),X2(:,2),'b+',Y2(:,1),Y2(:,2),'ro')
      hold on
      h=plot([X2(:,1) Y2(:,1)]',[X2(:,2) Y2(:,2)]','k-');
      hold off
      title([int2str(n_good) ' correspondences (warped X)'])
      drawnow	
   end
   
   if display_flag
      % show the correspondences between the untransformed images
      figure(3)
      plot(X(:,1),X(:,2),'b+',Y(:,1),Y(:,2),'ro')
      ind=cvec(ind_good);
      hold on
      plot([X2b(:,1) Y2(:,1)]',[X2b(:,2) Y2(:,2)]','k-')
      hold off
      title([int2str(n_good) ' correspondences (unwarped X)'])
      drawnow	
   end

   % estimate regularized TPS transformation
   [cx,cy,E]=bookstein(X3b,Y3,beta_k);

   % calculate affine cost
   A=[cx(n_good+2:n_good+3,:) cy(n_good+2:n_good+3,:)];
   s=svd(A);
   aff_cost=log(s(1)/s(2));
   
   % calculate shape context cost
   [a1,b1]=min(costmat,[],1);
   [a2,b2]=min(costmat,[],2);
   sc_cost=max(mean(a1),mean(a2));
   
   % warp each coordinate
   fx_aff=cx(n_good+1:n_good+3)'*[ones(1,nsamp1); X'];
   d2=max(dist2(X3b,X),0);
   U=d2.*log(d2+eps);
   fx_wrp=cx(1:n_good)'*U;
   fx=fx_aff+fx_wrp;
   fy_aff=cy(n_good+1:n_good+3)'*[ones(1,nsamp1); X'];
   fy_wrp=cy(1:n_good)'*U;
   fy=fy_aff+fy_wrp;

   Z=[fx; fy]';
   
   % compute the mean squared error between synthetic warped image
   % and estimated warped image (using ground-truth correspondences
   % on TPS transformed image) 
   mse2=mean((y2(:,1)-Z(:,1)).^2+(y2(:,2)-Z(:,2)).^2);
   % Chui actually does mean of non-squared distance
   mse1=mean(sqrt((y2(:,1)-Z(:,1)).^2+(y2(:,2)-Z(:,2)).^2));
   disp(['error = ' num2str(mse1)])

   if display_flag
      figure(4)
      plot(Z(:,1),Z(:,2),'b+',Y(:,1),Y(:,2),'ro');
      title(['recovered TPS transformation (k=' int2str(k) ', \lambda_o=' num2str(beta_init*r^(k-1)) ', I_f=' num2str(E) ', error=' num2str(mse1) ')']) 
      % show warped coordinate grid
      fx_aff=cx(n_good+1:n_good+3)'*[ones(1,M); x'; y'];
      d2=dist2(X3b,[x y]);
      fx_wrp=cx(1:n_good)'*(d2.*log(d2+eps));
      fx=fx_aff+fx_wrp;
      fy_aff=cy(n_good+1:n_good+3)'*[ones(1,M); x'; y'];
      fy_wrp=cy(1:n_good)'*(d2.*log(d2+eps));
      fy=fy_aff+fy_wrp;
      hold on
      plot(fx,fy,'k.','markersize',1)
      hold off
      drawnow
   end
   
   % update Xk for the next iteration
   Xk=Z;
   
   % stop early if shape context score is sufficiently low
   if k==n_iter
      s=0;
   else
      k=k+1;
   end
end

