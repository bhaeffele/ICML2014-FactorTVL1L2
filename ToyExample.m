%Here we'll generate a toy dataset.  It is a movie of 200 frames, where
%each frame is 100x100 pixels.  The spatial components are 5 20x20
%rectangles going down the diagonal of the frame, and the temporal
%components are sinusoids of different frequencies.

%size of the data
m = 100;
n = 100;
t = 200;

%Make true spatial components
Z_true = zeros(m,n,5);

for i=1:5
    Z_true((i-1)*20+1:i*20,(i-1)*20+1:i*20,i) = 1;
end

%Reshape into vectors
Z_true = reshape(Z_true,m*n,5);

%Make true temporal components
A_true = zeros(t,5);

for i=1:5
    A_true(:,i) = sin((0:t-1)*2*pi/t*i)';
end

%Make true data volume
X_true = A_true*Z_true';

%Add noise
X = X_true + 3*randn(size(X_true));

%%%%%%%
%Show a movie of the dataset.
for i=1:t
   imagesc([reshape(X(i,:),m,n) reshape(X_true(i,:),m,n)],[-5 5]);
   axis image; colormap gray;
   title('   Raw Data      |      True Signal');
   drawnow;
end
%%%%%%%

%Setup regularization parameters
param.lam = 1.25;
param.kA = [0 0 1];
param.kZ = [1 1 1];
param.idx = GetLatticeIndex(m,n);
param.rank = 25;

figure;
tic;
[A,Z,~,obj,A_norm,Z_norm,~,L_stats] = FactorTVL1L2_v1(X,[],[],param,100);
toc;

[~,si] = sort(A_norm.*Z_norm,'descend');

%show the first 5 most significant spatial components.
for i=1:5
    subplot(2,3,i);
    %Since there's a bilinearity between A and Z we can scale Ai and Zi by
    %reciprocals of each other withouth changing the solution.  Here we'll
    %scale the components of Z so that the non-zero components have a
    %median of 1.
    scl(i) = median(nonzeros(Z(:,si(i))));
    imagesc(reshape(Z(:,si(i))/scl(i),m,n));
    axis image
    title(['Spatial Component ' num2str(i)]);
end

disp('Showing spatial components estimated without a temporal dictionary');
disp('Press key to continue...');
pause;

%Now show the first 5 temporal components.  Notice that they are of
%slightly smaller magnitude than the true temporal components because of
%the shrinkage from the regularizer.

for i=1:5
    subplot(2,3,i);
    %Find which component this is
    [~,mi] = max(A_true'*A(:,si(i))*scl(i));
    plot(A(:,si(i))*scl(i),'b');
    hold on
    plot(A_true(:,mi),'r');
    title(['Temporal Component ' num2str(i)]);
end

%Make a dummy figure just for the legend
subplot(2,3,6);
plot([0 1],[0 0],'b');
hold on
plot([0 1],[1 1],'r');
legend('Estimated','True');
title('Temporal Components, No Dictionary');

disp('Showing temporal components estimated without a temporal dictionary');
disp('Press key to continue...');
pause


%Now run again but with a temporal dictionary.  Here the dictionary is low 
%frequency components from a fourier basis.  Obviously this makes things 
%easier since the true signals are exactly single elements from the
%dictionary.

%Make the dictionary
Df = MakeFourierD(t,13);

%Also adjust the regularization parameters to reflect the fact that we now
%expect A to be sparse.

param.kA = [1 0 1];

figure;
tic;
[A_dict,Z_dict,~,obj_dict,A_norm_dict,Z_norm_dict,~,~] = ...
    FactorTVL1L2_v1(X,Df,[],param,100);
toc;

[~,si_dict] = sort(A_norm_dict.*Z_norm_dict,'descend');

%show the first 5 most significant spatial components.
for i=1:5
    subplot(2,3,i);
    scl_dict(i) = median(nonzeros(Z_dict(:,si_dict(i))));
    imagesc(reshape(Z_dict(:,si_dict(i))/scl_dict(i),m,n));
    axis image
    title(['Spatial Component ' num2str(i)]);
end

disp('Showing spatial components estimated using a temporal dictionary');
