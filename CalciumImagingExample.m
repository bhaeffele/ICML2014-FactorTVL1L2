load('Imgs1375.mat');

[m,n,t] = size(imgs);

%First show a movie of the data

disp('Raw data loaded, press key to show movie of data');
pause;

close all;

for i=1:t
    imshow(imgs(:,:,i),[0 1500]);
    title('Raw Data');
    drawnow;
end

disp('Press any key to continue');
pause;

close all;

dt = 0.2;  %frame rate for data
tau = -0.51/log(0.5); %decay constant for GCaMP5G (from Akerboom...Looger, 2012)

%magic regularization numbers
param.kA = [1 0 1];
param.kZ = [1 0.6 1];

%do non-negative factorization
param.posA = true;
param.posZ = true;

%Get the neighborhood structure of the pixels
param.idx = GetLatticeIndex(m,n);

%Initialize A to be every 5th column of an identity matrix
param.A_init = eye(t);
param.A_init = param.A_init(:,1:5:end);

%Make dictionary of decaying exponentials
Da = MakeD(t,dt,tau);

%Dictionary to represent background intensity
%(Slowest 3 real/imag pairs of a fourier basis)
Db = MakeFourierD(t,3);

%Reshape data so that the time series for each pixel is in the columns
imgs = reshape(imgs,m*n,t)';

%Calculate the mean (in time) of each pixel
mn_img = mean(imgs);

%Set lam to be approximately 0.5 the standard deviation of the noise.
%Lowering this level will capture more neurons with smaller signal to noise
%ratios at the expense of possibly having more noise in the factorized
%matrices (A,Z).
param.lam = 0.5*norm(imgs-repmat(mn_img,t,1),'fro')/sqrt(numel(imgs));

%Run
tic;
[A,Z,B,obj,Anrm,Znrm,Bnrm,Ls] = FactorTVL1L2_v1(imgs,Da,Db,param,50);
toc;

%Sort columns of A and Z by magnitude
[~,si] = sort(Anrm.*Znrm,'descend');
A = A(:,si);
Z = Z(:,si);
Anrm = Anrm(si);
Znrm = Znrm(si);

figure();
title('25 most significant spatial features');

%Display first 25 columns of Z.
for i=1:25
    subplot(5,5,i)
    imagesc(reshape(Z(:,i),m,n));
end

pause;

%Reconstruct estimated signal
F_rec = Da*A*Z'; %Estimated transients
B_rec = Db*B';   %Estimated background

%%Make a movie of the reconstructed signal 

close all;

figure;

for i=1:t
    imshow([reshape(imgs(i,:)-mn_img,[m,n]);
        reshape(F_rec(i,:),m,n)],[0 500]);
    title('Top: Raw data (mean subtracted). Bottom: Recovered calcium signal');
    mv(i) = getframe;
end
