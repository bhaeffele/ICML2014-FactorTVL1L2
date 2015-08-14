function B = MakeFourierB(t_pts,num_fourier)
%B = MakeFourierB(t_pts,num_fourier)
%
%Makes a fourier basis with t_pts points for the first num_fourier
%compenents.


B = fft(eye(t_pts,num_fourier));
B = [real(B(:,1:num_fourier)) imag(B(:,2:num_fourier))];
        
B = B./repmat(sqrt(sum(B.^2)),t_pts,1);