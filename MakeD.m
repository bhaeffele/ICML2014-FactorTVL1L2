function [D0] = MakeD(num_pts,dt,tau)
%[D0] = MakeD(num_pts,dt,tau)

exp_sig = exp(-(0:num_pts)*dt/tau);

D0 = conv2(exp_sig',eye(num_pts));
D0 = D0(1:num_pts,:);

D0 = D0./repmat(sqrt(sum(D0.^2)),num_pts,1);

end

