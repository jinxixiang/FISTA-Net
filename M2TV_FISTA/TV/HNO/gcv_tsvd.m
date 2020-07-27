function tol = gcv_tsvd(s, bhat)
%GCV_TSVD Choose GCV parameter for TSVD image deblurring.
%
%function tol = gcv_tsvd(s, bhat)
%
%         tol = gcv_tsvd(s, bhat);
%
%  This function uses generalized cross validation (GCV) to choose
%  a truncation parameter for TSVD regularization.
%
%  Input:
%        s  Vector containing singular or spectral values.
%     bhat  Vector containing the spectral coefficients of the blurred
%             image.
%
%  Output:
%      tol  Truncation parameter; all abs(s) < tol should be truncated.

% Reference: See Chapter 6, 
%            "Deblurring Images - Matrices, Spectra, and Filtering"
%            by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
%            SIAM, Philadelphia, 2006.

%
% Sort absolute values of singular/spectral values in descending order.
%
[s, idx] = sort(abs(s)); s = flipud(s); idx = flipud(idx);
bhat = abs( bhat(idx) );
n = length(s);
%
% The GCV function G for TSVD has a finite set of possible values 
% corresponding to the truncation levels.  It is computed using
% rho, a vector containing the squared 2-norm of the residual for 
% all possible truncation parameters tol.
%
rho = zeros(n-1,1);
rho(n-1) = bhat(n)^2;
G = zeros(n-1,1);
G(n-1) = rho(n-1);
for k=n-2:-1:1
  rho(k) = rho(k+1) + bhat(k+1)^2;
  G(k) = rho(k)/(n - k)^2;
end
% Ensure that the parameter choice will not be fooled by pairs of
% equal singular values.
for k=1:n-2,
  if (s(k)==s(k+1))
     G(k) = inf;
  end
end
%
% Now find the minimum of the discrete GCV function.
%
[minG,reg_min] = min(G);
%
% reg_min is the truncation index, and tol is the truncation parameter.
% That is, any singular values < tol are truncated.
%
tol = s(reg_min(1));