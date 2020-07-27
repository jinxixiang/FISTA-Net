function [X, alpha] = gtik_fft(B, PSF, center, Pd, alpha)
%GTIK_FFT Generalized Tikhonov image deblurring using the FFT algorithm.
%
%function [X, alpha] = gtik_fft(B, PSF, center, Pd, alpha)
%
%            X = gtik_fft(B, PSF, center);
%            X = gtik_fft(B, PSF, center, Pd);
%            X = gtik_fft(B, PSF, center, Pd, alpha);
%   [X, alpha] = gtik_fft(B, PSF, ...);
%
%  Compute restoration using an FFT-based Tikhonov filter, 
%  with a general regularization operator.
%
%  Input:
%        B  Array containing blurred image.
%      PSF  Array containing the point spread function; same size as B.
%   center  [row, col] = indices of center of PSF.
%       Pd  3-by-3 stencil for regularization operator, e.g.:
%                [0 0 0;0  1 0;0 0 0] <-- identity
%                [0 0 0;1 -1 0;0 0 0] <-- 1st deriv of rows
%                [0 1 0;0 -1 0;0 0 0] <-- 1st deriv of cols
%                [0 0 0;1 -2 1;0 0 0] <-- 2nd deriv of rows
%                [0 1 0;0 -2 0;0 1 0] <-- 2nd deriv of cols
%                [0 1 0;1 -4 1;0 1 0] <-- Laplacian
%             Default is to use the identity.
%    alpha  Regularization parameter.
%             Default parameter chosen by generalized cross validation.
%
%  Output:
%        X  Array containing computed restoration.
%    alpha  Regularization parameter used to construct restoration.

% Reference: See Chapter 7, 
%            "Deblurring Images - Matrices, Spectra, and Filtering"
%            by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
%            SIAM, Philadelphia, 2006.

%
% Check number of inputs and set default parameters.
%
if (nargin < 3)
   error('B, PSF, and center must be given.')
end
if (nargin < 4)
   Pd = [0 0 0;0 1 0;0 0 0];
end
if (nargin < 5)
   alpha = [];
end

if any(size(Pd) ~= 3)
  error('Illegal stencil for regularization operator')
end
center_d = [2, 2];

%
% Compute the eigenvalues of the BCCB blurring matrix.
%
bhat = fft2(B);
bhat = bhat(:);
Sa = fft2( circshift(PSF, 1-center) );
sa = Sa(:);

%
% Compute the eigenvalues of the BCCB regularization operator.
%
Pd = padPSF(Pd, size(B));
Sd = fft2( circshift(Pd, 1-center_d) );
sd = Sd(:);

%
% If a regularization parameter is not given, use GCV to find one.
%
if (ischar(alpha) | isempty(alpha))
  alpha = gcv_gtik(sa, sd, bhat);
end 

%
% Compute the Tikhonov regularized solution.
%
D = conj(sa).*sa + abs(alpha)^2 * conj(sd).*sd;
bhat = conj(sa) .* bhat;
xhat = bhat ./ D;
xhat = reshape(xhat, size(B));
X = real(ifft2(xhat));