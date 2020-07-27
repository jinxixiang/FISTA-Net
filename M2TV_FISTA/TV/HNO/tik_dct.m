function [X, alpha] = tik_dct(B, PSF, center, alpha)
%TIK_DCT Tikhonov image deblurring using the DCT algorithm.
%
%function [X, alpha] = tik_dct(B, PSF, center, alpha)
%
%            X = tik_dct(B, PSF, center);
%            X = tik_dct(B, PSF, center, alpha);
%   [X, alpha] = tik_dct(B, PSF, ...);
%
%  Compute restoration using a DCT-based Tikhonov filter, 
%  with the identity matrix as the regularization operator.
%
%  Input:
%        B  Array containing blurred image.
%      PSF  Array containing the point spread function; same size as B.
%   center  [row, col] = indices of center of PSF.
%    alpha  Regularization parameter.
%             Default parameter chosen by generalized cross validation.
%
%  Output:
%        X  Array containing computed restoration.
%    alpha  Regularization parameter used to construct restoration.

% Reference: See Chapter 6, 
%            "Deblurring Images - Matrices, Spectra, and Filtering"
%            by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
%            SIAM, Philadelphia, 2006.
% Last revised April 27, 2007.

%
% Check number of inputs and set default parameters.
%
if (nargin < 3)
   error('B, PSF, and center must be given.')
end
if (nargin < 4)
   alpha = [];
end

%
% Use the DCT to compute the eigenvalues of the symmetric 
% BTTB + BTHB + BHTB + BHHB blurring matrix.
%
% Check to see if the built-in dct2 function is available; if not, 
% use our simple codes.
e1 = zeros(size(PSF)); e1(1,1) = 1;
if exist('dct2') == 2
  bhat = dct2(B);
  S = dct2( dctshift(PSF, center) ) ./ dct2(e1);
else
  bhat = dcts2(B);
  S = dcts2( dctshift(PSF, center) ) ./ dcts2(e1);
end

%
% If a regularization parameter is not given, use GCV to find one.
%
bhat = bhat(:);
s = S(:);
if (ischar(alpha) | isempty(alpha))
  alpha = gcv_tik(s, bhat);
end 

%
% Compute the Tikhonov regularized solution.
%
D = s.^2 + abs(alpha)^2;
bhat = s .* bhat;
xhat = bhat ./ D;
xhat = reshape(xhat, size(B));
% Check again to see if the built-in dct2 function is available.
if exist('dct2') == 2
  X = idct2(xhat);
else
  X = idcts2(xhat);
end