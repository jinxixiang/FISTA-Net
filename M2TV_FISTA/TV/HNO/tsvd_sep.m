function [X, tol] = tsvd_sep(B, PSF, center, tol, BC)
%TSVD_SEP Truncated SVD image deblurring using Kronecker decomposition.
%
%function [X, tol] = tsvd_sep(B, PSF, center, tol, BC)
%
%           X = tsvd_sep(B, PSF, center);
%           X = tsvd_sep(B, PSF, center, tol);
%           X = tsvd_sep(B, PSF, center, tol, BC);
%    [X, tol] = tsvd_sep(B, PSF, ...);
%
%  Compute restoration using a Kronecker product decomposition and
%  a truncated SVD.
%
%  Input:
%        B  Array containing blurred image.
%      PSF  Array containing the point spread function; same size as B.
%   center  [row, col] = indices of center of PSF.
%      tol  Regularization parameter (truncation tolerance).
%             Default parameter chosen by generalized cross validation.
%       BC  String indicating boundary condition.
%             ('zero', 'reflexive', or 'periodic'; default is 'zero'.)
%
%  Output:
%        X  Array containing computed restoration.
%      tol  Regularization parameter used to construct restoration.

% Reference: See Chapter 6, 
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
   tol = [];
end
if (nargin < 5)
   BC = 'zero';
end

%
% First compute the Kronecker product terms, Ar and Ac, where
% A = kron(Ar, Ac).  Note that if the PSF is not separable, this
% step computes a Kronecker product approximation to A.
%
[Ar, Ac] = kronDecomp(PSF, center, BC);

%
% Compute SVD of the blurring matrix.
%
[Ur, Sr, Vr] = svd(Ar);
[Uc, Sc, Vc] = svd(Ac);

%
% If a regularization parameter is not given, use GCV to find one.
%
bhat = Uc'*B*Ur;
bhat = bhat(:);
s = kron(diag(Sr),diag(Sc));
if (ischar(tol) | isempty(tol))
  tol = gcv_tsvd(s, bhat(:));
end

%
% Compute the TSVD regularized solution.
%
Phi = (abs(s) >= tol);
idx = (Phi~=0);
Sfilt = zeros(size(Phi));
Sfilt(idx) = Phi(idx) ./ s(idx);
Bhat = reshape(bhat .*Sfilt , size(B));
X = Vc*Bhat*Vr';