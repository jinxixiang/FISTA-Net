function [X, alpha] = tik_sep(B, PSF, center, alpha, BC)
%TIK_SEP Tikhonov image deblurring using the Kronecker decomposition.
%
%function [X, alpha] = tik_sep(B, PSF, center, alpha, BC)
%
%            X = tik_sep(B, PSF, center);
%            X = tik_sep(B, PSF, center, alpha);
%            X = tik_sep(B, PSF, center, alpha, BC);
%   [X, alpha] = tik_sep(B, PSF, ...);
%
%  Compute restoration using a Kronecker product decomposition and a
%  Tikhonov filter, with the identity matrix as the regularization operator.
%
%  Input:
%        B  Array containing blurred image.
%      PSF  Array containing the point spread function; same size as B.
%   center  [row, col] = indices of center of PSF.
%    alpha  Regularization parameter.
%             Default parameter chosen by generalized cross validation.
%       BC  String indicating boundary condition.
%             ('zero', 'reflexive', or 'periodic')
%           Default is 'zero'.
%
%  Output:
%        X  Array containing computed restoration.
%    alpha  Regularization parameter used to construct restoration.

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
   alpha = [];
end
if (nargin < 5)
   BC = 'zero';
end

%
% First compute the Kronecker product terms, Ar and Ac, where
% the blurring matrix  A = kron(Ar, Ac).  
% Note that if the PSF is not separable, this
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
if (ischar(alpha) | isempty(alpha))
  alpha = gcv_tik(s, bhat);
end 

%
% Compute the Tikhonov regularized solution.
%
D = abs(s).^2 + abs(alpha)^2;
bhat = s .* bhat;
xhat = bhat ./ D;
xhat = reshape(xhat, size(B));
X = Vc*xhat*Vr';