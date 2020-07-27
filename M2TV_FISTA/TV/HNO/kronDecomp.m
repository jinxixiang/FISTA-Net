function [Ar, Ac] = kronDecomp(P, center, BC)
%KRONDECOMP Kronecker product decomposition of a PSF array
%
%function [Ar, Ac] = kronDecomp(P, center, BC)
%
%      [Ar, Ac] = kronDecomp(P, center);
%      [Ar, Ac] = kronDecomp(P, center, BC);
%
%  Compute terms of Kronecker product factorization A = kron(Ar, Ac),
%  where A is a blurring matrix defined by a PSF array.  The result is
%  an approximation only, if the PSF array is not rank-one.
%
%  Input:
%        P  Array containing the point spread function.
%   center  [row, col] = indices of center of PSF, P.
%       BC  String indicating boundary condition.
%             ('zero', 'reflexive', or 'periodic')
%           Default is 'zero'.
%
%  Output:
%   Ac, Ar  Matrices in the Kronecker product decomposition.  Some notes:
%             * If the PSF, P is not separable, a warning is displayed 
%               indicating the decomposition is only an approximation.
%             * The structure of Ac and Ar depends on the BC:
%                 zero      ==> Toeplitz
%                 reflexive ==> Toeplitz-plus-Hankel
%                 periodic  ==> circulant

% Reference: See Chapter 4, 
%            "Deblurring Images - Matrices, Spectra, and Filtering"
%            by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
%            SIAM, Philadelphia, 2006.

%
% Check inputs and set default parameters.
%
if (nargin < 2)
   error('P and center must be given.')
end
if (nargin < 3)
   BC = 'zero';
end

%
% Find the two largest singular values and corresponding singular vectors
% of the PSF -- these are used to see if the PSF is separable.
%
[U, S, V] = svds(P, 2);
if ( S(2,2) / S(1,1) > sqrt(eps) )
  warning('The PSF, P is not separable; using separable approximation.')
end

% 
% Since the PSF has nonnegative entries, we would like the vectors of the
% rank-one decomposition of the PSF to have nonnegative components.  That
% is, the singular vectors corresponding to the largest singular value of P
% should have nonnegative entries. The next few statements check this, and 
% change sign if necessary.
%
minU = abs(min(U(:,1)));
maxU = max(abs(U(:,1)));
if minU == maxU
  U = -U;
  V = -V;
end

% 
% The matrices Ar and Ac are defined by vectors r and c, respectively.
% These vectors can be computed as follows:
%
c = sqrt(S(1,1))*U(:,1);
r = sqrt(S(1,1))*V(:,1);

%
% The structure of Ar and Ac depends on the imposed boundary condition.
%
switch BC
  case 'zero'
    % Build Toeplitz matrices here
    Ar = buildToep(r, center(2));
    Ac = buildToep(c, center(1));
  case 'reflexive'
    % Build Toeplitz-plus-Hankel matrices here
    Ar = buildToep(r, center(2)) + buildHank(r, center(2));
    Ac = buildToep(c, center(1)) + buildHank(c, center(1));
  case 'periodic'
    % Build circulant matrices here
    Ar = buildCirc(r, center(2));
    Ac = buildCirc(c, center(1));
  otherwise
    error('Invalid boundary condition.')
end



function T = buildToep(c, k)
%
%  Build a banded Toeplitz matrix from a central column and an index
%  denoting the central column.
%
n = length(c);
col = zeros(n,1);
row = col';
col(1:n-k+1,1) = c(k:n);
row(1,1:k) = c(k:-1:1)';
T = toeplitz(col, row);


function C = buildCirc(c, k)
%
%  Build a banded circulant matrix from a central column and an index
%  denoting the central column.
%
n = length(c);
col = [c(k:n); c(1:k-1)];
row = [c(k:-1:1)', c(n:-1:k+1)'];
C = toeplitz(col, row);


function H = buildHank(c, k)
%
%  Build a Hankel matrix for separable PSF and reflexive boundary
%  conditions.
%
n = length(c);
col = zeros(n,1);
col(1:n-k) = c(k+1:n);
row = zeros(n,1);
row(n-k+2:n) = c(1:k-1);
H = hankel(col, row);