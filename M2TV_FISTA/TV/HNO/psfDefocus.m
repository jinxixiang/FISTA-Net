function [PSF, center] = psfDefocus(dim, R)
%PSFDEFOCUS Array with point spread function for out-of-focus blur.
%
%function [PSF, center] = psfDefocus(dim, R)
%
%            PSF = psfDefocus(dim);
%            PSF = pdfDefocus(dim, R);
%  [PSF, center] = psfGauss(...)
%
%  Construct a defocus blur point spread function,  which is
%  1/(pi*R*R) inside a circle of radius R, and zero otherwise.
%
%  Input:
%      dim  Desired dimension of the PSF array.  For example,
%             PSF = psfDefocus(60) or
%             PSF = psfDefocus([60,60]) creates a 60-by-60 array,
%           while 
%             PSF = psfDefocus([40,50]) creates a 40-by-50 array.
%        R  Radius of defocus
%             Default is min(fix((dim+1)/2) - 1)
%
%  Output:
%      PSF  Array containing the point spread function.
%   center  [row, col] gives index of center of PSF

% Reference: See Chapter 3,
%            "Deblurring Images - Matrices, Spectra, and Filtering"
%            by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
%            SIAM, Philadelphia, 2006.
% Last revised April 25, 2007.

%
% Check inputs and set default parameters.
%
if (nargin < 1)
   error('dim  must be given.')
end
l = length(dim);
if l == 1
  m = dim;
  n = dim;
else
  m = dim(1);
  n = dim(2);
end
center = fix(([m,n]+1)/2);
if (nargin < 2)
   R = min(center - 1);
end

if R == 0
  % If R=0, then the PSF is a delta function, so the blurring matrix is 
  % the identity.
  PSF = zeros(m,n);
  PSF(center(1),center(2)) = 1;
else
  PSF = ones(m,n)/(pi*R*R);
  k = (1:max(m,n))';
  idx1 = meshgrid((k-center(1)).^2)' + meshgrid((k-center(2)).^2) > R^2;
  idx = idx1(1:m,1:n);
  PSF(idx) = 0;
end
PSF = PSF / sum(PSF(:));