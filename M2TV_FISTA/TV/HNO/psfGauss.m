function [PSF, center] = psfGauss(dim, s)
%PSFGAUSS Array with point spread function for Gaussian blur.
%
%function [PSF, center] = psfGauss(dim, s)
%
%            PSF = psfGauss(dim);
%            PSF = psfGauss(dim, s);
%  [PSF, center] = psfGauss(...)
%
%  Construct a Gaussian blur point spread function. 
%
%  Input:
%      dim  Desired dimension of the PSF array.  For example,
%             PSF = psfGauss(60) or
%             PSF = psfGauss([60,60]) creates a 60-by-60 array,
%           while
%             PSF = psfGauss([40,50]) creates a 40-by-50 array.
%         s  Vector with standard deviations of the Gaussian along
%            the vertical and horizontal directions.
%            If s is a scalar then both standard deviations are s.
%              Default is s = 2.0.
%
%  Output:
%      PSF  Array containing the point spread function.
%   center  [row, col] gives index of center of PSF

% Reference: See Chapter 3, 
%            "Deblurring Images - Matrices, Spectra, and Filtering"
%            by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
%            SIAM, Philadelphia, 2006.
% Last revised October 8, 2007.

%
% Check number of inputs and set default parameters.
%
if (nargin < 1)
   error('dim must be given.')
end
l = length(dim);
if l == 1
  m = dim;
  n = dim;
else
  m = dim(1);
  n = dim(2);
end
if (nargin < 2)
  s = 2.0;
end
if length(s) == 1
  s = [s,s];
end

%
% Set up grid points to evaluate the Gaussian function.
%
x = -fix(n/2):ceil(n/2)-1;
y = -fix(m/2):ceil(m/2)-1;
[X,Y] = meshgrid(x,y);

%
% Compute the Gaussian, and normalize the PSF.
%
PSF = exp( -(X.^2)/(2*s(1)^2) - (Y.^2)/(2*s(2)^2) );
PSF = PSF / sum(PSF(:));

%
% Get center ready for output.
%
if nargout == 2
  [mm, nn] = find(PSF == max(PSF(:)));
  center = [mm(1), nn(1)];
end