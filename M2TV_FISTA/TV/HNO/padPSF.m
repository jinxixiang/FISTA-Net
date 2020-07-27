function P = padPSF(PSF, m, n)
%PADPSF Pad a PSF array with zeros to make it bigger.
%
%function P = padPSF(PSF, m, n)
%
%      P = padPSF(PSF, m);
%      P = padPSF(PSF, m, n);
%      P = padPSF(PSF, [m,n]);
%
%  Pad PSF with zeros to make it an m-by-n array. 
%
%  If the PSF is an array with dimension smaller than the blurred image,
%  then deblurring codes may require padding first, such as:
%      PSF = padPSF(PSF, size(B));
%  where B is the blurred image array.
%
%  Input:
%      PSF  Array containing the point spread function.
%     m, n  Desired dimension of padded array.  
%             If only m is specified, and m is a scalar, then n = m.
%
%  Output:
%        P  Padded m-by-n array.

% Reference: See Chapter 4, 
%            "Deblurring Images - Matrices, Spectra, and Filtering"
%            by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
%            SIAM, Philadelphia, 2006.

%
% Set default parameters.
%
if nargin == 2
  if length(m) == 1
    n = m;
  else
    n = m(2); 
    m = m(1);
  end
end

%
% Pad the PSF with zeros.
%
P = zeros(m, n);
P(1:size(PSF,1), 1:size(PSF,2)) = PSF;