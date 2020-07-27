function Ps = dctshift(PSF, center)
%DCTSHIFT Create array containing the first column of a blurring matrix.
%
%function Ps = dctshift(PSF, center)
%
%         Ps = dctshift(PSF, center);
%
%  Create an array containing the first column of a blurring matrix
%  when implementing reflexive boundary conditions.
%
%  Input:
%      PSF  Array containing the point spread function.
%   center  [row, col] = indices of center of PSF.
%
%  Output:
%       Ps  Array (vector) containing first column of blurring matrix.

% Reference: See Chapter 4, 
%            "Deblurring Images - Matrices, Spectra, and Filtering"
%            by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
%            SIAM, Philadelphia, 2006.

[m,n] = size(PSF);

if nargin == 1
  error('The center must be given.')
end

i = center(1);
j = center(2);
k = min([i-1,m-i,j-1,n-j]);

%
% The PSF gives the entries of a central column of the blurring matrix.
% The first column is obtained by reordering the entries of the PSF; for
% a detailed description of this reordering, see the reference cited
% above.
%
PP = PSF(i-k:i+k,j-k:j+k);

Z1 = diag(ones(k+1,1),k);
Z2 = diag(ones(k,1),k+1);

PP = Z1*PP*Z1' + Z1*PP*Z2' + Z2*PP*Z1' + Z2*PP*Z2';

Ps = zeros(m,n);
Ps(1:2*k+1,1:2*k+1) = PP;