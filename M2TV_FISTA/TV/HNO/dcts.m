function y = dcts(x)
%DCTS Model implementation of discrete cosine transform.
%
%function y = dcts(x)
%
%         y = dcts(x);
%
%  Compute the discrete cosine transform of x.  
%  This is a very simple implementation.  If the Signal Processing
%  Toolbox is available, then you should use the function dct.
%
%  Input:
%        x  column vector, or a matrix.  If x is a matrix then dcts(x)
%           computes the DCT of each column
%
%  Output:
%        y  contains the discrete cosine transform of x.

% Reference: See Chapter 4, 
%            "Deblurring Images - Matrices, Spectra, and Filtering"
%            by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
%            SIAM, Philadelphia, 2006.
%   
% If an FFT routine is available, then it can be used to compute
% the DCT.  Since the FFT is part of the standard MATLAB distribution,
% we use this approach.  For further details on the formulas, see:

%
%            "Computational Frameworks for the Fast Fourier Transform"
%            by C. F. Van Loan, SIAM, Philadelphia, 1992.
%
%            "Fundamentals of Digital Image Processing"
%            by A. Jain, Prentice-Hall, NJ, 1989.
%
[n, m] = size(x);

omega = exp(-i*pi/(2*n));
d = [1/sqrt(2); omega.^(1:n-1).'] / sqrt(2*n);
d = d(:,ones(1,m));

xt = [x; flipud(x)];
yt = fft(xt);
y = real(d .* yt(1:n,:));