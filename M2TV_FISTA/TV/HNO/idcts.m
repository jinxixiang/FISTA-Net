function y = idcts(x)
%IDCTS Model implementation of inverse discrete cosine transform.
%
%function y = idcts(x)
%
%         y = idcts(x);
%
%  Compute the inverse discrete cosine transform of x.  
%  This is a very simple implementation.  If the Signal Processing
%  Toolbox is available, then you should use the function idct.
%
%  Input:
%        x  column vector, or a matrix.  If x is a matrix then idcts
%           computes the IDCT of each column.
%
%  Output:
%        y  contains the inverse discrete cosine transform of x.

% Reference: See Chapter 4, 
%            "Deblurring Images - Matrices, Spectra, and Filtering"
%            by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
%            SIAM, Philadelphia, 2006.
%   
%   
% If an inverse FFT routine is available, then it can be used to compute
% the inverse DCT.  Since the inverse FFT is part of the standard MATLAB 
% distribution, we use this approach.  For further details on the formulas,
% see
%            "Computational Frameworks for the Fast Fourier Transform"
%            by C. F. Van Loan, SIAM, Philadelphia, 1992.
%
%            "Fundamentals of Digital Image Processing"
%            by A. Jain, Prentice-Hall, NJ, 1989.
%
[n, m] = size(x);

omega = exp(i*pi/(2*n));
d = sqrt(2*n) * omega.^(0:n-1).';
d(1) = d(1) * sqrt(2);
d = d(:,ones(1,m));

xt = [d.*x; zeros(1,m); -i*d(2:n,:).*flipud(x(2:n,:))];
yt = ifft(xt);
y = real(yt(1:n,:));