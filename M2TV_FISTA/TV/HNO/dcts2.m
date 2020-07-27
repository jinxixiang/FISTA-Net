function y = dcts2(x)
%DCTS2 Model implementation of 2-D discrete cosine transform.
%
%function y = dcts2(x)
%
%         y = dcts2(x);
%
%  Compute the two-dimensional discrete cosine transform of x.  
%  This is a very simple implementation.  If the Image Processing Toolbox 
%  is available, then you should use the function dct2.
%
%  Input:
%        x  array
%
%  Output:
%        y  contains the two-dimensional discrete cosine transform of x.

% Reference: See Chapter 4, 
%            "Deblurring Images - Matrices, Spectra, and Filtering"
%            by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
%            SIAM, Philadelphia, 2006.
%   
%            See also:
%            "Computational Frameworks for the Fast Fourier Transform"
%            by C. F. Van Loan, SIAM, Philadelphia, 1992.
%
%            "Fundamentals of Digital Image Processing"
%            by A. Jain, Prentice-Hall, NJ, 1989.
%
% The two-dimensional DCT is obtained by computing a one-dimensional DCT of
% the columns, followed by a one-dimensional DCT of the rows.
%
y = dcts(dcts(x).').';