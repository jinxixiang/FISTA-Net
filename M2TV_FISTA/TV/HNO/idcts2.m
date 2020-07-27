function y = idcts2(x)
%IDCTS2 Model implementation of 2-D inverse discrete cosine transform.
%
%function y = idcts2(x)
%
%         y = idcts2(x);
%
%  Compute the inverse two-dimensional discrete cosine transform of x.  
%  This is a very simple implementation.  If the Image Processing Toolbox 
%  is available, then you should use the function idct2.
%
%  Input:
%        x  array
%
%  Output:
%        y  contains the two-dimensional inverse discrete cosine
%           transform of x.

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
% The two-dimensional inverse DCT is obtained by computing a one-dimensional 
% inverse DCT of the columns, followed by a one-dimensional inverse DCT of 
% the rows.
%
y = idcts(idcts(x).').';