function alpha = gcv_gtik(sa, sd, bhat)
%GCV_GTIK Choose GCV parameter for gtik_fft deblurring function.
%
%function alpha = gcv_gtik(sa, sd, bhat)
%
%         alpha = gcv_gtik(sa, sd, bhat);
%
%  This function uses generalized cross validation (GCV) to choose
%  a regularization parameter for generalized Tikhonov filtering.
%
%  Input:
%       sa  Vector containing singular or spectral values of the
%             blurring matrix.
%       sd  Vector containing singular of spectral values of the
%             regularization operator.
%     bhat  Vector containing the spectral coefficients of the blurred
%             image.
%
%  Output:
%    alpha  Regularization parameter.

% Reference: See Chapter 7, 
%            "Deblurring Images - Matrices, Spectra, and Filtering"
%            by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
%            SIAM, Philadelphia, 2006.

alpha = fminbnd(@GCV, min(abs(sa)), max(abs(sa)), [], sa, sd, bhat);

  function G = GCV(alpha, sa, sd, bhat)
    %
    %  This is a nested function that evaluates the GCV function for
    %  Tikhonov filtering.  It is called by fminbnd.
    %
    denom = abs(sa).^2 + alpha^2 * abs(sd).^2;
    %
    %  NOTE: It is possible to get division by zero if using a derivative
    %        operator for the regularization operator.
    %
    phi_d = abs(sd).^2 ./ denom;
    G = sum(abs(bhat.*phi_d).^2) / (sum(phi_d)^2);
  end
  
end