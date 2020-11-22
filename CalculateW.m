%% A script to calculate the dictionary with minimal coherence with D:
%  min_W \|W^T D\|^2 subject to diag(W^T D) = 1
%  This is an implement of Algorithm 1 in Appendix E.1 of(Liu et al, 2019).
%  Just run this script without any arguments.

%  Author: Jialin Liu, UCLA math department (danny19921123@gmail.com)
%  Last Modified: 2019-2-15

%% Note: if the algorithm diverges, then try a smaller step size: eta.

%% Script starts.
clear;

% load dictionary D
load('./D.mat','D');
[m,n] = size(D);

% Initialization
W = D;
f = func(D,D);

% Step size
eta = 0.1;

% Main iteration
fprintf('Calculation Starts...\n');
for t = 1: 1000
    
    % calculate residual and gradient
    res = D' * W - eye(n);
    gra = D * res;
    
    % gradient descent
    W_next = W - eta * gra;
    
    % projection
    W_next = proj(W_next,D);
    
    % calculate objective function value
    f_next = func(W_next,D);
    
    % stopping condition
    if abs(f-f_next)/f < 1e-12,  break; end
    
    % update
    W = W_next;
    f = f_next;
    
    % report function values
    if mod(t,50) == 0, fprintf('t: %d\t, func: %f\n', t, f); end
end

% save to file
save('W.mat','W');
fprintf('Calculation ends. Results are saved in W.mat.\n');

% visualization
visualization(D,W);


%% functions
function f = func(W,D)
% calculate function values
n = size(D,2);
res = D' * W - eye(n);
Q = ones(n,n)+eye(n)*(-1);
res = res .* sqrt(Q);
f = sum(sum(res.*res)); 
end

function W_next = proj(W,D)
% conduct projection
aw = diag(D'*W);
aw = repmat(aw',[size(D,1), 1]);
W_next = W + (1-aw).*D;
end

function visualization(D,W)
% function for visualizing the coherences between A and W
n = size(D,2);

res = D' * W - eye(n);
res0 = D' * D - eye(n);

figure ('Units', 'pixels', 'Position', [300 300 800 275]) ;

subplot(1,2,1);
histogram(res(~eye(n)),'BinWidth',1e-2);
hold on;
histogram(res0(~eye(n)),'BinWidth',1e-2);
title('off-diagonal');
legend('W','A');
hold off;

subplot(1,2,2);
histogram(res(logical(eye(n))),'BinWidth',1e-5);
hold on;
histogram(res0(logical(eye(n))),'BinWidth',1e-5);
hold off;
title('diagonal');

end
