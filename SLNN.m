%% Pattern recognition with neural networks (OM/GCED)
% Authors: _Alex Carrillo Alza, Oriol Narvaez Serrano_ (2nd, GCED)

%% 1. Generation of the training data set
% Let |Xtr,ytr| be the training data set.

num_target = [4];
noise_freq = 0.1;
tr_freq    = 0.5;       % Training parameters.
tr_p       = 500;
tr_seed    = 123456;

% Training data generation.
[Xtr, ytr] = om_uo_nn_dataset(tr_seed, tr_p, num_target, tr_freq, noise_freq);

%%
% Preview of some training data generated |Xtr|.

for i=1:10
    figure(1);
    subplot(2,5,i); numplot(Xtr,i);
end

%% 2. Definition of the loss function and its gradient
% If we define the sigmoid matrix of inputs $\sigma(X^{TR})$ and the row
% vector of residuals $y(X^{TR},w)$ as

sig = @(X)   1./(1+exp(-X));    % Sigmoid matrix.
y   = @(X,w) sig(w'*sig(X));    % Row vector of residuals.

%%
% Then, the value of the loss function $\tilde{L}$ and its gradient $\nabla
% \tilde{L}$ can be expressed as

la = 0.0;                                                               % Lambda.
L  = @(w) norm(y(Xtr,w)-ytr)^2 + (la*norm(w)^2)/2;                      % Loss function.
gL = @(w) 2*sig(Xtr)*((y(Xtr,w)-ytr).*y(Xtr,w).*(1-y(Xtr,w)))'+la*w;    % Gradient.

%% 3. Finding of the value _w*_ minimizing _L(w)_
% Now, the value of $w^{*}$ minimizing $\tilde{L}(w;X^{TR},y^{TR},\lambda)$
% can be found using the optimization routines developed during the course.

w = zeros(1, 35)';  % Weights of the SLNN.
h = []; Q = [];     % Unused hessian forms.

kmax    = 1000;  epsG   = 1.0e-06;
kmaxBLS = 30;   epsBLS = 1.0e-03;   % Parameters of methods.

almax = 2; almin = 10^-3; rho = 0.5; c1 = 0.01; c2 = 0.45;
iW = 2; isd = 3; icg = 1; irc = 1; nu = 0.1; delta = 0.1;

% Optimization with the chosen method.
[xk, dk, alk, iWk, betak] = DM(w,L,gL,h,Q,epsG,kmax,almax,almin,rho,c1,c2,iW,isd,icg,irc,nu,delta,kmaxBLS,epsBLS);

%% 4. Computation of the training accuracy
% The training accuracy is computated using the formula
% $\frac{100}{p} \cdot \sum_{j=1}^{p}{\delta[y(x_j^{TR},w^*)],y^{TR}}$

iWk = [iWk, NaN];
niter = size(xk,2); xo = xk(:,niter);     % Number of iterations performed.

tr_accuracy = 0;
train = round(y(Xtr, xk(:,niter)));       % Prediction of training data.
for j = 1:tr_p
    if train(j) == ytr(j) tr_accuracy = tr_accuracy + 1; end
end
tr_accuracy = 100/tr_p * tr_accuracy;     % Percentage of correct answers.

%% 5. Generation of the test data set
% Let |Xte,yte| be the test data set.

te_q    = 5000;
te_seed = 789101;   % Test parameters.

% Test data generation.
[Xte, yte] = om_uo_nn_dataset(te_seed, te_q, num_target, tr_freq, noise_freq);

%% 6. Computation of the test accuracy
% The test accuracy is computated using the formula
% $\frac{100}{q} \cdot \sum_{j=1}^{q}{\delta[y(x_j^{TE},w^*)],y^{TE}}$

te_accuracy = 0;
test = round(y(Xte, xk(:,niter)));        % Prediction of training data.
for j = 1:te_q
    if test(j) == yte(j) te_accuracy = te_accuracy + 1; end
end
te_accuracy = 100/te_q * te_accuracy;     % Percentage of correct answers.

%% 7. Computation of some analysis information
% The objective function and gradient are evaluated for every point.
% Also the rate of convergence and the descent condition is calculated
% for every iteration.

fk = []; gk = []; rk = [NaN]; gdk = [];
for k = 1:niter fk = [fk,L(xk(:,k))]; gk=[gk,gL(xk(:,k))]; end    % f(xk) and g(xk).
for k = 2:niter rk = [rk,(fk(k)-L(xo))/(fk(k-1)-L(xo))]; end      % Rate of convergence.
for k = 1:niter-1 gdk = [gdk,gk(:,k)'*dk(:,k)]; end               % Descent condition.
gdk = [gdk,0];

%% 8. Visualization of the results
% These are the results of the SLNN for the given target.

fprintf(':::::::::::::::::::::::::::::::::::::::::::::::::::: \n');
fprintf(' Pattern recognition with neural networks (OM/GCED) \n');
fprintf(':::::::::::::::::::::::::::::::::::::::::::::::::::: \n');
fprintf('Training data set generation. \n');
fprintf('    num_target = %d \n', num_target);
fprintf('    noise_freq = %.2f \n', noise_freq);
fprintf('    tr_freq    = %.2f \n', tr_freq);
fprintf('    tr_p       = %d \n', tr_p);
fprintf('    tr_seed    = %d \n', tr_seed);
fprintf('Optimization \n');
fprintf('    L2 reg. lambda = %.2f \n', la);
fprintf('    epsG    = %.1e,  kmax = %d \n', epsG, kmax);
fprintf('    kmaxBLS = %d,  epsBLS = %.1e \n', kmaxBLS, epsBLS);
fprintf('    c1 = %.2f,  c2 = %.2f,  isd = %d \n', c1, c2, isd);
fprintf(' - Algorithm: \n');
fprintf('    k     alpha   wolfe     g''*d      f(w)     ||g||     r \n');
fprintf('%5d %9.1e %5d %10.1e %9.1e %9.1e %9.1e \n', 1, alk(1), iWk(1), gdk(1), L(xk(1)), norm(gk(:,1)), rk(1));
fprintf('%5d %9.1e %5d %10.1e %9.1e %9.1e %9.1e \n', 2, alk(2), iWk(2), gdk(2), L(xk(2)), norm(gk(:,2)), rk(2));
fprintf('   ...     ...     ...       ...       ...       ...    ... \n');
fprintf('%5d %9.1e %5d %10.1e %9.1e %9.1e %9.1e \n', niter-1, alk(niter-1), iWk(niter-1), gdk(niter-2), L(xk(niter-1)), norm(gk(:,niter-1)), rk(niter-1));
fprintf('%5d %9.1e %5d %10.1e %9.1e %9.1e %9.1e \n', niter, alk(niter), iWk(niter), gdk(niter-1), L(xk(niter)), norm(gk(:,niter)), rk(niter));
fprintf('w* = [ \n       ');
for i = 1:length(w)
    fprintf('%8.1e,', xk(i,niter)); if mod(i,5) == 0 fprintf('\n       '); end
end
fprintf('                                              ] \n');
fprintf('Test data set generation. \n');
fprintf('    te_q    = %d \n', te_q);
fprintf('    te_seed = %d \n', te_seed);
fprintf('Accuracy. \n');
fprintf('    tr_accuracy = %.1f \n', tr_accuracy);
fprintf('    te_accuracy = %.1f \n', te_accuracy);
fprintf('\n');

figure(2);
subplot(1,2,1);
plot(1:niter, rk);
title('Rate of convergence');
xlabel('k');
ylabel('r');
subplot(1,2,2);
numplot(xk, niter);

%% The code down below is just for the purpose of the BFGS analysis.
% PLEASE COMMENT THIS PART IF IT IS NOT USED.

% Plot of the representation of optimal weights for each number.
figure(3);
subplot(2,5,1);
numplot(xk(:,niter),1); Wk = [];
for i=2:10
    [tr_accuracy, te_accuracy, Wk] = BFGS_training(i, tr_accuracy, te_accuracy, Wk);
    subplot(2,5,i);
    numplot(Wk(:,end),1)
end

% Table of accuracies differences between all numbers.
target_number = {'1';'2';'3';'4';'5';'6';'7';'8';'9';'0'};
tr_accuracy = tr_accuracy';
te_accuracy = te_accuracy';
T = table(tr_accuracy,te_accuracy,'RowNames',target_number)

% Similarities between numbers 3, 8 and 9.
figure(4);
subplot(1,3,1);
numplot(Wk(:,2),1);
subplot(1,3,2);
numplot(Wk(:,7),1);
subplot(1,3,3);
numplot(Wk(:,8),1);
