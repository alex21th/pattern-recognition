function  [xk,dk,alk,iWk,betak] = CG(x,f,g,Q,eps,kmax,almax,almin,rho,c1,c2,iW,icg,irc,nu,kmaxBLS,epsBLS)

k     = 0;        % Start iteration.         
d     = -g(x);    % First descent direction.
al    = almax;    % Maximum step length.
xk    = [x];      % Vector of solution points.
dk    = [];       % Vector of descent directions.
iWk   = [];       % Vector of Wolfe conditions.
alk   = [al];     % Vector of step lengths.

betak = [];       % Vector of beta coefficients.

while norm(g(x)) > eps & k < kmax
    
    % Maximum step length update.
    if k == 0 almax = 1;
    else      almax = 2*(f(x)-f(xk(:,k)))/(g(x)'*d); end
    
    % Backtracking linesearch.
    if iW == 0 al = -g(x)'*d/(d'*Q*d); iWi = 3;
    else       [al,iWi] = om_uo_BLSNW32(f,g,x,d,almax,c1,c2,kmaxBLS,epsBLS); end
    
    % Current solution point.
    x = x + al*d;
    k = k + 1;
    
    % Gradient setting.
    gx  = g(x);
    gxk = g(xk(1:end,k));
    
    % Update of beta coefficient.
    if icg == 1 beta = gx'*gx/norm(gxk)^2;
    else        beta = max(0,gx'*(gx-gxk)/norm(gxk)^2); end
    
    % Restart conditions.
    if     irc == 1 & mod(k,nu) == 0                d = -gx;
    elseif irc == 2 & abs(gx'*gxk)/norm(gx)^2 >= nu d = -gx; 
    else                                            d = -gx+beta*d; end
    
    % Parameters update.
    xk    = [xk x];
    dk    = [dk d];
    iWk   = [iWk iWi];
    alk   = [alk al];
    betak = [betak beta];

end

end