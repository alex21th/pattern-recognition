function [xk,dk,alk,iWk] = G(x,f,g,Q,epsG,kmax,almax,almin,rho,c1,c2,iW,kmaxBLS,epsBLS)

k   = 1;        % Start iteration.
al  = almax;    % Maximum step length.
xk  = [x];      % Vector of solution points.
dk  = [];       % Vector of descent directions.
iWk = [];       % Vector of Wolfe conditions.
alk = [al];     % Vector of step lengths.

while norm(g(x)) > epsG & k < kmax
    
    % Descent direction.
    d = -g(x);
    
    % Maximum step length update.
    if k == 1 almax = 1;
    else      almax = 2*(f(x)-f(xk(:,k-1)))/(g(x)'*d); end
    
    % Backtracking linesearch.
    if iW == 0 al = -g(x)'*d/(d'*Q*d); iWi = 3;
    else       [al,iWi] = om_uo_BLSNW32(f,g,x,d,almax,c1,c2,kmaxBLS,epsBLS); end
    
    % Current solution point.
    x = x + al*d;
    
    % Parameters update.
    k   = k + 1;
    xk  = [xk x];
    dk  = [dk d];
    iWk = [iWk iWi];
    alk = [alk al];
    
end

end

