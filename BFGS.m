
function [xk,dk,alk,iWk,Hk] = BFGS(x,f,g,Q,epsG,kmax,almax,almin,rho,c1,c2,iW,kmaxBLS,epsBLS)

k   = 0;                     % Start iteration.
al  = almax;                 % Maximum step length.
xk  = [x];                   % Vector of solution points.
dk  = [];                    % Vector of descent directions.
iWk = [];                    % Vector of Wolfe conditions.
alk = [al];                  % Vector of step lengths.

I   = eye(size(g(x),1));     % Identity matrix.
H   = I;                     % Initialization of H matrix.
Hk  = [];                    % Vector of H matrices.

while norm(g(x)) > epsG & k < kmax
    
    % Gradient setting.
    gx = g(x);
    
    % Descent direction.
    d  = -H * gx;
    
    % Maximum step length update.
    if k == 0 almax = 1;
    else      almax = 2*(f(x)-f(xk(:,k)))/(g(x)'*d); end
    
    % Backtracking linesearch.
    if iW == 0 al = -gx'*d/(d'*Q*d); iWi = 3;
    else       [al,iWi] = om_uo_BLSNW32(f,g,x,d,almax,c1,c2,kmaxBLS,epsBLS); end
    
    % Current solution point.
    x  = x + al*d;
    
    % Components definition.
    sk = al*d; yk = g(x)- gx; pk = 1/(yk'*sk);
    
    % BFGS update.
    H = (I-pk*sk*yk')*H*(I-pk*yk*sk')+pk*sk*sk';
    
    % Parameters update.
    k   = k + 1;
    dk  = [dk d];
    xk  = [xk x];
    Hk  = [Hk H]; 
    iWk = [iWk iWi];
    alk = [alk al];
   
end

end