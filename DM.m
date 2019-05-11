function [xk,dk,alk,iWk,betak,Hk,tauk] = DM(x,f,g,h,Q,epsG,kmax,almax,almin,rho,c1,c2,iW,isd,icg,irc,nu,delta,kmaxBLS,epsBLS)

betak = 0;
Hk    = 0;
tauk  = 0; 

if isd == 1
    [xk,dk,alk,iWk] = G(x,f,g,Q,epsG,kmax,almax,almin,rho,c1,c2,iW,kmaxBLS,epsBLS);
    
elseif isd == 2
    [xk,dk,alk,iWk,betak] = CG(x,f,g,Q,epsG,kmax,almax,almin,rho,c1,c2,iW,icg,irc,nu,kmaxBLS,epsBLS);
    
elseif isd == 3
    [xk,dk,alk,iWk, Hk] = BFGS(x,f,g,Q,epsG,kmax,almax,almin,rho,c1,c2,iW,kmaxBLS,epsBLS);
    
end

end

