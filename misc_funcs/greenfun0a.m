function [d,A,upamp,usamp] = greenfun0a(xr,xs,vp,vs,rho,M);
% Usage: [d,A] = greenfun0a(xr,xs,vp,vs,rho,M);
%
% Program to calculate amplitude values for a homogeneous medium
%
% Input variables:
%
% xr,xs (vectors) - location of receiver and source (Cartesian)
% vp,vs,rho - parameters of the medium (SI units)
% M - 3x3 symmetric array defining the moment tensor
%
%
% Output:
%
% d - vector of amplitude values for that direction (6)
%     displacement, [px py pz sx sy sz]
%
% A - matrix of coefficients
%     

Ap = 1/(4*pi*rho*vp^3);
As = 1/(4*pi*rho*vs^3);

x = xr - xs;
r = sqrt(sum(x.*x));

gam = x/r;

up = [0,0,0];
A = [0,0,0,0,0,0
    0,0,0,0,0,0
    0,0,0,0,0,0
    0,0,0,0,0,0
    0,0,0,0,0,0
    0,0,0,0,0,0];

for i = 1:3;
   for j = 1:3;
       for k = 1:3;
           
           if j == 1 & k == 1
               l = 1;
           end
           if j == 2 & k == 2
               l = 2;
           end          
           if j == 3 & k == 3
               l = 3;
           end
           if j == 1 & k == 2;
               l = 4;
           end
           if j == 2 & k == 1;
               l = 4;
           end
           if j == 1 & k == 3;
               l = 5;
           end
           if j == 3 & k == 1;
               l = 5;
           end
           if j == 2 & k == 3;
               l = 6;
           end
           if j == 3 & k == 2;
               l = 6;
           end
           
               
           up(i) = up(i) + gam(i)*gam(j)*gam(k)*M(j,k);
           A(l,i) = A(l,i) + Ap*gam(i)*gam(j)*gam(k);
           
       end
   end
end

us = [0,0,0];

for i = 1:3;
    
   for j = 1:3;
       for k = 1:3;
        
           if j == 1 & k == 1
               l = 1;
           end
           if j == 2 & k == 2
               l = 2;
           end          
           if j == 3 & k == 3
               l = 3;
           end
           if j == 1 & k == 2;
               l = 4;
           end
           if j == 2 & k == 1;
               l = 4;
           end
           if j == 1 & k == 3;
               l = 5;
           end
           if j == 3 & k == 1;
               l = 5;
           end
           if j == 2 & k == 3;
               l = 6;
           end
           if j == 3 & k == 2;
               l = 6;
           end
           
           if i == j;
               A(l,i+3) = A(l,i+3) + As*(1.0 - gam(i)*gam(j))*gam(k);
               us(i) = us(i) + (1.0 - gam(i)*gam(j))*gam(k)*M(j,k);
           else
               A(l,i+3) = A(l,i+3) - As*gam(i)*gam(j)*gam(k);
               us(i) = us(i) - gam(i)*gam(j)*gam(k)*M(j,k);
           end
           
       end
   end
end

d = [up'*Ap
    us'*As];
upamp = sqrt(up(1)^2 + up(2)^2 + up(3)^2);
usamp = sqrt(us(1)^2 + us(2)^2 + us(3)^2);

A = A';