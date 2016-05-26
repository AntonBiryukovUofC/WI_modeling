function  plot3_mt(minput,wt,sc);
% Usage: plot3_mt(minput,wt,sc);
%
% 3-D surface plot of source radiation pattern
%
% M is moment tensor (3 x 3 matrix format)
% wt is wavetype
%
% wt = 1 is P radiation pattern
% wt = 2 is S radiation pattern

vp = 4000;
vs = 2000;
rho = 2400;

M(1,1) = minput(1);
M(1,2) = minput(4);
M(1,3) = minput(5);
M(2,1) = M(1,2);
M(2,2) = minput(2);
M(2,3) = minput(6);
M(3,1) = M(1,3);
M(3,2) = M(2,3);
M(3,3) = minput(3);

[x,y,z] = sphere(50);
[phi,theta,r] = cart2sph(x,y,z);
clear r;

% Compute source, receiver positions

xs = [0,0,0];

xx = reshape(x,51*51,1);
yy = reshape(y,51*51,1);
zz = reshape(z,51*51,1);

xr = [xx,yy,zz];

for i = 1:2601;
    [d,A] = greenfun0a(xr(i,:),xs,vp,vs,rho,M);
    rp(i) = sqrt(d(1)^2+d(2)^2+d(3)^2);
    rs(i) = sqrt(d(4)^2+d(5)^2+d(6)^2);
    sp(i) = sign(dot(xr(i,:),d(1:3)/rp(i)));
    ss(i) = sign(d(3+sc));
end

if wt == 1;
    r = rp/max(abs(rp));
    c = r.*sp;
else
    r = rs/max(abs(rs));
    c = r.*ss;
end

rr = reshape(r,51,51);
cc = reshape(c,51,51);

[X,Y,Z] = sph2cart(phi,theta,rr);

surf(X,Y,Z,cc);
shading interp;

%colormap copper;
%shading interp;

if wt == 1;
    title('P-wave radiation pattern, positive out');
else
    if sc == 1;
       title('S-wave radiation pattern shaded by x');
    end
    if sc == 2;
       title('S-wave radiation pattern shaded by y');
    end
    if sc == 3;
       title('S-wave radiation pattern shaded by z');
    end
end

colorbar vert;
axis image;
xlabel('X');
ylabel('Y');
zlabel('Z');
caxis([-1,1]);

    