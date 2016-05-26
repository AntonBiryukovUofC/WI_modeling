function  [spratio,ang] = plot_spratio(minput,strike,dip,N);
% Usage: [spratio,ang] = plot_spratio(m,strike,dip,N);
%
% Plot SP amplitude ratio versus angle from fault normal
%
% m is moment tensor (1 x 6 format)
% strike and dip are in degrees, E-N,Zup co-ordinates
%
% N is an integer that controls the plot resolution
%
% N = 50 for a quick and dirty plot
% N = 500 for a high resolution plot

close all;

vp = 4000;
vs = 2000;
rho = 2400;

phi = (strike+180)*pi/180;
delta = dip*pi/180;

n = [-1*sin(delta)*sin(phi)
    sin(delta)*cos(phi)
    cos(delta)];

M(1,1) = minput(1);
M(1,2) = minput(4);
M(1,3) = minput(5);
M(2,1) = M(1,2);
M(2,2) = minput(2);
M(2,3) = minput(6);
M(3,1) = M(1,3);
M(3,2) = M(2,3);
M(3,3) = minput(3);

[x,y,z] = sphere(N);
[phi,theta,r] = cart2sph(x,y,z);
clear r;

% Compute source, receiver positions

xs = [0,0,0];

xx = reshape(x,(N+1)*(N+1),1);
yy = reshape(y,(N+1)*(N+1),1);
zz = reshape(z,(N+1)*(N+1),1);

xr = [xx,yy,zz];

for i = 1:(N+1)*(N+1);
    [d,A] = greenfun0a(xr(i,:),xs,vp,vs,rho,M);
    rp(i) = sqrt(d(1)^2+d(2)^2+d(3)^2);
    rs(i) = sqrt(d(4)^2+d(5)^2+d(6)^2);
    spratio(i) = rs(i)/rp(i);
    ang(i) = acosd(dot(xr(i,:),n));
end

figure;
plot(ang,spratio,'mo','MarkerFaceColor','m');
axis([0,180,0,20]);
xlabel('Angle from fault normal');
ylabel('S/P Amplitude Ratio');

figure;
hist(spratio(spratio<100));
xlabel('S/P Amplitude Ratio');
ylabel('Count');
title(num2str(mean(spratio(spratio<20))));



    