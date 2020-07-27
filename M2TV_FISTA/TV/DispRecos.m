function DispRecos(result, pnum,interpstyle, autoscale)
% EMT Image reconstruction display function
% 
% Author: Jinxi XIANG;
% Courtesy to Yunjie Yang
% version 1.0
% 2020-03-20
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT: 
% twodmap: mapping from 1d distribution to 2d image
% pnum: pixel number of the imaging area, 32x32 or 64x64
% 
% interpstyle: interpolation method
%    'nearest'   Nearest neighbor interpolation
%    'linear'    Linear interpolation (default)
%    'natural'   Natural neighbor interpolation
% 
% autoscale: auto scale the result vector
%    1   autoscale
%    0   not autoscale

% Gen coordinates
x=(-1+1/pnum:2/pnum:1-1/pnum);
y=(-1+1/pnum:2/pnum:1-1/pnum);

% Get the points in circle
k=1;
for i=1:pnum
    for j=1:pnum
        if x(j)^2+y(i)^2<=1
            data(k,1)=x(j);
            data(k,2)=y(i);
            data(k,3)=result(k);
            k=k+1;
        end
    end
end
totalPixel=k-1;

% Autoscale the result
if (autoscale)
    data(:,3)=data(:,3)./max(result); % Autoscale display
end

% Interpolation
Fz=scatteredInterpolant(data(:,1),data(:,2),data(:,3),interpstyle);

pn=256; % Pixel number after interpolation
x1=(-1+1/pn:2/pn:1-1/pn);
y1=(-1+1/pn:2/pn:1-1/pn);
[xi,yi]=meshgrid(x1,y1);

zz=Fz(xi,yi);

% Transparent color outside the cicle
k=1;
for i=1:pn
    for j=1:pn
        if x1(j)^2+y1(i)^2 > 1
            zz(j,i)=NaN;
            k=k+1;
        end
    end
end

% Display the image
figure('color', 'white');
surf(x1,y1,zz);
view(90,90);
shading interp

a=max(result);
b=min(result);

myColorMap=load('myColorMap.txt');

caxis([b a]);
% colormap(jet(150));
colormap(myColorMap);
colorbar('Box','off','FontSize',12);
% colorbar('Box','off','FontSize',18,'YTickLabel',{'0','0.2','0.4','0.6','0.8','1.0'});
axis square
axis off

end