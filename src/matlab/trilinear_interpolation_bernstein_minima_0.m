clear,clc

pkg load symbolic % OCTAVE version
pkg load optim % OCTAVE version

%% --------------------------------------------------------------------
%% Declare functions
%% --------------------------------------------------------------------

syms f000 f100 f010 f001 f011 f101 f110 f111 real
F8 = [f000 f100 f010 f001 f011 f101 f110 f111];

%% --------------------------------------------------------------------
%% Declare scenario surface x = min(1-y, 1-z) x,y,z in [0,1]
%% --------------------------------------------------------------------
p010 = [0, 1, 0];
p011 = [0, 1, 1];
p111 = [1, 1, 1];

[cB, ijk] = triangleBernsteinCoeffsFromTrilinearInterpolation(F8, p010, p011, p111);


for m=1:length(cB)
    fprintf("c(%d,%d,%d) = %s\n", ijk(m,1), ijk(m,2), ijk(m,3), char(cB(m)));
end

cB_inv_max = reduceMaximaSubconvex(-cB, F8, 1e-8, false);
cB_min = -cB_inv_max;

fprintf('\nReduced symbolic total minima:\n');
disp(cB_min);

% Results
% cB_min(1)  = (f010 + f011 + f110)/3;
% cB_min(2)  = (f011 + f110 + f111)/3;
% cB_min(3)  = f010;
% cB_min(4)  = f011;
% cB_min(5)  = f111;