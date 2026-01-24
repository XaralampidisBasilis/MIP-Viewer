clear,clc

pkg load symbolic % OCTAVE version
pkg load optim % OCTAVE version

%% --------------------------------------------------------------------
%% Declare functions
%% --------------------------------------------------------------------
run("trilinear_bernstein_functions.m")

syms f000 f100 f010 f001 f011 f101 f110 f111 real
F8 = [f000 f100 f010 f001 f011 f101 f110 f111];

%% --------------------------------------------------------------------
%% Declare scenario surfaces z = 0 and z = min(1-x, 1-y) x,y in [0,1]
%% --------------------------------------------------------------------
p100 = [1, 0, 0];
p010 = [0, 1, 0];
p001 = [0, 0, 1];
p110 = [1, 1, 0];

[cB_A, ijk_A] = triangleBernsteinCoeffsFromTrilinearInterpolation(F8, p100, p110, p001);
[cB_B, ijk_B] = triangleBernsteinCoeffsFromTrilinearInterpolation(F8, p010, p110, p001);


for m=1:length(cB_A)
    fprintf("c_A(%d,%d,%d) = %s\n", ijk_A(m,1), ijk_A(m,2), ijk_A(m,3), char(cB_A(m)));
end

for m=1:length(cB_B)
    fprintf("c_B(%d,%d,%d) = %s\n", ijk_B(m,1), ijk_B(m,2), ijk_B(m,3), char(cB_B(m)));
end

cB_inv_max = reduceMaximaSubconvex([-cB_A(:); -cB_B(:)], F8, 1e-8, false);
cB_min = -cB_inv_max;

fprintf('\nReduced symbolic total minima:\n');
disp(cB_min);

% Results
% cB_min(1)  = (f000 + f100 + f101)/3;
% cB_min(2)  = (f000 + f001 + f101)/3;
% cB_min(3)  = (f000 + f010 + f011)/3;
% cB_min(4)  = (f000 + f001 + f011)/3;
% cB_min(5)  = (f010 + f100 + f111)/3;
% cB_min(6)  = (f000 + f011 + f101)/3;
% cB_min(7)  = f100;
% cB_min(8)  = f010;
% cB_min(9)  = f110;
% cB_min(10) = f001;
