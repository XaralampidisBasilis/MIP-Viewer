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
%% Declare scenario surface z = 0 in x,y in [0, 1]
%% --------------------------------------------------------------------
p1 = [0, 0, 0];
p2 = [0, 1, 0];
p3 = [1, 0, 0];
p4 = [1, 1, 0];

[cB, ij] = quadrilateralBernsteinCoeffsFromTrilinearInterpolation(F8, p1, p2, p3, p4);


for m=1:length(cB)
    fprintf("c(%d,%d) = %s\n", ij(m,1), ij(m,2), char(cB(m)));
end

cB_inv_max = reduceMaximaSubconvex([-cB(:)], F8, 1e-8, false);
cB_min = -cB_inv_max;

fprintf('\nReduced symbolic total minima:\n');
disp(cB_min);

% Results
% cB_min(1) = f000;
% cB_min(2) = f010;
% cB_min(3) = f100;
% cB_min(4) = f110;

