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
%% Declare scenario x = min(1-y, 1-z) y,z in [0,1], x in [0, y]
%% --------------------------------------------------------------------
p1 = [0, 1, 0];
p2 = [0, 1, 1];
p3 = [1, 0, 0];
p4 = [0, 0, 1];
p5 = [0.5, 0.5, 0];
p6 = [0.5, 0.5, 0.5];

[cB_A, ijk_A] = triangleBernsteinCoeffsFromTrilinearInterpolation(F8, p1, p6, p5);
[cB_B, ijk_B] = triangleBernsteinCoeffsFromTrilinearInterpolation(F8, p1, p6, p2);
[cB_C, ijk_C] = triangleBernsteinCoeffsFromTrilinearInterpolation(F8, p2, p6, p4);


for m=1:length(cB_A)
    fprintf("c_A(%d,%d,%d) = %s\n", ijk_A(m,1), ijk_A(m,2), ijk_A(m,3), char(cB_A(m)));
end

for m=1:length(cB_B)
    fprintf("c_B(%d,%d,%d) = %s\n", ijk_B(m,1), ijk_B(m,2), ijk_B(m,3), char(cB_B(m)));
end

for m=1:length(cB_C)
    fprintf("c_C(%d,%d,%d) = %s\n", ijk_C(m,1), ijk_C(m,2), ijk_C(m,3), char(cB_C(m)));
end

cB_inv_max = reduceMaximaSubconvex([-cB_A(:); -cB_B(:); -cB_C(:)], F8, 1e-8, false);
cB_min = -cB_inv_max;

fprintf('\nReduced symbolic total minima:\n');
disp(cB_min);

Results
% cB_min(1)  = (f000 + f001 + f010 + f011 + f100 + f101 + f110 + f111)/8;
% cB_min(2)  =  f000/6  + f001/12 + f010/4  + f011/6  + f100/12 + f110/6  + f111/12;
% cB_min(3)  =  f000/12 + f001/6  + f010/6  + f011/4  + f101/12 + f110/12 + f111/6;
% cB_min(4)  =  f000/6  + f001/4  + f010/12 + f011/6  + f100/12 + f101/6  + f111/12;
% cB_min(5)  =  f000/4  + (5*f010)/12 + f100/12 + f110/4;
% cB_min(6)  = (f000 + f010 + f100 + f110)/4;
% cB_min(7)  =  f000/6  + f010/2 + f011/6 + f110/6;
% cB_min(8)  =  f001/6  + f010/6 + f011/2 + f111/6;
% cB_min(9)  =  f000/6  + f001/2 + f011/6 + f101/6;
% cB_min(10) =  f000/6  + (2*f010)/3 + f110/6;
% cB_min(11) = f010;
% cB_min(12) = f011;
% cB_min(13) = f001;

