clear,clc

pkg load symbolic % OCTAVE version
pkg load optim % OCTAVE version

%% --------------------------------------------------------------------
%% Declare functions
%% --------------------------------------------------------------------

syms f000 f100 f010 f001 f011 f101 f110 f111 real
F8 = [f000 f100 f010 f001 f011 f101 f110 f111];


% --------------------------------------------------------------------
% Declare scenario x = min(1-y, 1-z) x in [0, y], y,z in [0,1], 
% --------------------------------------------------------------------
% p010 = [0, 1, 0];
% p011 = [0, 1, 1];
% p001 = [0, 0, 1];
% pmmm = [0.5, 0.5, 0.5];

% [cB_B, ijk_B] = triangleBernsteinCoeffsFromTrilinearInterpolation(F8, p010, pmmm, p011);
% [cB_C, ijk_C] = triangleBernsteinCoeffsFromTrilinearInterpolation(F8, p011, pmmm, p001);


% for m=1:length(cB_B)
%     fprintf("c_B(%d,%d,%d) = %s\n", ijk_B(m,1), ijk_B(m,2), ijk_B(m,3), char(cB_B(m)));
% end

% for m=1:length(cB_C)
%     fprintf("c_C(%d,%d,%d) = %s\n", ijk_C(m,1), ijk_C(m,2), ijk_C(m,3), char(cB_C(m)));
% end

% cB_inv_max = reduceMaximaSubconvex([-cB_B(:); -cB_C(:)], F8, 1e-8, false);
% cB_min = -cB_inv_max;

% fprintf('\nReduced symbolic total minima:\n');
% disp(cB_min);

% % Results
% % cB_min(1)  = (f000 + f001 + f010 + f011 + f100 + f101 + f110 + f111)/8;
% % cB_min(2)  = (f001 + f100 + f111 + 2*f000 + 2*f011 + 2*f110 + 3*f010)/12;
% % cB_min(3)  = (f000 + f101 + f110 + 2*f001 + 2*f010 + 2*f111 + 3*f011)/12;
% % cB_min(4)  = (f010 + f100 + f111 + 2*f000 + 2*f011 + 2*f101 + 3*f001)/12;
% % cB_min(5)  = (f000 + f011 + f110 + 3*f010)/6;
% % cB_min(6)  = (f001 + f010 + f111 + 3*f011)/6;
% % cB_min(7)  = (f000 + f011 + f101 + 3*f001)/6;
% % cB_min(8)  = f010;
% % cB_min(9)  = f011;
% % cB_min(10) = f001;

%% --------------------------------------------------------------------
%% Declare scenario x = min(1-y, 1-z) x in [0, y], y,z in [0,1], 
%% --------------------------------------------------------------------
% p010 = [0, 1, 0];
% p011 = [0, 1, 1];
% p001 = [0, 0, 1];
% pmm0 = [0.5, 0.5, 0];
% pmmm = [0.5, 0.5, 0.5];

% [cB_A, ijk_A] = triangleBernsteinCoeffsFromTrilinearInterpolation(F8, p010, pmmm, pmm0);
% [cB_B, ijk_B] = triangleBernsteinCoeffsFromTrilinearInterpolation(F8, p010, pmmm, p011);
% [cB_C, ijk_C] = triangleBernsteinCoeffsFromTrilinearInterpolation(F8, p011, pmmm, p001);


% for m=1:length(cB_A)
%     fprintf("c_A(%d,%d,%d) = %s\n", ijk_A(m,1), ijk_A(m,2), ijk_A(m,3), char(cB_A(m)));
% end

% for m=1:length(cB_B)
%     fprintf("c_B(%d,%d,%d) = %s\n", ijk_B(m,1), ijk_B(m,2), ijk_B(m,3), char(cB_B(m)));
% end

% for m=1:length(cB_C)
%     fprintf("c_C(%d,%d,%d) = %s\n", ijk_C(m,1), ijk_C(m,2), ijk_C(m,3), char(cB_C(m)));
% end

% cB_inv_max = reduceMaximaSubconvex([-cB_A(:); -cB_B(:); -cB_C(:)], F8, 1e-8, false);
% cB_min = -cB_inv_max;

% fprintf('\nReduced symbolic total minima:\n');
% disp(cB_min);

% % Results
% % cB_min(1)  = (f000 + f001 + f010 + f011 + f100 + f101 + f110 + f111)/8;
% % cB_min(2)  = (2*f000 + 1*f001 + 3*f010 + 2*f011 + 1*f100 + 2*f110 + 1*f111)/12;
% % cB_min(3)  = (1*f000 + 2*f001 + 2*f010 + 3*f011 + 1*f101 + 1*f110 + 2*f111)/12;
% % cB_min(4)  = (2*f000 + 3*f001 + 1*f010 + 2*f011 + 1*f100 + 2*f101 + 1*f111)/12;
% % cB_min(5)  = (3*f000 + 5*f010 + 1*f100 + 3*f110)/12;
% % cB_min(6)  = (f000 + f010 + f100 + f110)/4;
% % cB_min(7)  = (f000 + 3*f010 + f011 + f110)/6;
% % cB_min(8)  = (f001 + f010 + 3*f011 + f111)/6;
% % cB_min(9)  = (f000 + 3*f001 + f011 + f101)/6;
% % cB_min(10) = (f000 + 4*f010 + f110)/6;
% % cB_min(11) = f010;
% % cB_min(12) = f011;
% % cB_min(13) = f001;

%% --------------------------------------------------------------------
%% Declare scenario x = min(1-y, 1-z) x in [0, y], y,z in [0,1], 
%% --------------------------------------------------------------------
% p010 = [0, 1, 0];
% p011 = [0, 1, 1];
% p001 = [0, 0, 1];
% pmm0 = [0.5, 0.5, 0];
% pmmm = [0.5, 0.5, 0.5];

% [cB_A, ijk_A] = triangleBernsteinCoeffsFromTrilinearInterpolation(F8, p011, pmmm, p001);
% [cB_B, ijk_B] = quadrilateralBernsteinCoeffsFromTrilinearInterpolation(F8, p010, pmm0, pmmm, p011);


% for m=1:length(cB_A)
%     fprintf("c_A(%d,%d,%d) = %s\n", ijk_A(m,1), ijk_A(m,2), ijk_A(m,3), char(cB_A(m)));
% end

% for m=1:length(cB_B)
%     fprintf("c_B(%d,%d) = %s\n", ijk_B(m,1), ijk_B(m,2), char(cB_B(m)));
% end

% cB_inv_max = reduceMaximaSubconvex([-cB_A(:); -cB_B(:);], F8, 1e-8, false);
% cB_min = -cB_inv_max;

% fprintf('\nReduced symbolic total minima:\n');
% disp(cB_min);

% % Results
% % cB_min(1)  = (f000 + f001 + f010 + f011 + f100 + f101 + f110 + f111)/8;
% % cB_min(2)  = (2*f000 + 1*f001 + 4*f010 + 1*f011 + 1*f101 + 2*f110 + 1*f111)/12;
% % cB_min(3)  = (2*f000 + 3*f001 + 1*f010 + 2*f011 + 1*f100 + 2*f101 + 1*f111)/12;
% % cB_min(4)  = (2*f000 + 1*f001 + 3*f010 + 2*f011 + 1*f100 + 2*f110 + 1*f111)/12;
% % cB_min(5)  = (1*f000 + 2*f001 + 2*f010 + 3*f011 + 1*f101 + 1*f110 + 2*f111)/12;
% % cB_min(6)  = (3*f000 + 5*f010 + 1*f100 + 3*f110)/12;
% % cB_min(7)  = (f000 + 3*f001 + f011 + f101)/6;
% % cB_min(8)  = (f000 + f010 + f100 + f110)/4;
% % cB_min(9)  = (f000 + 3*f010 + f011 + f110)/6;
% % cB_min(10) = (f001 + 2*f010 + 2*f011 + f111)/6;
% % cB_min(11) = (f001 + f010 + 3*f011 + f111)/6;
% % cB_min(12) = (f000 + 4*f010 + f110)/6;
% % cB_min(13) = f001;
% % cB_min(14) = f010;
% % cB_min(15) = f011;

% --------------------------------------------------------------------
% Declare scenario x = min(1-y, 1-z) x in [0, y], y,z in [0,1], 
% --------------------------------------------------------------------
syms m real

p010 = [0, 1, 0];
p011 = [0, 1, 1];
p001 = [0, 0, 1];
p101 = [1, 0, 1];
pmmm = [m, m, m];

[cB_A, ijk_A] = triangleBernsteinCoeffsFromTrilinearInterpolation(F8, p010, pmmm, p011);
[cB_B, ijk_B] = triangleBernsteinCoeffsFromTrilinearInterpolation(F8, p011, pmmm, p001);
[cB_C, ijk_C] = triangleBernsteinCoeffsFromTrilinearInterpolation(F8, p101, pmmm, p001);


for m=1:length(cB_A)
    fprintf("c_A(%d,%d,%d) = %s\n", ijk_A(m,1), ijk_A(m,2), ijk_A(m,3), char(cB_A(m)));
end

for m=1:length(cB_B)
    fprintf("c_B(%d,%d,%d) = %s\n", ijk_B(m,1), ijk_B(m,2), ijk_B(m,3), char(cB_B(m)));
end

for m=1:length(cB_C)
    fprintf("c_C(%d,%d,%d) = %s\n", ijk_C(m,1), ijk_C(m,2), ijk_C(m,3), char(cB_C(m)));
end

cB = [cB_A(:); cB_B(:); cB_C(:)]

cB_inv_max = reduceMaximaSubconvex([-cB_A(:); -cB_B(:); -cB_C(:)], F8, 1e-8, false);
cB_min = -cB_inv_max;

fprintf('\nReduced symbolic total minima:\n');
disp(cB_min);

% Results
% cB_min(1)  = (f001 + f100 + f111 + 2*f000 + 2*f011 + 2*f110 + 3*f010)/12;
% cB_min(2)  = (f000 + f101 + f110 + 2*f001 + 2*f010 + 2*f111 + 3*f011)/12;
% cB_min(3)  = (f000 + f011 + f110 + 2*f001 + 2*f100 + 2*f111 + 3*f101)/12;
% cB_min(4)  = (f010 + f100 + f111 + 2*f000 + 2*f011 + 2*f101 + 3*f001)/12;
% cB_min(5)  = (f000 + f011 + f110 + 3*f010)/6;
% cB_min(6)  = (f001 + f010 + f111 + 3*f011)/6;
% cB_min(7)  = (f001 + f100 + f111 + 3*f101)/6;
% cB_min(8)  = (f000 + f011 + f101 + 3*f001)/6;
% cB_min(9)  =  f010;
% cB_min(10) =  f011;
% cB_min(11) =  f101;
% cB_min(12) =  f001;

