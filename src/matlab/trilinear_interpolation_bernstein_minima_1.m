clear,clc

pkg load symbolic % OCTAVE version
pkg load optim % OCTAVE version

%% --------------------------------------------------------------------
%% Declare Symbols
%% --------------------------------------------------------------------
syms f000 f100 f010 f001 f011 f101 f110 f111 real
F8 = [f000 f100 f010 f001 f011 f101 f110 f111];

%% --------------------------------------------------------------------
%% Declare scenario surface x = min(1-y, 1-z) x,y,z in [0,1]
%% --------------------------------------------------------------------
p010 = [0, 1, 0];
p011 = [0, 1, 1];
p100 = [1, 0, 0];
p001 = [0, 0, 1];

[cB_A, ijk_A] = triangleBernsteinCoeffsFromTrilinearInterpolation(F8, p010, p011, p100);
[cB_B, ijk_B] = triangleBernsteinCoeffsFromTrilinearInterpolation(F8, p001, p011, p100);


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
% cB_min(1)  = (f000 + f010 + f110)/3;
% cB_min(2)  = (f000 + f100 + f110)/3;
% cB_min(3)  = (f000 + f001 + f101)/3;
% cB_min(4)  = (f000 + f100 + f101)/3;
% cB_min(5)  = (f001 + f010 + f111)/3;
% cB_min(6)  = (f000 + f101 + f110)/3;
% cB_min(7)  = f010;
% cB_min(8)  = f001;
% cB_min(9)  = f011;
% cB_min(10) = f100;

%% --------------------------------------------------------------------
%% Declare scenario surface x = max(1-y, 1-z) x,y,z in [0,1]
%% --------------------------------------------------------------------
% p100 = [1, 0, 0];
% p101 = [1, 0, 1];
% p110 = [1, 1, 0];
% p011 = [0, 1, 1];

% [cB_A, ijk_A] = triangleBernsteinCoeffsFromTrilinearInterpolation(F8, p100, p101, p011);
% [cB_B, ijk_B] = triangleBernsteinCoeffsFromTrilinearInterpolation(F8, p100, p110, p011);


% for m=1:length(cB_A)
%     fprintf("c_A(%d,%d,%d) = %s\n", ijk_A(m,1), ijk_A(m,2), ijk_A(m,3), char(cB_A(m)));
% end

% for m=1:length(cB_B)
%     fprintf("c_B(%d,%d,%d) = %s\n", ijk_B(m,1), ijk_B(m,2), ijk_B(m,3), char(cB_B(m)));
% end

% cB_inv_max = reduceMaximaSubconvex([-cB_A(:); -cB_B(:)], F8, 1e-8, false);
% cB_min = -cB_inv_max;

% fprintf('\nReduced symbolic total minima:\n');
% disp(cB_min);

% % Results
% % cB_min(1)  = (f001 + f101 + f111)/3;
% % cB_min(2)  = (f001 + f011 + f111)/3;
% % cB_min(3)  = (f000 + f101 + f110)/3;
% % cB_min(4)  = (f001 + f010 + f111)/3;
% % cB_min(5)  = (f010 + f110 + f111)/3;
% % cB_min(6)  = (f010 + f011 + f111)/3;
% % cB_min(7)  = f101;
% % cB_min(8)  = f100;
% % cB_min(9)  = f110;
% % cB_min(10) = f011;
