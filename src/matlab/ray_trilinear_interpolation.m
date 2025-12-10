clc,clear

% trilinear_ray_symbolic_with_mapping.m
% pkg load symbolic

%% --------------------------------------------------------------------
%% 1. Declare symbols
%% --------------------------------------------------------------------
syms x y z t real

% corner values
syms f000 f100 f010 f001 f011 f101 f110 f111 real

% symmetric linear combinations of corner values
syms v000 v100 v010 v001 v011 v101 v110 v111 real

% Ray endpoints
syms ax ay az bx by bz dx dy dz syx szx real

F = [f000 f100 f010 f001 f011 f101 f110 f111];
V = [v000 v100 v010 v001 v011 v101 v110 v111];
A = [ax, ay, az];
B = [bx, by, bz];
D = [dx, dy, dz];

%% --------------------------------------------------------------------
%% 2. Trilinear interpolation f(x,y,z)
%% --------------------------------------------------------------------
f_xyz = f000 * (1-x) * (1-y) * (1-z) ...
      + f100 *    x  * (1-y) * (1-z) ...
      + f010 * (1-x) *    y  * (1-z) ...
      + f001 * (1-x) * (1-y) *    z  ...
      + f011 * (1-x) *    y  *    z  ...
      + f101 *    x  * (1-y) *    z  ...
      + f110 *    x  *    y  * (1-z) ...
      + f111 *    x  *    y  *    z;

%% --------------------------------------------------------------------
%% 3. Ray substitution r(t) = a(1-t) + b*t
%% --------------------------------------------------------------------
% x_t = ax*(1 - t) + bx*t;
% y_t = ay*(1 - t) + by*t;
% z_t = az*(1 - t) + bz*t;
x_t = ax + dx*t;
y_t = ay + dy*t;
z_t = az + dz*t;

f_t = simplify( subs(f_xyz, [x y z], [x_t, y_t, z_t]) );
f_t = collect(f_t, t);

%% Extract coefficients (efficient for GLSL)
[f_t_coeffs, f_t_terms] = coeffs(f_t, t);

disp("Mapped expression coefficients and terms");
disp([f_t_coeffs(:), f_t_terms(:)]);

%% --------------------------------------------------------------------
%% 4. Simplification patterns (YOUR REQUESTED SECTION)
%% --------------------------------------------------------------------

% Mapping:
% f000 = v000
% f100 = v100 + v000
% f010 = v010 + v000
% f001 = v001 + v000
% f011 = v011 + v001 + v010 + v000
% f101 = v101 + v001 + v100 + v000
% f110 = v110 + v010 + v100 + v000
% f111 = v111 + v011 + v101 + v110 + v100 + v010 + v001 + v000

% Reverse Mapping:
% v000 = f000
% v100 = f100 - f000
% v010 = f010 - f000
% v001 = f001 - f000
% v011 = f000 - f001 - f010 + f011
% v101 = f000 - f001 - f100 + f101
% v110 = f000 - f010 - f100 + f110
% v111 = f001 - f000 + f010 - f011 + f100 - f101 - f110 + f111

F2V = [
    v000, ...
    v100 + v000, ...
    v010 + v000, ...
    v001 + v000, ...
    v011 + v001 + v010 + v000, ...
    v101 + v001 + v100 + v000, ...
    v110 + v010 + v100 + v000, ...
    v111 + v011 + v101 + v110 + v100 + v010 + v001 + v000, ...
];

V2F = [
    f000, ...
    f100 - f000, ...
    f010 - f000, ...
    f001 - f000, ...
    f000 - f001 - f010 + f011, ...
    f000 - f001 - f100 + f101, ...
    f000 - f010 - f100 + f110, ...
    f001 - f000 + f010 - f011 + f100 - f101 - f110 + f111, ...
];

% Apply forward mapping
v_xyz = simplify( subs(f_xyz, F, F2V) );
v_t = simplify( subs(f_t, F, F2V) );
v_t = collect(v_t, [t, bx, by, bz, ax, ay, az dx dy dz]);

%% Extract coefficients (efficient for GLSL)
[v_t_coeffs, v_t_terms] = coeffs(v_t, t);

disp("Mapped expression coefficients and terms (a_coeffs, a_terms):");
disp([v_t_coeffs(:), v_t_terms(:)]);
