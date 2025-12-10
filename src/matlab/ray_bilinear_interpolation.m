clc,clear

% bilinear_ray_symbolic_with_mapping.m
pkg load symbolic

%% --------------------------------------------------------------------
%% 1. Declare symbols
%% --------------------------------------------------------------------
syms x y t real

syms f00 f10 f01 f11 real
F = [f00 f10 f01 f11];

% Ray endpoints
syms ax ay bx by dx dy real

%% --------------------------------------------------------------------
%% 2. Bilinear interpolation f(x,y)
%% --------------------------------------------------------------------
f_xy = f00 * (1-x) * (1-y) ...
     + f10 *    x  * (1-y) ...
     + f01 * (1-x) *    y  ...
     + f11 *    x  *    y;

%% --------------------------------------------------------------------
%% 3. Ray substitution r(t) = a(1-t) + b*t
%% --------------------------------------------------------------------
x_t = ax*(1 - t) + bx*t;
y_t = ay*(1 - t) + by*t;
% x_t = ax + dx*t;
% y_t = ay + dy*t;

f_t = simplify( subs(f_xy, [x y], [x_t, y_t]) );
f_t = collect(f_t, t);

%% Extract coefficients (efficient for GLSL)
[f_coeffs, f_terms] = coeffs(f_t, [t]);

disp("Mapped expression coefficients and terms");
disp([f_coeffs(:), f_terms(:)]);

%% --------------------------------------------------------------------
%% 4. Simplification patterns
%% --------------------------------------------------------------------

% Declare v00 ... v11 instead of d...
syms v00 v10 v01 v11 real
V = [v00 v10 v01 v11];

%% Forward mapping: f-values → v-values
% Mapping:
% f00 = v00
% f10 = v10 + v00
% f01 = v01 + v00
% f11 = v11 + v10 + v01 + v00
F2V = [
    v00, ...
    v10 + v00, ...
    v01 + v00, ...
    v11 + v10 + v01 + v00, ...
];

%% Apply forward mapping
v_xy = simplify( subs(f_xy, F, F2V) );
v_t = simplify( subs(f_t, F, F2V) );
v_t = collect(v_t, [t, bx, by, ax, ay]);

%% Extract coefficients (efficient for GLSL)
[v_coeffs, v_terms] = coeffs(v_t, [t]);

disp("Mapped expression coefficients and terms (v_coeffs, v_terms):");
disp([v_coeffs(:), v_terms(:)]);

%% --------------------------------------------------------------------
%% 5. Reverse mapping (v-values → f-values)
%% --------------------------------------------------------------------
% v00 = f00
% v10 = f10 - f00
% v01 = f01 - f00
% v11 = f11 - f10 - f01 + f00
V2F = [
    f00, ...
    f10 - f00, ...
    f01 - f00, ...
    f11 - f10 - f01 + f00, ...
];

%% --------------------------------------------------------------------
%% 6. Bernstein form of v(t) on [0,1] and Bernstein coefficients
%% --------------------------------------------------------------------

% We assume v(t) is a quadratic polynomial in t:
% v(t) = c0 + c1*t + c2*t^2

c0 = simplify( v_coeffs(3) );
c1 = simplify( v_coeffs(2) );
c2 = simplify( v_coeffs(1) );

% Bernstein coefficients for degree-2 polynomial on [0,1]:
% v(t) = b0*(1 - t)^2 + b1*2*t*(1 - t) + b2*t^2
B0 = simplify(c0);
B1 = simplify(c0 + c1*1/2);
B2 = simplify(c0 + c1 + c2);

% Collect coefficients
B0 = collect(B0, V);
B1 = collect(B1, V);
B2 = collect(B2, V);

% Bernstein-form polynomial (for verification / export)
B_t = simplify( ...
    B0 * 1 * t^0 * (1 - t)^2 ...
  + B1 * 2 * t^1 * (1 - t)^1 ...
  + B2 * 1 * t^2 * (1 - t)^0);

disp("Bernstein coefficients b0, b1, b2:");
disp(B0);
disp(B1);
disp(B2);

disp("Sanity check v_t - b_t (should be 0):");
disp(simplify(v_t - B_t));


%% --------------------------------------------------------------------
%% 7. Bernstein coeffs min/max
%% --------------------------------------------------------------------

% Collect coefficients
B0_F = collect(expand(simplify(subs(B0, V, V2F))), F);
B1_F = collect(expand(simplify(subs(B1, V, V2F))), F);
B2_F = collect(expand(simplify(subs(B2, V, V2F))), F);
BF = [B0_F B1_F B2_F];
