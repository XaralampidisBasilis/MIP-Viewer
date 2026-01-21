clc,clear

pkg load symbolic % OCTAVE version
pkg load optim % OCTAVE version

%% --------------------------------------------------------------------
%% Declare symbols
%% --------------------------------------------------------------------

% corner values
% symmetric linear combinations of corner values
syms f000 f100 f010 f001 f011 f101 f110 f111 real
syms v000 v100 v010 v001 v011 v101 v110 v111 real

F = [f000 f100 f010 f001 f011 f101 f110 f111];
V = [v000 v100 v010 v001 v011 v101 v110 v111];

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

% Variables
syms x y z t real

r = [x, y, z];

% Ray endpoints
syms ax ay az bx by bz dx dy dz sx sy sz real

a = [ax, ay, az];
b = [bx, by, bz];
d = [dx, dy, dz];
s = [sx, sy, sz];

%% --------------------------------------------------------------------
%% Trilinear interpolation f(x,y,z)
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
%% Ray substitution r(t) 
%% --------------------------------------------------------------------

x_t = ax + dx*t;
y_t = ay + dy*t;
z_t = az + dz*t;

% x_t = bx - dx*(1-t);
% y_t = by - dy*(1-t);
% z_t = bz - dz*(1-t);

% x_t = ax*(1-t) + bx*t;
% y_t = ay*(1-t) + by*t;
% z_t = az*(1-t) + bz*t;

r_t = [x_t, y_t, z_t];

%% --------------------------------------------------------------------
%% Trilinear function over ray  
%% --------------------------------------------------------------------

f_t = simplify( subs(f_xyz, r, r_t) );
f_t = collect(f_t, t);

% Extract coefficients
[f_t_coeffs, f_t_terms] = coeffs(f_t, t);

%% --------------------------------------------------------------------
%% Bernstein Trilinear function over ray 
%% --------------------------------------------------------------------

% Bernstein coefficients
Bf_t_coeffs = [
    simplify(f_t_coeffs(4)), ...
    simplify(f_t_coeffs(4) + f_t_coeffs(3)*1/3), ...
    simplify(f_t_coeffs(4) + f_t_coeffs(3)*2/3 + f_t_coeffs(2)*1/3), ...
    simplify(f_t_coeffs(4) + f_t_coeffs(3) + f_t_coeffs(2) + f_t_coeffs(1)), ...
];

% Bernstein terms
Bf_t_terms = [ ...
    1 * t^0 * (1 - t)^3, ...
    3 * t^1 * (1 - t)^2, ...
    3 * t^2 * (1 - t)^1, ...
    1 * t^3 * (1 - t)^0, ...
];

% Bernstein trilinear function over ray
Bf_t = dot(Bf_t_coeffs, Bf_t_terms);

disp("Sanity check f(t) - Bf_t(t), should be 0:");
disp(simplify(f_t - Bf_t));

%% --------------------------------------------------------------------
%% Trilinear function derivative over ray  
%% --------------------------------------------------------------------

% Compute partial derivatives
fx = simplify(diff(f_xyz, x));
fy = simplify(diff(f_xyz, y));
fz = simplify(diff(f_xyz, z));

% Gradient vector
Gf_xyz = [fx, fy, fz];

% directional derivative
Df_xyz = dot(Gf_xyz, d);

% directional derivative over ray
Gf_t = simplify( subs(Gf_xyz, r, r_t) );
Df_t = simplify( subs(Df_xyz, r, r_t) );

% coefficients
[Df_t_coeffs, Df_t_terms] = coeffs(Df_t, t);

%% --------------------------------------------------------------------
%% Bernstein trilinear function derivative over ray  
%% --------------------------------------------------------------------

% Bernstein coefficients
BDf_t_coeffs = [
    simplify(Df_t_coeffs(3)), ...
    simplify(Df_t_coeffs(3) + Df_t_coeffs(2)/2), ...
    simplify(Df_t_coeffs(3) + Df_t_coeffs(2) + Df_t_coeffs(1)), ...
];

% Bernstein terms
BDf_t_terms = [
    1 * t^0 * (1 - t)^2, ...
    2 * t^1 * (1 - t)^1, ...
    1 * t^2 * (1 - t)^0, ...
];

% Bernstein trilinear derivative function over ray
BDf_t = dot(BDf_t_coeffs, BDf_t_terms);

disp("Sanity check Df(t) - BDf(t), should be 0:");
disp(simplify(Df_t - BDf_t));

%% --------------------------------------------------------------------
%% Simplification patterns
%% --------------------------------------------------------------------

v_xyz        = simplify( subs(f_xyz, F, F2V) );
v_t          = simplify( subs(f_t, F, F2V) );
v_t_coeffs   = simplify( subs(f_t_coeffs, F, F2V) );
Dv_xyz       = simplify( subs(Df_xyz, F, F2V) );
Dv_t         = simplify( subs(Df_t, F, F2V) );
Dv_t_coeffs  = simplify( subs(Df_t_coeffs, F, F2V) );
Bv_t_coeffs  = simplify( subs(Bf_t_coeffs, F, F2V) );
BDv_t_coeffs = simplify( subs(BDf_t_coeffs, F, F2V) );

%% --------------------------------------------------------------------
%% Find bernstein minima over the diagonal surface x+y+z=1 
%% --------------------------------------------------------------------

% FIX your line: simplify(subs, f_xyz, x, 1-y-z)  <-- wrong call
g_yz = simplify( subs(f_xyz, x, 1 - y - z) );
g_yz = expand(g_yz);
g_yz = collect(g_yz, [y z]);

% Degrees (biquadratic)
ny = 2;
nz = 2;

% ---- 1) Extract power-basis coefficient matrix A(k+1,l+1) for y^k z^l ----
A = sym(zeros(ny+1, nz+1));

% Octave symbolic: use symengine coeff extraction
for k = 0:ny
for l = 0:nz

    % coefficient of y^k is a polynomial in z
    g_yk = feval(symengine, 'coeff', g_yz, y, k);
        A(k+1,l+1) = feval(symengine, 'coeff', g_yk, z, l);
    end
end

disp("Power-basis coefficient matrix A (rows y^k, cols z^l):");
disp(A);

% ---- 2) Build 1D power->Bernstein transform matrices (degree 2) ----
% For degree n: M(i,k)=C(i,k)/C(n,k) for i>=k, else 0
My = sym(zeros(ny+1, ny+1));
for i = 0:ny
    for k = 0:i
        My(i+1,k+1) = nchoosek(i,k) / nchoosek(ny,k);
    end
end

Mz = sym(zeros(nz+1, nz+1));
for j = 0:nz
    for l = 0:j
        Mz(j+1,l+1) = nchoosek(j,l) / nchoosek(nz,l);
    end
end

% ---- 3) Tensor-product conversion ----
B = simplify(My * A * transpose(Mz));

disp("2D Bernstein coefficient net B (rows i for y, cols j for z):");
disp(B);

% ---- 4) Sanity check: rebuild polynomial from Bernstein net ----
By = sym(zeros(ny+1,1));
for i=0:ny
    By(i+1) = nchoosek(ny,i) * y^i * (1-y)^(ny-i);
end

Bz = sym(zeros(nz+1,1));
for j=0:nz
    Bz(j+1) = nchoosek(nz,j) * z^j * (1-z)^(nz-j);
end

g_rebuilt = sym(0);
for i=0:ny
    for j=0:nz
        g_rebuilt = g_rebuilt + B(i+1,j+1) * By(i+1) * Bz(j+1);
    end
end
g_rebuilt = expand(g_rebuilt);

disp("Sanity check g_yz - g_rebuilt (should be 0):");
disp(simplify(g_yz - g_rebuilt));

% ---- 5) Bernstein min lower bound over [0,1]^2 ----
% min_{(y,z)} g_yz >= min_{i,j} B(i,j)
B_list = reshape(B, 1, []);
disp("Bernstein coeffs as a 1x9 list:");
disp(B_list);

% symbolic min (will stay symbolic/piecewise); for numeric, substitute then double(min(...))
disp("Lower bound: min(B_list):");
disp(min(B_list));

