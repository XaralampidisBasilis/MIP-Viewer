clc, clear

pkg load symbolic % OCTAVE version
pkg load optim    % OCTAVE version (not used here)

%% --------------------------------------------------------------------
%% 1. Declare symbols (2D)
%% --------------------------------------------------------------------
syms x y t real

% corner values (bilinear: 4 corners)
syms f00 f10 f01 f11 real
syms v00 v10 v01 v11 real

F = [f00 f10 f01 f11];
V = [v00 v10 v01 v11];

% Symmetric (cumulative / inclusion–exclusion) combinations in 2D:
% v00 = f00
% v10 = f10 - f00
% v01 = f01 - f00
% v11 = f00 - f10 - f01 + f11   (the "mixed" term)
F2V = [ ...
    v00, ...
    v10 + v00, ...
    v01 + v00, ...
    v11 + v10 + v01 + v00 ...
];

V2F = [ ...
    f00, ...
    f10 - f00, ...
    f01 - f00, ...
    f00 - f10 - f01 + f11 ...
];

% Ray endpoints / direction in 2D
syms ax ay bx by dx dy sx sy real
Pa = [ax, ay];
Pb = [bx, by];
Pd = [dx, dy];

%% --------------------------------------------------------------------
%% 2. Bilinear interpolation f(x,y)
%% --------------------------------------------------------------------
f_xy = f00 * (1-x) * (1-y) ...
     + f10 *    x  * (1-y) ...
     + f01 * (1-x) *    y  ...
     + f11 *    x  *    y;

%% --------------------------------------------------------------------
%% 3. Ray substitution r(t) = a + d*t  (equivalently a(1-t)+b*t with d=b-a)
%% --------------------------------------------------------------------
% x_t = ax * (1-t) + bx * t;
% y_t = ay * (1-t) + by * t;

% x_t = ax + dx*t;
% y_t = ay + dy*t;

x_t = bx - dx*(1-t);
y_t = by - dy*(1-t);

f_t = simplify( subs(f_xy, [x y], [x_t, y_t]) );
f_t = collect(f_t, t);

% Extract coefficients (f(t) is quadratic in t for bilinear-on-a-line)
[f_t_coeffs, f_t_terms] = coeffs(f_t, t);

disp("Mapped expression coefficients and terms");
disp([f_t_coeffs(:), f_t_terms(:)]);

%% --------------------------------------------------------------------
%% 4. Apply symmetric mapping (F -> V) and repeat
%% --------------------------------------------------------------------
v_xy = simplify( subs(f_xy, F, F2V) );
v_t  = simplify( subs(f_t,  F, F2V) );

v_t = collect(v_t, [t, ax, ay, bx, by, dx, dy]);

% Extract coefficients
[v_t_coeffs, v_t_terms] = coeffs(v_t, t);

disp("Mapped expression coefficients and terms (v_coeffs, v_terms):");
disp([v_t_coeffs(:), v_t_terms(:)]);

%% --------------------------------------------------------------------
%% 5. Convert ray bilinear polynomial to Bernstein form (degree 2)
%%     f(t) = c0 + c1*t + c2*t^2
%% --------------------------------------------------------------------
% coeffs() may return in different order depending on toolbox;
% use explicit extraction via coeffs list assuming terms include t^0,t^1,t^2
c0 = simplify( subs(f_t, t, 0) );
c1 = simplify( subs(diff(f_t, t), t, 0) );
c2 = simplify( 1/2 * subs(diff(f_t, t, 2), t, 0) );

% Bernstein coefficients for degree-2 polynomial on [0,1]:
% B0 = c0
% B1 = c0 + c1/2
% B2 = c0 + c1 + c2
Bf0 = simplify(c0);
Bf1 = simplify(c0 + c1/2);
Bf2 = simplify(c0 + c1 + c2);

Bf0 = collect(Bf0, F);
Bf1 = collect(Bf1, F);
Bf2 = collect(Bf2, F);

% Differences (often useful)
Bf10 = collect(simplify(Bf1 - Bf0), Pd);
Bf20 = collect(simplify(Bf2 - Bf0), Pd);

% Bernstein-form polynomial for verification
Bf_t = simplify( ...
    Bf0 * (1 - t)^2 ...
  + Bf1 * 2 * t * (1 - t) ...
  + Bf2 * t^2 );

disp("Bernstein coefficients (F-form)");
disp(Bf0);
disp(Bf1);
disp(Bf2);

disp("Sanity check f_t - Bf_t, should be 0:");
disp(simplify(f_t - Bf_t));

%% --------------------------------------------------------------------
%% 6. Same Bernstein conversion for symmetric (V) form
%% --------------------------------------------------------------------
cv0 = simplify( subs(v_t, t, 0) );
cv1 = simplify( subs(diff(v_t, t), t, 0) );
cv2 = simplify( 1/2 * subs(diff(v_t, t, 2), t, 0) );

Bv0 = simplify(cv0);
Bv1 = simplify(cv0 + cv1/2);
Bv2 = simplify(cv0 + cv1 + cv2);

Bv0 = collect(Bv0, V);
Bv1 = collect(Bv1, V);
Bv2 = collect(Bv2, V);

Bv10 = collect(simplify(Bv1 - Bv0), Pd);
Bv20 = collect(simplify(Bv2 - Bv0), Pd);

Bv_t = simplify( ...
    Bv0 * (1 - t)^2 ...
  + Bv1 * 2 * t * (1 - t) ...
  + Bv2 * t^2 );

disp("Bernstein coefficients (V-form)");
disp(Bv0);
disp(Bv1);
disp(Bv2);

disp("Sanity check v_t - Bv_t, should be 0:");
disp(simplify(v_t - Bv_t));
