clc, clear

pkg load symbolic   % OCTAVE
pkg load optim      % OCTAVE (for linprog)

%% --------------------------------------------------------------------
%% 1. Declare symbols (2D)
%% --------------------------------------------------------------------
syms x y t real

% corner values (bilinear)
syms f00 f10 f01 f11 real
syms v00 v10 v01 v11 real

F = [f00 f10 f01 f11];
V = [v00 v10 v01 v11];

% F <-> V mappings (2D inclusion-exclusion / multilinear coefficients)
% v00 = f00
% v10 = f10 - f00d
% v01 = f01 - f00
% v11 = f11 - f10 - f01 + f00
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
    f11 - f10 - f01 + f00 ...
];

% Ray endpoints / direction
syms ax ay bx by dx dy real
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
%% 3. Ray substitution r(t) = a + d*t, t in [0,1]
%% --------------------------------------------------------------------
x_t = ax + dx*t;
y_t = ay + dy*t;

f_t = simplify( subs(f_xy, [x y], [x_t, y_t]) );
f_t = collect(f_t, t);

% Extract coefficients (quadratic in t)
[f_t_coeffs, f_t_terms] = coeffs(f_t, t);

disp("Mapped expression coefficients and terms");
disp([f_t_coeffs(:), f_t_terms(:)]);

%% --------------------------------------------------------------------
%% 4. Simplification patterns (apply F -> V)
%% --------------------------------------------------------------------
v_xy = simplify( subs(f_xy, F, F2V) );
v_t  = simplify( subs(f_t,  F, F2V) );
v_t  = collect(v_t, [t, bx, by, ax, ay, dx, dy]);

% Extract coefficients
[v_t_coeffs, v_t_terms] = coeffs(v_t, t);

disp("Mapped expression coefficients and terms (v_coeffs, v_terms):");
disp([v_t_coeffs(:), v_t_terms(:)]);

%% --------------------------------------------------------------------
%% 5. Bernstein form for ray bilinear (degree 2)
%%    f(t) = c0 + c1*t + c2*t^2
%% --------------------------------------------------------------------
% coeffs() order can vary; use derivatives for robust power coefficients
c0 = simplify( subs(f_t, t, 0) );
c1 = simplify( subs(diff(f_t, t), t, 0) );
c2 = simplify( 1/2 * subs(diff(f_t, t, 2), t, 0) );

% Bernstein coefficients for degree-2 on [0,1]
Bf0 = simplify(c0);
Bf1 = simplify(c0 + c1/2);
Bf2 = simplify(c0 + c1 + c2);

Bf0 = collect(Bf0, F);
Bf1 = collect(Bf1, F);
Bf2 = collect(Bf2, F);

% Differences
Bf10 = collect(simplify(Bf1 - Bf0), Pd);
Bf20 = collect(simplify(Bf2 - Bf0), Pd);

% Verify Bernstein reconstruction
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
%% 6. Bernstein form for symmetric (V) ray polynomial (degree 2)
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

%% --------------------------------------------------------------------
%% 7. Partial derivatives + directional derivative (2D)
%% --------------------------------------------------------------------
dfx = simplify(diff(f_xy, x));
dfy = simplify(diff(f_xy, y));

Df_xy = dfx*dx + dfy*dy;

% Your restriction: 0 <= dy/dx <= 1, and assume dx > 0.
% So set dx = 1 and dy in {0,1} for extreme slopes.
% Df_xy is affine in (x,y) and also affine in dy, so maxima over:
%   x,y in {0,1} and dy in {0,1} cover all cases.
maxima = [
    simplify(subs(Df_xy, [x,y,dx,dy], [0,0,1,0])), ...
    simplify(subs(Df_xy, [x,y,dx,dy], [0,0,1,1])), ...
    simplify(subs(Df_xy, [x,y,dx,dy], [0,1,1,0])), ...
    simplify(subs(Df_xy, [x,y,dx,dy], [0,1,1,1])), ...
    simplify(subs(Df_xy, [x,y,dx,dy], [1,0,1,0])), ...
    simplify(subs(Df_xy, [x,y,dx,dy], [1,0,1,1])), ...
    simplify(subs(Df_xy, [x,y,dx,dy], [1,1,1,0])), ...
    simplify(subs(Df_xy, [x,y,dx,dy], [1,1,1,1]))  ...
];

maxima = unique(maxima);
maxima = maxima(:);

%% --------------------------------------------------------------------
%% 8. Sort maxima by expression complexity
%% --------------------------------------------------------------------
n_expr = length(maxima);
complexity = zeros(n_expr, 1);

for i = 1:n_expr
    complexity(i) = length(char(maxima(i)));
end

[complexity_sorted, order] = sort(complexity, 'descend');
maxima = maxima(order);

fprintf("Sorted expressions (from simplest to most complex):\n");
disp(maxima);

%% --------------------------------------------------------------------
%% 9. Build coefficient matrix A for inequalities A*F <= 0
%% --------------------------------------------------------------------
n_expr = length(maxima);
n_var  = length(F);
A_sym  = sym(zeros(n_expr, n_var));

for j = 1:n_var
    basisF = zeros(1, n_var);
    basisF(j) = 1;
    A_sym(:, j) = subs(maxima, F, basisF);
end

A = double(A_sym);

fprintf('Coefficient matrix A * F <= 0:\n');
disp(A);

%% --------------------------------------------------------------------
%% 10. Search for redundant inequalities via subconvex combination
%% --------------------------------------------------------------------
keep = true(n_expr,1);
tol_eq = 1e-8;
tol_ub = 1e-8;

for k = 1:n_expr

    rows = keep(:);
    rows(k) = false;
    M_other = A(rows,:);
    m_k = A(k,:).';

    if isempty(M_other)
        fprintf('Inequality %d appears essential (no others).\n', k);
        continue;
    end

    % Want: M_other' * lambda = m_k, sum(lambda) <= 1, lambda >= 0
    Aeq = M_other.';
    beq = m_k;
    n_lambda = size(Aeq,2);

    fobj = ones(n_lambda,1);

    Aineq = ones(1, n_lambda);
    bineq = 1;

    lb = zeros(n_lambda,1);
    ub = [];

    lambda = linprog(fobj, Aineq, bineq, Aeq, beq, lb, ub);

    redundant = false;
    if ~isempty(lambda)
        req = norm(Aeq*lambda - beq, Inf);
        sum_lambda = sum(lambda);
        if req <= tol_eq && sum_lambda <= 1 + tol_ub
            redundant = true;
        end
    end

    if redundant
        keep(k) = false;
        fprintf('Inequality %d is redundant.\n', k);
    else
        if ~isempty(lambda)
            fprintf('Inequality %d appears essential. sum(lambda) = %.6g\n', k, sum(lambda));
        else
            fprintf('Inequality %d appears essential (no feasible lambda).\n', k);
        end
    end
end

%% --------------------------------------------------------------------
%% 11. Reduced system
%% --------------------------------------------------------------------
maxima_reduced = maxima(keep);
A_reduced = A(keep,:);

fprintf('\nEssential maxima (kept):\n');
disp(find(keep).');

fprintf('\nReduced coefficient matrix A_reduced:\n');
disp(A_reduced);

fprintf('\nReduced symbolic maxima maxima_reduced:\n');
disp(maxima_reduced);
