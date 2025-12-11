clc,clear

%pkg load symbolic % OCTAVE version
%pkg load optim % OCTAVE version

%% --------------------------------------------------------------------
%% 1. Declare symbols
%% --------------------------------------------------------------------
syms x y z t real

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

% Ray endpoints
syms ax ay az bx by bz dx dy dz real

Pa = [ax, ay, az];
Pb = [bx, by, bz];
Pd = [dx, dy, dz];

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
x_t = ax + dx*t;
y_t = ay + dy*t;
z_t = az + dz*t;

% x_t = bx - dx*(1-t);
% y_t = by - dy*(1-t);
% z_t = bz - dz*(1-t);

% x_t = ax*(1-t) + bx*t;
% y_t = ay*(1-t) + by*t;
% z_t = az*(1-t) + bz*t;

f_t = simplify( subs(f_xyz, [x y z], [x_t, y_t, z_t]) );
f_t = collect(f_t, t);

% Extract coefficients 
[f_t_coeffs, f_t_terms] = coeffs(f_t, t);

disp("Mapped expression coefficients and terms");
disp([f_t_coeffs(:), f_t_terms(:)]);


%% --------------------------------------------------------------------
%% Simplification patterns 
%% --------------------------------------------------------------------

% Apply forward mapping
v_xyz = simplify( subs(f_xyz, F, F2V) );
v_t = simplify( subs(f_t, F, F2V) );
v_t = collect(v_t, [t, bx, by, bz, ax, ay, az dx dy dz]);

% Extract coefficients
[v_t_coeffs, v_t_terms] = coeffs(v_t, t);

disp("Mapped expression coefficients and terms (a_coeffs, a_terms):");
disp([v_t_coeffs(:), v_t_terms(:)]);

%% --------------------------------------------------------------------
%% Include ray trilinear interpolation equations in bernstein form
%% --------------------------------------------------------------------

% We assume f(t) is a cubic polynomial in t:
% f(t) = c0 + c1*t + c2*t^2 + c3*t^3

Cf0 = simplify( f_t_coeffs(4) );
Cf1 = simplify( f_t_coeffs(3) );
Cf2 = simplify( f_t_coeffs(2) );  
Cf3 = simplify( f_t_coeffs(1) );  

% Bernstein coefficients for degree-3 polynomial on [0,1]:
Bf0 = simplify(Cf0);
Bf1 = simplify(Cf0 + Cf1*1/3);
Bf2 = simplify(Cf0 + Cf1*2/3 + Cf2*1/3);
Bf3 = simplify(Cf0 + Cf1 + Cf2 + Cf3);

% Collect coefficients
Bf0 = collect(Bf0, F);
Bf1 = collect(Bf1, F);
Bf2 = collect(Bf2, F);
Bf3 = collect(Bf3, F);

% Collect differences
Bf10 = collect(simplify(Bf1 - Bf0), Pd);
Bf20 = collect(simplify(Bf2 - Bf0), Pd);
Bf30 = collect(simplify(Bf3 - Bf0), Pd);

% Bernstein-form polynomial (for verification / export)
Bf_t = simplify( ...
    Bf0 * 1 * t^0 * (1 - t)^3 ...
  + Bf1 * 3 * t^1 * (1 - t)^2 ...
  + Bf2 * 3 * t^2 * (1 - t)^1 ...
  + Bf3 * 1 * t^3 * (1 - t)^0);

disp("Bernstein coefficients");
disp(Bf0);
disp(Bf1);
disp(Bf2);
disp(Bf3);

disp("Sanity check f(t) - Bf_t(t), should be 0:");
disp(simplify(f_t - Bf_t));

%% --------------------------------------------------------------------
%% Convert ray trilinear interpolation symmetric equation into bernstein form
%% --------------------------------------------------------------------

% We assume f(t) is a cubic polynomial in t:
% f(t) = c0 + c1*t + c2*t^2 + c3*t^3

Cv0 = simplify( v_t_coeffs(4) );
Cv1 = simplify( v_t_coeffs(3) );
Cv2 = simplify( v_t_coeffs(2) );  
Cv3 = simplify( v_t_coeffs(1) );  

% Bernstein coefficients for degree-3 polynomial on [0,1]:
Bv0 = simplify(Cv0);
Bv1 = simplify(Cv0 + Cv1*1/3);
Bv2 = simplify(Cv0 + Cv1*2/3 + Cv2*1/3);
Bv3 = simplify(Cv0 + Cv1 + Cv2 + Cv3);

% Collect coefficients
Bv0 = collect(Bv0, V);
Bv1 = collect(Bv1, V);
Bv2 = collect(Bv2, V);
Bv3 = collect(Bv3, V);

% Collect differences
Bv10 = collect(simplify(Bv1 - Bv0), Pd);
Bv20 = collect(simplify(Bv2 - Bv0), Pd);
Bv30 = collect(simplify(Bv3 - Bv0), Pd);

% Bernstein-form polynomial (for verification / export)
Bv_t = simplify( ...
    Bv0 * 1 * t^0 * (1 - t)^3 ...
  + Bv1 * 3 * t^1 * (1 - t)^2 ...
  + Bv2 * 3 * t^2 * (1 - t)^1 ...
  + Bv3 * 1 * t^3 * (1 - t)^0);

disp("Bernstein coefficients");
disp(Bv0);
disp(Bv1);
disp(Bv2);
disp(Bv3);

disp("Sanity check v_t - Bv_t, should be 0:");
disp(simplify(v_t - Bv_t));

%% --------------------------------------------------------------------
%% Compute bernstein maxima of f(t) in t [0, 1], when ax = 0
%% --------------------------------------------------------------------

Bf10_subs = simplify(subs(Bf10, [ax, dx], [0, 1]));
Bf20_subs = simplify(subs(Bf20, [ax, dx], [0, 1]));
Bf30_subs = simplify(subs(Bf30, [ax, dx], [0, 1]));

maxima = [
    simplify(subs(Bf10_subs, [ay, az, dy, dz], [0,0,0,0])), ...
    simplify(subs(Bf10_subs, [ay, az, dy, dz], [0,0,0,1])), ...
    simplify(subs(Bf10_subs, [ay, az, dy, dz], [0,0,1,0])), ...
    simplify(subs(Bf10_subs, [ay, az, dy, dz], [0,0,1,1])), ...
    simplify(subs(Bf10_subs, [ay, az, dy, dz], [0,1,0,0])), ...
    simplify(subs(Bf10_subs, [ay, az, dy, dz], [0,1,0,1])), ...
    simplify(subs(Bf10_subs, [ay, az, dy, dz], [0,1,1,0])), ...
    simplify(subs(Bf10_subs, [ay, az, dy, dz], [0,1,1,1])), ...
    simplify(subs(Bf10_subs, [ay, az, dy, dz], [1,0,0,0])), ...
    simplify(subs(Bf10_subs, [ay, az, dy, dz], [1,0,0,1])), ...
    simplify(subs(Bf10_subs, [ay, az, dy, dz], [1,0,1,0])), ...
    simplify(subs(Bf10_subs, [ay, az, dy, dz], [1,0,1,1])), ...
    simplify(subs(Bf10_subs, [ay, az, dy, dz], [1,1,0,0])), ...
    simplify(subs(Bf10_subs, [ay, az, dy, dz], [1,1,0,1])), ...
    simplify(subs(Bf10_subs, [ay, az, dy, dz], [1,1,1,0])), ...
    simplify(subs(Bf10_subs, [ay, az, dy, dz], [1,1,1,1])), ...
    simplify(subs(Bf20_subs, [ay, az, dy, dz], [0,0,0,0])), ...
    simplify(subs(Bf20_subs, [ay, az, dy, dz], [0,0,0,1])), ...
    simplify(subs(Bf20_subs, [ay, az, dy, dz], [0,0,1,0])), ...
    simplify(subs(Bf20_subs, [ay, az, dy, dz], [0,0,1,1])), ...
    simplify(subs(Bf20_subs, [ay, az, dy, dz], [0,1,0,0])), ...
    simplify(subs(Bf20_subs, [ay, az, dy, dz], [0,1,0,1])), ...
    simplify(subs(Bf20_subs, [ay, az, dy, dz], [0,1,1,0])), ...
    simplify(subs(Bf20_subs, [ay, az, dy, dz], [0,1,1,1])), ...
    simplify(subs(Bf20_subs, [ay, az, dy, dz], [1,0,0,0])), ...
    simplify(subs(Bf20_subs, [ay, az, dy, dz], [1,0,0,1])), ...
    simplify(subs(Bf20_subs, [ay, az, dy, dz], [1,0,1,0])), ...
    simplify(subs(Bf20_subs, [ay, az, dy, dz], [1,0,1,1])), ...
    simplify(subs(Bf20_subs, [ay, az, dy, dz], [1,1,0,0])), ...
    simplify(subs(Bf20_subs, [ay, az, dy, dz], [1,1,0,1])), ...
    simplify(subs(Bf20_subs, [ay, az, dy, dz], [1,1,1,0])), ...
    simplify(subs(Bf20_subs, [ay, az, dy, dz], [1,1,1,1])), ...
    simplify(subs(Bf30_subs, [ay, az, dy, dz], [0,0,0,0])), ...
    simplify(subs(Bf30_subs, [ay, az, dy, dz], [0,0,0,1])), ...
    simplify(subs(Bf30_subs, [ay, az, dy, dz], [0,0,1,0])), ...
    simplify(subs(Bf30_subs, [ay, az, dy, dz], [0,0,1,1])), ...
    simplify(subs(Bf30_subs, [ay, az, dy, dz], [0,1,0,0])), ...
    simplify(subs(Bf30_subs, [ay, az, dy, dz], [0,1,0,1])), ...
    simplify(subs(Bf30_subs, [ay, az, dy, dz], [0,1,1,0])), ...
    simplify(subs(Bf30_subs, [ay, az, dy, dz], [0,1,1,1])), ...
    simplify(subs(Bf30_subs, [ay, az, dy, dz], [1,0,0,0])), ...
    simplify(subs(Bf30_subs, [ay, az, dy, dz], [1,0,0,1])), ...
    simplify(subs(Bf30_subs, [ay, az, dy, dz], [1,0,1,0])), ...
    simplify(subs(Bf30_subs, [ay, az, dy, dz], [1,0,1,1])), ...
    simplify(subs(Bf30_subs, [ay, az, dy, dz], [1,1,0,0])), ...
    simplify(subs(Bf30_subs, [ay, az, dy, dz], [1,1,0,1])), ...
    simplify(subs(Bf30_subs, [ay, az, dy, dz], [1,1,1,0])), ...
    simplify(subs(Bf30_subs, [ay, az, dy, dz], [1,1,1,1])), ...
];

maxima = simplify(maxima);
maxima = unique(maxima);
maxima = maxima(:);

%% --------------------------------------------------------------------
%% Sort maxima by expression complexity
%% --------------------------------------------------------------------

% Assume F = [f000 f001 ... f111] is already defined
n_expr = length(maxima);
complexity = zeros(n_expr, 1);

for i = 1:n_expr
    % complexity = number of nonzero linear coefficients
    complexity(i) = strlength(string(maxima(i)));
end

% Now sort by complexity
[complexity_sorted, order] = sort(complexity, 'descend');
maxima = maxima(order);

fprintf("Sorted expressions (from simplest to most complex):\n");
disp(maxima);

%% --------------------------------------------------------------------
%  Build coefficient matrix A
%% --------------------------------------------------------------------
n_expr = length(maxima);
n_var  = length(F);
A_sym = sym(zeros(n_expr, n_var));

for j = 1:n_var
    % basis vector e_j: F(j) = 1, others = 0
    basisF = zeros(1, n_var);
    basisF(j) = 1;
    A_sym(:, j) = subs(maxima, F, basisF);
end

A = double(A_sym);

fprintf('Coefficient matrix A * F <= 0:\n');
disp(A);

%% --------------------------------------------------------------------
%  Search for subconvex combination maxima to remove
%% --------------------------------------------------------------------
keep = true(n_expr,1);  % assume all are essential initially
tol_eq = 1e-8;   % tolerance for Aeq*lambda ≈ beq
tol_ub = 1e-8;   % tolerance for sum(lambda)<= 1

for k = 1:n_expr

    % Build M_other: all rows except k that are currently kept
    rows = keep(:);
    rows(k) = false;
    M_other = A(rows,:);          % (n_other) x n_var
    m_k = A(k,:).';               % n_var x 1

    if isempty(M_other)
        % No "other" inequalities left -> can't be redundant
        fprintf('Inequality %d appears essential (no others).\n', k);
        continue;
    end

    % We want: M_other' * lambda = m_k, sum(lambda) <= 1, lambda >= 0

    Aeq = M_other.';              % n_var x n_lambda
    beq = m_k;                    % n_var x 1
    n_lambda = size(Aeq,2);

    % Objective doesn't matter (feasibility problem)
    f = ones(n_lambda,1);

    % Inequality: sum(lambda) <= 1
    Aineq = ones(1, n_lambda);    % 1 x n_lambda
    bineq = 1;

    % Bounds: lambda >= 0
    lb = zeros(n_lambda,1);
    ub = []; % no upper bound, other than sum(lambda)<=1

    % linprog: minimize f' * lambda
    lambda = linprog(f, Aineq, bineq, Aeq, beq, lb, ub);

    redundant = false;

    if ~isempty(lambda)
        % Solution found that *already* satisfies:
        %   M_other' * lambda = m_k  (within solver tolerance)
        %   sum(lambda) <= 1
        %   lambda >= 0
        %
        % Optionally, add a safeguard check with your tolerances:
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
%  Reduced system
%% --------------------------------------------------------------------
maxima_reduced = maxima(keep);
A_reduced = A(keep,:);

fprintf('\nEssential maxima (kept):\n');
disp(find(keep).');   % indices of non-redundant maxima

fprintf('\nReduced coefficient matrix A_reduced:\n');
disp(A_reduced);

fprintf('\nReduced symbolic maxima maxima_reduced:\n');
disp(maxima_reduced);
