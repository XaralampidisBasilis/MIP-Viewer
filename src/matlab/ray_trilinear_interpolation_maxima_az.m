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

% v_xyz        = simplify( subs(f_xyz, F, F2V) );
% v_t          = simplify( subs(f_t, F, F2V) );
% v_t_coeffs   = simplify( subs(f_t_coeffs, F, F2V) );
% Dv_xyz       = simplify( subs(Df_xyz, F, F2V) );
% Dv_t         = simplify( subs(Df_t, F, F2V) );
% Dv_t_coeffs  = simplify( subs(Df_t_coeffs, F, F2V) );
% Bv_t_coeffs  = simplify( subs(Bf_t_coeffs, F, F2V) );
% BDv_t_coeffs = simplify( subs(BDf_t_coeffs, F, F2V) );

%% --------------------------------------------------------------------
%% Maxima of trilinear derivative over ray
%% --------------------------------------------------------------------

maxima_terms = simplify(subs(Bf_t_coeffs, [d], [b-a]));

maxima = [
    simplify(subs( maxima_terms(1), [a, b], [[0,0,0], [1,0,0]]) ), ...
    simplify(subs( maxima_terms(1), [a, b], [[0,0,0], [1,1,0]]) ), ...
    simplify(subs( maxima_terms(1), [a, b], [[0,0,0], [1,0,1]]) ), ...
    simplify(subs( maxima_terms(1), [a, b], [[0,0,0], [1,1,1]]) ), ...
    simplify(subs( maxima_terms(1), [a, b], [[0,1,0], [1,1,0]]) ), ...
    simplify(subs( maxima_terms(1), [a, b], [[0,1,0], [1,1,1]]) ), ...

    simplify(subs( maxima_terms(2), [a, b], [[0,0,0], [1,0,0]]) ), ...
    simplify(subs( maxima_terms(2), [a, b], [[0,0,0], [1,1,0]]) ), ...
    simplify(subs( maxima_terms(2), [a, b], [[0,0,0], [1,0,1]]) ), ...
    simplify(subs( maxima_terms(2), [a, b], [[0,0,0], [1,1,1]]) ), ...
    simplify(subs( maxima_terms(2), [a, b], [[0,1,0], [1,1,0]]) ), ...
    simplify(subs( maxima_terms(2), [a, b], [[0,1,0], [1,1,1]]) ), ...

    simplify(subs( maxima_terms(3), [a, b], [[0,0,0], [1,0,0]]) ), ...
    simplify(subs( maxima_terms(3), [a, b], [[0,0,0], [1,1,0]]) ), ...
    simplify(subs( maxima_terms(3), [a, b], [[0,0,0], [1,0,1]]) ), ...
    simplify(subs( maxima_terms(3), [a, b], [[0,0,0], [1,1,1]]) ), ...
    simplify(subs( maxima_terms(3), [a, b], [[0,1,0], [1,1,0]]) ), ...
    simplify(subs( maxima_terms(3), [a, b], [[0,1,0], [1,1,1]]) ), ...

    simplify(subs( maxima_terms(4), [a, b], [[0,0,0], [1,0,0]]) ), ...
    simplify(subs( maxima_terms(4), [a, b], [[0,0,0], [1,1,0]]) ), ...
    simplify(subs( maxima_terms(4), [a, b], [[0,0,0], [1,0,1]]) ), ...
    simplify(subs( maxima_terms(4), [a, b], [[0,0,0], [1,1,1]]) ), ...
    simplify(subs( maxima_terms(4), [a, b], [[0,1,0], [1,1,0]]) ), ...
    simplify(subs( maxima_terms(4), [a, b], [[0,1,0], [1,1,1]]) ), ...
];

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
    complexity(i) = length(char(maxima(i)));
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
    basis = zeros(1, n_var);
    basis(j) = 1;
    A_sym(:, j) = subs(maxima, F, basis);
end

A = double(A_sym);

fprintf('Coefficient matrix A * F <= 0:\n');
disp(A);

%% --------------------------------------------------------------------
%  Search for subconvex combination maxima to remove
%% --------------------------------------------------------------------
keep = true(n_expr,1);  % assume all are essential initially
tol  = 1e-8;            % tolerance

for k = 1:n_expr

    % Build M_other: all rows except k that are currently kept
    rows = keep(:);
    rows(k) = false;

    M_other = A(rows,:);          % (n_other) x n_var
    m_k     = A(k,:).';           % n_var x 1

    if isempty(M_other)
        % No "other" inequalities left -> can't be redundant
        fprintf('Inequality %d appears essential (no others).\n', k);
        continue;
    end

    % We want: M_other' * lambda = m_k, sum(lambda) = 1, lambda >= 0

    Aeq_base = M_other.';         % n_var x n_lambda
    beq_base = m_k;               % n_var x 1
    n_lambda = size(Aeq_base, 2);

    % Add equality: sum(lambda) = 1
    Aeq = [Aeq_base; ones(1, n_lambda)];   % (n_var+1) x n_lambda
    beq = [beq_base; 1];                   % (n_var+1) x 1

    % Objective doesn't matter much (feasibility-ish); keep something simple
    f = ones(n_lambda,1);

    % Bounds: lambda >= 0
    lb = zeros(n_lambda,1);
    ub = []; % no upper bounds

    % linprog: minimize f' * lambda subject to equalities + bounds
    % (No inequality constraints now)
    lambda = linprog(f, [], [], Aeq, beq, lb, ub);

    redundant = false;

    if ~isempty(lambda)
        % Safeguard check with tolerances:
        req = norm(Aeq_base*lambda - beq_base, Inf);
        sum_lambda = sum(lambda);

        if req <= tol && abs(sum_lambda - 1) <= tol
            redundant = true;
        end
    end

    if redundant
        keep(k) = false;
        fprintf('Inequality %d is redundant. sum(lambda) = %.6g\n', k, sum_lambda);
    else
        if ~isempty(lambda)
            fprintf('Inequality %d appears essential.\n', k);
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
