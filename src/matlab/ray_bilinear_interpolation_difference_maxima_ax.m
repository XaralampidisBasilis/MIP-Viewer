clc,clear

%pkg load symbolic % OCTAVE version
%pkg load optim % OCTAVE version

%% --------------------------------------------------------------------
%% Declare symbols
%% --------------------------------------------------------------------

% corner values
% symmetric linear combinations of corner values
syms f00 f10 f01 f11 real
syms v00 v10 v01 v11 real

F = [f00 f10 f01 f11];
V = [v00 v10 v01 v11];

F2V = [
    v00, ...
    v10 + v00, ...
    v01 + v00, ...
    v11 + v10 + v01 + v00, ...
];

V2F = [
    f00, ...
    f10 - f00, ...
    f01 - f00, ...
    f00 - f10 - f01 + f11, ...
];

% Variables
syms x y t real

r = [x, y];

% Ray endpoints
syms ax ay bx by dx dy sx sy real

a = [ax, ay];
b = [bx, by];
d = [dx, dy];
s = [sx, sy];

%% --------------------------------------------------------------------
%% Trilinear interpolation f(x,y,z)
%% --------------------------------------------------------------------

f_xy = f00 * (1-x) * (1-y) ...
     + f10 *    x  * (1-y) ...
     + f01 * (1-x) *    y  ...
     + f11 *    x  *    y;

%% --------------------------------------------------------------------
%% Ray substitution r(t) 
%% --------------------------------------------------------------------

x_t = ax + dx*t;
y_t = ay + dy*t;

% x_t = bx - dx*(1-t);
% y_t = by - dy*(1-t);

% x_t = ax*(1-t) + bx*t;
% y_t = ay*(1-t) + by*t;

r_t = [x_t, y_t];

%% --------------------------------------------------------------------
%% Trilinear function over ray  
%% --------------------------------------------------------------------

f_t = simplify( subs(f_xy, r, r_t) );
f_t = collect(f_t, t);

% Extract coefficients
[f_t_coeffs, f_t_terms] = coeffs(f_t, t);

%% --------------------------------------------------------------------
%% Bernstein Trilinear function over ray 
%% --------------------------------------------------------------------

% Bernstein coefficients
Bf_t_coeffs = [
    simplify(f_t_coeffs(3)), ...
    simplify(f_t_coeffs(3) + f_t_coeffs(2)*1/2), ...
    simplify(f_t_coeffs(3) + f_t_coeffs(2) + f_t_coeffs(1)), ...
];

% Bernstein terms
Bf_t_terms = [ ...
    1 * t^0 * (1 - t)^2, ...
    2 * t^1 * (1 - t)^1, ...
    1 * t^2 * (1 - t)^0, ...
];

% Bernstein trilinear function over ray
Bf_t = dot(Bf_t_coeffs, Bf_t_terms);

disp("Sanity check f(t) - Bf_t(t), should be 0:");
disp(simplify(f_t - Bf_t));

%% --------------------------------------------------------------------
%% Trilinear function derivative over ray  
%% --------------------------------------------------------------------

% Compute partial derivatives
fx = simplify(diff(f_xy, x));
fy = simplify(diff(f_xy, y));

% Gradient vector
Gf_xy = [fx, fy];

% directional derivative
Df_xy = dot(Gf_xy, d);

% directional derivative over ray
Gf_t = simplify( subs(Gf_xy, r, r_t) );
Df_t = simplify( subs(Df_xy, r, r_t) );

% coefficients
[Df_t_coeffs, Df_t_terms] = coeffs(Df_t, t);

%% --------------------------------------------------------------------
%% Bernstein trilinear function derivative over ray  
%% --------------------------------------------------------------------

% Bernstein coefficients
BDf_t_coeffs = [
    simplify(Df_t_coeffs(2)), ...
    simplify(Df_t_coeffs(2) + Df_t_coeffs(1)), ...
];

% Bernstein terms
BDf_t_terms = [
    1 * t^0 * (1 - t)^1, ...
    1 * t^1 * (1 - t)^0, ...
];

% Bernstein trilinear derivative function over ray
BDf_t = dot(BDf_t_coeffs, BDf_t_terms);

disp("Sanity check Df(t) - BDf(t), should be 0:");
disp(simplify(Df_t - BDf_t));

%% --------------------------------------------------------------------
%% Simplification patterns
%% --------------------------------------------------------------------

v_xy         = simplify( subs(f_xy, F, F2V) );
v_t          = simplify( subs(f_t, F, F2V) );
v_t_coeffs   = simplify( subs(f_t_coeffs, F, F2V) );
Dv_xy        = simplify( subs(Df_xy, F, F2V) );
Dv_t         = simplify( subs(Df_t, F, F2V) );
Dv_t_coeffs  = simplify( subs(Df_t_coeffs, F, F2V) );
Bv_t_coeffs  = simplify( subs(Bf_t_coeffs, F, F2V) );
BDv_t_coeffs = simplify( subs(BDf_t_coeffs, F, F2V) );

%% --------------------------------------------------------------------
%% Maxima of trilinear derivative over ray
%% --------------------------------------------------------------------

%% Subspace of ax = 0 and dy/dx <= 1

maxima_terms = [
    simplify(Bf_t_coeffs(2) - Bf_t_coeffs(1)), ...
    simplify(Bf_t_coeffs(3) - Bf_t_coeffs(1)), ...
];

maxima_terms = simplify(subs(maxima_terms, [d], [b-a]));

maxima = [
    simplify(subs( maxima_terms(1), [a, b], [[0,0], [1,0]]) ), ...
    simplify(subs( maxima_terms(1), [a, b], [[0,0], [1,1]]) ), ...
    simplify(subs( maxima_terms(1), [a, b], [[0,1], [1,1]]) ), ...

    simplify(subs( maxima_terms(2), [a, b], [[0,0], [1,0]]) ), ...
    simplify(subs( maxima_terms(2), [a, b], [[0,0], [1,1]]) ), ...
    simplify(subs( maxima_terms(2), [a, b], [[0,1], [1,1]]) ), ...
];

maxima = unique(maxima);
maxima = maxima(:);

%% --------------------------------------------------------------------
%% Sort maxima by expression complexity
%% --------------------------------------------------------------------

% Assume F = [f00 f10 f01 f11] is already defined
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
tol = 1e-8;   % tolerance

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

        if req <= tol && sum_lambda <= 1 + tol
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


%% --------------------------------------------------------------------
%  Results
%% --------------------------------------------------------------------
% v = max(
%     -f00 + f01/2 + f10/2,
%     -f00 + f10,
%     -f00 + f11,
%     -f01 + f11,
% );