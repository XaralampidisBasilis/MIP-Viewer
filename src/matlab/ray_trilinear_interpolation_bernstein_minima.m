clc,clear

pkg load symbolic % OCTAVE version
pkg load optim % OCTAVE version

%% --------------------------------------------------------------------
%% Include ray trilinear interpolation equations in bernstein form
%% --------------------------------------------------------------------
run('ray_trilinear_interpolation_bernstein.m')

%% --------------------------------------------------------------------
%% Compute bernstein maxima of f(t) in t [0, 1], when ax = 0
%% --------------------------------------------------------------------

Bf01 = simplify(Bf1 - Bf0);
Bf02 = simplify(Bf2 - Bf0);
Bf03 = simplify(Bf3 - Bf0);

maxima = [
    simplify(subs(Bf01, [ay, ax, az, dx, dy, dz], [0,0,0,1,0,0])), ...
    simplify(subs(Bf01, [ay, ax, az, dx, dy, dz], [0,0,0,1,0,1])), ...
    simplify(subs(Bf01, [ay, ax, az, dx, dy, dz], [0,0,0,1,1,0])), ...
    simplify(subs(Bf01, [ay, ax, az, dx, dy, dz], [0,0,0,1,1,1])), ...
    simplify(subs(Bf01, [ay, ax, az, dx, dy, dz], [0,0,1,1,0,0])), ...
    simplify(subs(Bf01, [ay, ax, az, dx, dy, dz], [0,0,1,1,0,1])), ...
    simplify(subs(Bf01, [ay, ax, az, dx, dy, dz], [0,0,1,1,1,0])), ...
    simplify(subs(Bf01, [ay, ax, az, dx, dy, dz], [0,0,1,1,1,1])), ...
    simplify(subs(Bf01, [ay, ax, az, dx, dy, dz], [0,1,0,1,0,0])), ...
    simplify(subs(Bf01, [ay, ax, az, dx, dy, dz], [0,1,0,1,0,1])), ...
    simplify(subs(Bf01, [ay, ax, az, dx, dy, dz], [0,1,0,1,1,0])), ...
    simplify(subs(Bf01, [ay, ax, az, dx, dy, dz], [0,1,0,1,1,1])), ...
    simplify(subs(Bf01, [ay, ax, az, dx, dy, dz], [0,1,1,1,0,0])), ...
    simplify(subs(Bf01, [ay, ax, az, dx, dy, dz], [0,1,1,1,0,1])), ...
    simplify(subs(Bf01, [ay, ax, az, dx, dy, dz], [0,1,1,1,1,0])), ...
    simplify(subs(Bf01, [ay, ax, az, dx, dy, dz], [0,1,1,1,1,1])), ...
    simplify(subs(Bf02, [ay, ax, az, dx, dy, dz], [0,0,0,1,0,0])), ...
    simplify(subs(Bf02, [ay, ax, az, dx, dy, dz], [0,0,0,1,0,1])), ...
    simplify(subs(Bf02, [ay, ax, az, dx, dy, dz], [0,0,0,1,1,0])), ...
    simplify(subs(Bf02, [ay, ax, az, dx, dy, dz], [0,0,0,1,1,1])), ...
    simplify(subs(Bf02, [ay, ax, az, dx, dy, dz], [0,0,1,1,0,0])), ...
    simplify(subs(Bf02, [ay, ax, az, dx, dy, dz], [0,0,1,1,0,1])), ...
    simplify(subs(Bf02, [ay, ax, az, dx, dy, dz], [0,0,1,1,1,0])), ...
    simplify(subs(Bf02, [ay, ax, az, dx, dy, dz], [0,0,1,1,1,1])), ...
    simplify(subs(Bf02, [ay, ax, az, dx, dy, dz], [0,1,0,1,0,0])), ...
    simplify(subs(Bf02, [ay, ax, az, dx, dy, dz], [0,1,0,1,0,1])), ...
    simplify(subs(Bf02, [ay, ax, az, dx, dy, dz], [0,1,0,1,1,0])), ...
    simplify(subs(Bf02, [ay, ax, az, dx, dy, dz], [0,1,0,1,1,1])), ...
    simplify(subs(Bf02, [ay, ax, az, dx, dy, dz], [0,1,1,1,0,0])), ...
    simplify(subs(Bf02, [ay, ax, az, dx, dy, dz], [0,1,1,1,0,1])), ...
    simplify(subs(Bf02, [ay, ax, az, dx, dy, dz], [0,1,1,1,1,0])), ...
    simplify(subs(Bf02, [ay, ax, az, dx, dy, dz], [0,1,1,1,1,1])), ...
    simplify(subs(Bf03, [ay, ax, az, dx, dy, dz], [0,0,0,1,0,0])), ...
    simplify(subs(Bf03, [ay, ax, az, dx, dy, dz], [0,0,0,1,0,1])), ...
    simplify(subs(Bf03, [ay, ax, az, dx, dy, dz], [0,0,0,1,1,0])), ...
    simplify(subs(Bf03, [ay, ax, az, dx, dy, dz], [0,0,0,1,1,1])), ...
    simplify(subs(Bf03, [ay, ax, az, dx, dy, dz], [0,0,1,1,0,0])), ...
    simplify(subs(Bf03, [ay, ax, az, dx, dy, dz], [0,0,1,1,0,1])), ...
    simplify(subs(Bf03, [ay, ax, az, dx, dy, dz], [0,0,1,1,1,0])), ...
    simplify(subs(Bf03, [ay, ax, az, dx, dy, dz], [0,0,1,1,1,1])), ...
    simplify(subs(Bf03, [ay, ax, az, dx, dy, dz], [0,1,0,1,0,0])), ...
    simplify(subs(Bf03, [ay, ax, az, dx, dy, dz], [0,1,0,1,0,1])), ...
    simplify(subs(Bf03, [ay, ax, az, dx, dy, dz], [0,1,0,1,1,0])), ...
    simplify(subs(Bf03, [ay, ax, az, dx, dy, dz], [0,1,0,1,1,1])), ...
    simplify(subs(Bf03, [ay, ax, az, dx, dy, dz], [0,1,1,1,0,0])), ...
    simplify(subs(Bf03, [ay, ax, az, dx, dy, dz], [0,1,1,1,0,1])), ...
    simplify(subs(Bf03, [ay, ax, az, dx, dy, dz], [0,1,1,1,1,0])), ...
    simplify(subs(Bf03, [ay, ax, az, dx, dy, dz], [0,1,1,1,1,1])), ...
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
    % Extract coefficients of this expression w.r.t. F
    [c, terms] = coeffs(maxima(i), F);

    % complexity = number of nonzero linear coefficients
    complexity(i) = length(c);
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
%  10. Detect redundant maxima via conic combination test
%
%  Inequality k: A(k,:) * F <= 0  is redundant if
%       A(k,:) = sum_{i!=k} lambda_i * A(i,:),
%       with lambda_i >= 0.
%
%  We check this using linprog for each row k.
%  'Algorithm','dual-simplex' or 'interior-point' needed
%% --------------------------------------------------------------------
keep = true(n_expr,1);  % assume all are essential initially
tol_eq = 1e-8;   % tolerance for Aeq*lambda ≈ beq
tol_ub = 1e-8;  % tolerance for sum(lambda)<= 1

%options = optimoptions('linprog', 'Algorithm', 'dual-simplex', 'Display', 'none');  % MATLAB version

for k = 1:n_expr
    % Build M_other: all rows except k
    rows = keep(:);
    rows(k) = false;
    M_other = A(rows,:);          % (n_expr-1) x n_var
    m_k     = A(k,:).';           % column (n_var x 1)

    % We want: M_other' * lambda = m_k sum(lambda) <= 1
    Aeq = M_other.';              % n_var x (n_expr-1)
    beq = m_k;                    % n_var x 1
    n_lambda = size(Aeq,2);
    f = zeros(n_lambda,1);        % objective doesn't matter (feasibility problem)

    % linprog signature:
    lambda = linprog(f, [], [], Aeq, beq); % OCTAVE version

    redundant = false;

    if ~isempty(lambda) && all(isfinite(lambda))
        % Check equality Aeq*lambda ≈ beq
        req = norm(Aeq*lambda - beq, Inf);
        % Check sum(lambda) <= 1 (up to numerical noise)
        sum_lambda = sum(lambda);

        if req <= tol_eq && sum_lambda <= 1 + tol_ub
            redundant = true;
        end
    end

    if redundant
        keep(k) = false;
        fprintf('Inequality %d is redundant.\n', k);
    else
        fprintf('Inequality %d appears essential.\n', k);
    end
end


%% --------------------------------------------------------------------
%  11. Reduced system
%% --------------------------------------------------------------------
maxima_reduced = maxima(keep);
A_reduced = A(keep,:);

fprintf('\nEssential maxima (kept):\n');
disp(find(keep).');   % indices of non-redundant maxima

fprintf('\nReduced coefficient matrix A_reduced:\n');
disp(A_reduced);

fprintf('\nReduced symbolic maxima maxima_reduced <= 0:\n');
disp(maxima_reduced);
