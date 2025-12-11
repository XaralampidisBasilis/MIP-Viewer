clc,clear

pkg load symbolic % OCTAVE version
pkg load optim % OCTAVE version

%% --------------------------------------------------------------------
%% Include ray trilinear interpolation equations in bernstein form
%% --------------------------------------------------------------------
run('ray_trilinear_interpolation.m')

%% --------------------------------------------------------------------
%% Partial derivatives
%% --------------------------------------------------------------------

Gfx = simplify(diff(f_xyz, x));
Gfy = simplify(diff(f_xyz, y));
Gfz = simplify(diff(f_xyz, z));
Gf_xyz = [Gfx, Gfy, Gfz];


% directional derivative
Df_xyz = dot(Gf_xyz, D);

% Since we want directional derivative Df_xyz <= 0
% we set a strong condition of Gfx <= 0 && Gfy <= 0 && Gfz <= 0
% we set dx = 1

inequalities = [
    simplify(subs(Gfx, [x, y, z], [0,0,0])), ...
    simplify(subs(Gfx, [x, y, z], [1,0,0])), ...
    simplify(subs(Gfx, [x, y, z], [0,1,0])), ...
    simplify(subs(Gfx, [x, y, z], [1,1,0])), ...
    simplify(subs(Gfx, [x, y, z], [0,0,1])), ...
    simplify(subs(Gfx, [x, y, z], [1,0,1])), ...
    simplify(subs(Gfx, [x, y, z], [0,1,1])), ...
    simplify(subs(Gfx, [x, y, z], [1,1,1])), ...
    simplify(subs(Gfy, [x, y, z], [0,0,0])), ...
    simplify(subs(Gfy, [x, y, z], [1,0,0])), ...
    simplify(subs(Gfy, [x, y, z], [0,1,0])), ...
    simplify(subs(Gfy, [x, y, z], [1,1,0])), ...
    simplify(subs(Gfy, [x, y, z], [0,0,1])), ...
    simplify(subs(Gfy, [x, y, z], [1,0,1])), ...
    simplify(subs(Gfy, [x, y, z], [0,1,1])), ...
    simplify(subs(Gfy, [x, y, z], [1,1,1])), ...
    simplify(subs(Gfz, [x, y, z], [0,0,0])), ...
    simplify(subs(Gfz, [x, y, z], [1,0,0])), ...
    simplify(subs(Gfz, [x, y, z], [0,1,0])), ...
    simplify(subs(Gfz, [x, y, z], [1,1,0])), ...
    simplify(subs(Gfz, [x, y, z], [0,0,1])), ...
    simplify(subs(Gfz, [x, y, z], [1,0,1])), ...
    simplify(subs(Gfz, [x, y, z], [0,1,1])), ...
    simplify(subs(Gfz, [x, y, z], [1,1,1])), ...
];

inequalities = unique(inequalities);
inequalities = inequalities(:);

%% --------------------------------------------------------------------
%% Sort inequalities by expression complexity
%% --------------------------------------------------------------------

% Assume F = [f000 f001 ... f111] is already defined
n_expr = length(inequalities);
complexity = zeros(n_expr, 1);

for i = 1:n_expr
    % Extract coefficients of this expression w.r.t. F
    [c, terms] = coeffs(inequalities(i), F);

    % complexity = number of nonzero linear coefficients
    complexity(i) = length(c);
end

% Now sort by complexity
[complexity_sorted, order] = sort(complexity, 'descend');
inequalities = inequalities(order);

fprintf("Sorted expressions (from simplest to most complex):\n");
disp(inequalities);

%% --------------------------------------------------------------------
%  Build coefficient matrix A
%% --------------------------------------------------------------------
n_expr = length(inequalities);
n_var  = length(F);
A_sym = sym(zeros(n_expr, n_var));

for j = 1:n_var
    % basis vector e_j: F(j) = 1, others = 0
    basisF = zeros(1, n_var);
    basisF(j) = 1;
    A_sym(:, j) = subs(inequalities, F, basisF);
end

A = double(A_sym);

fprintf('Coefficient matrix A * F <= 0:\n');
disp(A);

%% --------------------------------------------------------------------
%  10. Detect redundant inequalities via conic combination test
%
%  Inequality k: A(k,:) * F <= 0  is redundant if
%       A(k,:) = sum_{i!=k} lambda_i * A(i,:),
%       with lambda_i >= 0.
%
%  We check this using linprog for each row k.
%  'Algorithm','dual-simplex' or 'interior-point' needed
%% --------------------------------------------------------------------
keep = true(n_ineq,1);  % assume all are essential initially
tol_eq = 1e-8;   % tolerance for Aeq*lambda ≈ beq
tol_lb = 1e-10;  % tolerance for lambda ≥ 0

%options = optimoptions('linprog', 'Algorithm', 'dual-simplex', 'Display', 'none');  % MATLAB version

for k = 1:n_ineq
    % Build M_other: all rows except k
    rows = keep(:);
    rows(k) = false;
    M_other = A(rows,:);          % (n_ineq-1) x n_var
    m_k     = A(k,:).';           % column (n_var x 1)

    % We want: M_other' * lambda = m_k, lambda >= 0.
    Aeq = M_other.';              % n_var x (n_ineq-1)
    beq = m_k;                    % n_var x 1

    n_lambda = size(Aeq,2);
    f = zeros(n_lambda,1);        % objective doesn't matter (feasibility problem)

    % Bounds: lambda >= 0
    lb = zeros(n_lambda,1);
    ub = inf(n_lambda,1);

    % linprog signature:
    [lambda, fval] = linprog(f, [], [], Aeq, beq, lb, ub); % OCTAVE version
    %[lambda, fval, exitflag] = linprog(f, [], [], Aeq, beq, lb, ub, options); % MATLAB version

   redundant = false;

    if ~isempty(lambda) && all(isfinite(lambda))
        % Check equality Aeq*lambda ≈ beq
        eq_resid = norm(Aeq*lambda - beq, Inf);
        % Check lambda >= 0 (up to numerical noise)
        min_lambda = min(lambda);

        if eq_resid <= tol_eq && min_lambda >= -tol_lb
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
inequalities_reduced = inequalities(keep);
A_reduced = A(keep,:);

fprintf('\nEssential inequalities (kept):\n');
disp(find(keep).');   % indices of non-redundant inequalities

fprintf('\nReduced coefficient matrix A_reduced:\n');
disp(A_reduced);

fprintf('\nReduced symbolic inequalities inequalities_reduced <= 0:\n');
disp(inequalities_reduced);
