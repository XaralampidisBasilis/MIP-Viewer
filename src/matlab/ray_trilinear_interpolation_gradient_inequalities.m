clc,clear

pkg load symbolic
pkg load optim

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
% and we have dy/dx, dz/dx in [0, 1] from restrictions
% we set dx = 1

inequalities = [
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [0,0,0,1,0,0])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [0,0,0,1,0,1])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [0,0,0,1,1,0])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [0,0,0,1,1,1])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [0,0,1,1,0,0])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [0,0,1,1,0,1])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [0,0,1,1,1,0])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [0,0,1,1,1,1])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [0,1,0,1,0,0])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [0,1,0,1,0,1])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [0,1,0,1,1,0])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [0,1,0,1,1,1])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [0,1,1,1,0,0])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [0,1,1,1,0,1])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [0,1,1,1,1,0])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [0,1,1,1,1,1])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [1,0,0,1,0,0])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [1,0,0,1,0,1])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [1,0,0,1,1,0])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [1,0,0,1,1,1])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [1,0,1,1,0,0])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [1,0,1,1,0,1])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [1,0,1,1,1,0])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [1,0,1,1,1,1])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [1,1,0,1,0,0])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [1,1,0,1,0,1])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [1,1,0,1,1,0])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [1,1,0,1,1,1])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [1,1,1,1,0,0])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [1,1,1,1,0,1])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [1,1,1,1,1,0])), ...
    simplify(subs(Df_xyz, [x, y, z, dx, dy, dz], [1,1,1,1,1,1])), ...
];

inequalities = unique(inequalities);
inequalities = inequalities(:);

%% --------------------------------------------------------------------
%  Build coefficient matrix A
%% --------------------------------------------------------------------
n_ineq = length(inequalities);
n_var  = length(F);

A_sym = sym(zeros(n_ineq, n_var));
zeroF = num2cell(zeros(1, n_var));   % F = 0

for i = 1:n_ineq
for j = 1:n_var
    A_sym(i,j) = subs(diff(inequalities(i), F(j)), F, zeroF);
end
end

A = double(A_sym);  % numeric matrix (all coefficients are small integers)

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


% Assuming no need for options, remove them for simplicity
for k = 1:n_ineq

    % Build M_other: all rows except k
    rows = true(n_ineq, 1);
    rows(k) = false;
    M_other = A(rows, :);  % (n_ineq-1) x n_var
    m_k = A(k, :)';        % column (n_var x 1)

    % We want: M_other' * lambda = m_k, lambda >= 0.
    Aeq = M_other.';        % n_var x (n_ineq-1)
    beq = m_k;              % n_var x 1

    n_lambda = size(Aeq, 2);
    f = zeros(n_lambda, 1);  % No objective function, so set to zero vector

    % Bounds: lambda >= 0
    lb = zeros(n_lambda, 1);  % Lower bound (lambda >= 0)
    ub = inf(n_lambda, 1);    % Upper bound (lambda <= infinity)

    % Correctly call linprog without options
    [lambda, fval] = linprog(f, [], [], Aeq, beq, lb, ub);

    % Check the result to determine if the inequality is redundant
    if all(lambda >= 0)
        % Found a feasible (actually optimal) solution: A(k,:) is in the cone
        % generated by the other rows → inequality k is redundant.
        keep(k) = false;
        fprintf('Inequality %d appears redundant.\n', k);
    else
        % If linprog returns a solution, it means the inequality is redundant
        fprintf('Inequality %d is essential.\n', k);
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
