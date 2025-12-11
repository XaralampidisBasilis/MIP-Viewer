clc,clear

%pkg load symbolic % OCTAVE version
%pkg load optim % OCTAVE version


%% --------------------------------------------------------------------
%% 1. Declare symbols
%% --------------------------------------------------------------------
syms x y z t real

% corner values
syms f000 f100 f010 f001 f011 f101 f110 f111 real

% Ray endpoints
syms ax ay az bx by bz dx dy dz real

F = [f000 f100 f010 f001 f011 f101 f110 f111];
pa = [ax, ay, az];
pb = [bx, by, bz];
pd = [dx, dy, dz];

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
% x_t = ax + dx*t;
% y_t = ay + dy*t;
% z_t = az + dz*t;

x_t = bx - dx*(1-t);
y_t = by - dy*(1-t);
z_t = bz - dz*(1-t);

% x_t = ax*(1-t) + bx*t;
% y_t = ay*(1-t) + by*t;
% z_t = az*(1-t) + bz*t;

f_t = simplify( subs(f_xyz, [x y z], [x_t, y_t, z_t]) );
f_t = collect(f_t, t);

%% Extract coefficients (efficient for GLSL)
[f_t_coeffs, f_t_terms] = coeffs(f_t, t);

disp("Mapped expression coefficients and terms");
disp([f_t_coeffs(:), f_t_terms(:)]);

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
BF = [Bf0, Bf1, Bf2, Bf3];

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
%% Compute bernstein maxima of f(t) in t [0, 1], when ax = 0
%% --------------------------------------------------------------------

Bf03 = simplify(Bf0 - Bf3);
Bf13 = simplify(Bf1 - Bf3);
Bf23 = simplify(Bf2 - Bf3);

Bf03_subs = simplify(subs(Bf03, [bx, dx], [1, 1]));
Bf13_subs = simplify(subs(Bf13, [bx, dx], [1, 1]));
Bf23_subs = simplify(subs(Bf23, [bx, dx], [1, 1]));

maxima = [
    simplify(subs(Bf03_subs, [by, bz, dy, dz], [0,0,0,0])), ...
    simplify(subs(Bf03_subs, [by, bz, dy, dz], [0,0,0,1])), ...
    simplify(subs(Bf03_subs, [by, bz, dy, dz], [0,0,1,0])), ...
    simplify(subs(Bf03_subs, [by, bz, dy, dz], [0,0,1,1])), ...
    simplify(subs(Bf03_subs, [by, bz, dy, dz], [0,1,0,0])), ...
    simplify(subs(Bf03_subs, [by, bz, dy, dz], [0,1,0,1])), ...
    simplify(subs(Bf03_subs, [by, bz, dy, dz], [0,1,1,0])), ...
    simplify(subs(Bf03_subs, [by, bz, dy, dz], [0,1,1,1])), ...
    simplify(subs(Bf03_subs, [by, bz, dy, dz], [1,0,0,0])), ...
    simplify(subs(Bf03_subs, [by, bz, dy, dz], [1,0,0,1])), ...
    simplify(subs(Bf03_subs, [by, bz, dy, dz], [1,0,1,0])), ...
    simplify(subs(Bf03_subs, [by, bz, dy, dz], [1,0,1,1])), ...
    simplify(subs(Bf03_subs, [by, bz, dy, dz], [1,1,0,0])), ...
    simplify(subs(Bf03_subs, [by, bz, dy, dz], [1,1,0,1])), ...
    simplify(subs(Bf03_subs, [by, bz, dy, dz], [1,1,1,0])), ...
    simplify(subs(Bf03_subs, [by, bz, dy, dz], [1,1,1,1])), ...
    simplify(subs(Bf13_subs, [by, bz, dy, dz], [0,0,0,0])), ...
    simplify(subs(Bf13_subs, [by, bz, dy, dz], [0,0,0,1])), ...
    simplify(subs(Bf13_subs, [by, bz, dy, dz], [0,0,1,0])), ...
    simplify(subs(Bf13_subs, [by, bz, dy, dz], [0,0,1,1])), ...
    simplify(subs(Bf13_subs, [by, bz, dy, dz], [0,1,0,0])), ...
    simplify(subs(Bf13_subs, [by, bz, dy, dz], [0,1,0,1])), ...
    simplify(subs(Bf13_subs, [by, bz, dy, dz], [0,1,1,0])), ...
    simplify(subs(Bf13_subs, [by, bz, dy, dz], [0,1,1,1])), ...
    simplify(subs(Bf13_subs, [by, bz, dy, dz], [1,0,0,0])), ...
    simplify(subs(Bf13_subs, [by, bz, dy, dz], [1,0,0,1])), ...
    simplify(subs(Bf13_subs, [by, bz, dy, dz], [1,0,1,0])), ...
    simplify(subs(Bf13_subs, [by, bz, dy, dz], [1,0,1,1])), ...
    simplify(subs(Bf13_subs, [by, bz, dy, dz], [1,1,0,0])), ...
    simplify(subs(Bf13_subs, [by, bz, dy, dz], [1,1,0,1])), ...
    simplify(subs(Bf13_subs, [by, bz, dy, dz], [1,1,1,0])), ...
    simplify(subs(Bf13_subs, [by, bz, dy, dz], [1,1,1,1])), ...
    simplify(subs(Bf23_subs, [by, bz, dy, dz], [0,0,0,0])), ...
    simplify(subs(Bf23_subs, [by, bz, dy, dz], [0,0,0,1])), ...
    simplify(subs(Bf23_subs, [by, bz, dy, dz], [0,0,1,0])), ...
    simplify(subs(Bf23_subs, [by, bz, dy, dz], [0,0,1,1])), ...
    simplify(subs(Bf23_subs, [by, bz, dy, dz], [0,1,0,0])), ...
    simplify(subs(Bf23_subs, [by, bz, dy, dz], [0,1,0,1])), ...
    simplify(subs(Bf23_subs, [by, bz, dy, dz], [0,1,1,0])), ...
    simplify(subs(Bf23_subs, [by, bz, dy, dz], [0,1,1,1])), ...
    simplify(subs(Bf23_subs, [by, bz, dy, dz], [1,0,0,0])), ...
    simplify(subs(Bf23_subs, [by, bz, dy, dz], [1,0,0,1])), ...
    simplify(subs(Bf23_subs, [by, bz, dy, dz], [1,0,1,0])), ...
    simplify(subs(Bf23_subs, [by, bz, dy, dz], [1,0,1,1])), ...
    simplify(subs(Bf23_subs, [by, bz, dy, dz], [1,1,0,0])), ...
    simplify(subs(Bf23_subs, [by, bz, dy, dz], [1,1,0,1])), ...
    simplify(subs(Bf23_subs, [by, bz, dy, dz], [1,1,1,0])), ...
    simplify(subs(Bf23_subs, [by, bz, dy, dz], [1,1,1,1])), ...
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
%% --------------------------------------------------------------------
keep = true(n_expr,1);  % assume all are essential initially
tol_eq = 1e-8;   % tolerance for Aeq*lambda ≈ beq
tol_lb = 1e-8;  % tolerance for sum(lambda) >= 1

options = optimoptions('linprog', 'Algorithm', 'dual-simplex', 'Display', 'none');  % MATLAB version

for k = 1:n_expr
    % Build M_other: all rows except k
    rows = keep(:);
    rows(k) = false;
    M_other = A(rows,:);          % (n_expr-1) x n_var
    m_k = A(k,:).';               % column (n_var x 1)

    % We want: M_other' * lambda = m_k sum(lambda) <= 1
    Aeq = M_other.';              % n_var x (n_expr-1)
    beq = m_k;                    % n_var x 1
    n_lambda = size(Aeq,2);
    f = zeros(n_lambda,1);        % objective doesn't matter (feasibility problem)

    % linprog signature:
    lambda = linprog(f, [], [], Aeq, beq, [], [], options); % OCTAVE version

    redundant = false;

    if ~isempty(lambda) && all(isfinite(lambda))
        % Check equality Aeq*lambda ≈ beq
        req = norm(Aeq*lambda - beq, Inf);
        % Check sum(lambda) <= 1 (up to numerical noise)
        sum_lambda = sum(lambda);

        if req <= tol_eq && sum_lambda >= 1 - tol_lb
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
