clc,clear

% pkg load symbolic % OCTAVE version
% pkg load optim % OCTAVE version

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

x_t = ax*(1-t) + bx*t;
y_t = ay*(1-t) + by*t;
z_t = az*(1-t) + bz*t;

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
%% Maxima of trilinear derivative over ray
%% --------------------------------------------------------------------

maxima_terms = [
    simplify(Bf_t_coeffs(2)), ...
    simplify(Bf_t_coeffs(3)), ...
    simplify(Bf_t_coeffs(4)), ...
];

maxima = [
    simplify(subs( Bf_t_coeffs(1), [a, b], [[0,0,0], [0,0,1]]) ), ...
    simplify(subs( Bf_t_coeffs(1), [a, b], [[0,0,0], [0,1,1]]) ), ...
    simplify(subs( Bf_t_coeffs(1), [a, b], [[0,0,0], [1,0,1]]) ), ...
    simplify(subs( Bf_t_coeffs(1), [a, b], [[0,0,0], [1,1,1]]) ), ...
    simplify(subs( Bf_t_coeffs(1), [a, b], [[0,1,0], [0,1,0]]) ), ...
    simplify(subs( Bf_t_coeffs(1), [a, b], [[0,1,0], [0,1,1]]) ), ...
    simplify(subs( Bf_t_coeffs(1), [a, b], [[0,1,0], [1,1,1]]) ), ...
    simplify(subs( Bf_t_coeffs(1), [a, b], [[0,0,1], [0,0,1]]) ), ...
    simplify(subs( Bf_t_coeffs(1), [a, b], [[0,1,1], [0,1,1]]) ), ...
    
    simplify(subs( Bf_t_coeffs(2), [a, b], [[0,0,0], [0,0,1]]) ), ...
    simplify(subs( Bf_t_coeffs(2), [a, b], [[0,0,0], [0,1,1]]) ), ...
    simplify(subs( Bf_t_coeffs(2), [a, b], [[0,0,0], [1,0,1]]) ), ...
    simplify(subs( Bf_t_coeffs(2), [a, b], [[0,0,0], [1,1,1]]) ), ...
    simplify(subs( Bf_t_coeffs(2), [a, b], [[0,1,0], [0,1,0]]) ), ...
    simplify(subs( Bf_t_coeffs(2), [a, b], [[0,1,0], [0,1,1]]) ), ...
    simplify(subs( Bf_t_coeffs(2), [a, b], [[0,1,0], [1,1,1]]) ), ...
    simplify(subs( Bf_t_coeffs(2), [a, b], [[0,0,1], [0,0,1]]) ), ...
    simplify(subs( Bf_t_coeffs(2), [a, b], [[0,1,1], [0,1,1]]) ), ...

    simplify(subs( Bf_t_coeffs(3), [a, b], [[0,0,0], [0,0,1]]) ), ...
    simplify(subs( Bf_t_coeffs(3), [a, b], [[0,0,0], [0,1,1]]) ), ...
    simplify(subs( Bf_t_coeffs(3), [a, b], [[0,0,0], [1,0,1]]) ), ...
    simplify(subs( Bf_t_coeffs(3), [a, b], [[0,0,0], [1,1,1]]) ), ...
    simplify(subs( Bf_t_coeffs(3), [a, b], [[0,1,0], [0,1,0]]) ), ...
    simplify(subs( Bf_t_coeffs(3), [a, b], [[0,1,0], [0,1,1]]) ), ...
    simplify(subs( Bf_t_coeffs(3), [a, b], [[0,1,0], [1,1,1]]) ), ...
    simplify(subs( Bf_t_coeffs(3), [a, b], [[0,0,1], [0,0,1]]) ), ...
    simplify(subs( Bf_t_coeffs(3), [a, b], [[0,1,1], [0,1,1]]) ), ...

    simplify(subs( Bf_t_coeffs(4), [a, b], [[0,0,0], [0,0,1]]) ), ...
    simplify(subs( Bf_t_coeffs(4), [a, b], [[0,0,0], [0,1,1]]) ), ...
    simplify(subs( Bf_t_coeffs(4), [a, b], [[0,0,0], [1,0,1]]) ), ...
    simplify(subs( Bf_t_coeffs(4), [a, b], [[0,0,0], [1,1,1]]) ), ...
    simplify(subs( Bf_t_coeffs(4), [a, b], [[0,1,0], [0,1,0]]) ), ...
    simplify(subs( Bf_t_coeffs(4), [a, b], [[0,1,0], [0,1,1]]) ), ...
    simplify(subs( Bf_t_coeffs(4), [a, b], [[0,1,0], [1,1,1]]) ), ...
    simplify(subs( Bf_t_coeffs(4), [a, b], [[0,0,1], [0,0,1]]) ), ...
    simplify(subs( Bf_t_coeffs(4), [a, b], [[0,1,1], [0,1,1]]) ), ...
];

maxima = unique(maxima);
maxima = maxima(:);
maxima_reduced = reduceMaximaSubconvex(maxima, F, 1e-8, false);

fprintf('\nReduced symbolic total maxima:\n');
disp(maxima_reduced);

% maxima_reduced(1)  = (f000 + f001 + f100)/3;
% maxima_reduced(2)  = (f001 + f010 + f100)/3;
% maxima_reduced(3)  = (f010 + f011 + f110)/3;
% maxima_reduced(4)  = (f001 + f100 + f101)/3;
% maxima_reduced(5)  = (f011 + f101 + f110)/3;
% maxima_reduced(6)  = (f011 + f110 + f111)/3;
% maxima_reduced(7)  = f000;
% maxima_reduced(8)  = f001;
% maxima_reduced(9)  = f010;
% maxima_reduced(10) = f011;
% maxima_reduced(11) = f101;
% maxima_reduced(12) = f111;

function [maxima_reduced, keep] = reduceMaximaSubconvex(maxima, F, tol, verbose)
%reduceMaximaSubconvex Remove redundant maxima via subconvex combination test.
%
% Inputs
%   maxima   : (n_expr x 1) symbolic vector of expressions in variables F
%   F        : (1 x n_var) or (n_var x 1) symbolic vector of variables [f000 f001 ...]
%   tol      : (optional) tolerance for redundancy check (default 1e-8)
%   verbose  : (optional) print progress (default true)
%
% Outputs
%   maxima_reduced     : -maxima_sorted(keep)
%   A_reduced          : A(keep,:)
%   keep               : logical mask of kept (non-redundant) inequalities
%   maxima_sorted      : maxima after sorting by complexity (descend)
%   order              : permutation indices used for sorting
%   complexity_sorted  : sorted complexity scores
%   A                  : full coefficient matrix after sorting

    if nargin < 3 || isempty(tol), tol = 1e-8; end
    if nargin < 4 || isempty(verbose), verbose = true; end

    % Octave detection (for small compatibility tweaks)
    isOctave = (exist('OCTAVE_VERSION','builtin') ~= 0);

    % Ensure column vectors where convenient
    maxima = maxima(:);
    F = F(:).';  % make F a row for subs(maxima, F, basis)

    %% ---------------------------------------------------------------
    % Sort maxima by expression complexity 
    %% ---------------------------------------------------------------
    n_expr = length(maxima);
    complexity = zeros(n_expr, 1);
    for ii = 1:n_expr
        complexity(ii) = length(char(maxima(ii)));
    end

    [complexity_sorted, order] = sort(complexity, 'descend');
    maxima_sorted = maxima(order);

    if verbose
        fprintf("Sorted expressions (most complex to simplest):\n");
        disp(maxima_sorted);
    end

    %% ---------------------------------------------------------------
    % Build coefficient matrix A: rows correspond to maxima expressions
    %% A * F <= 0 (after your sign convention)
    %% ---------------------------------------------------------------
    n_expr = length(maxima_sorted);
    n_var  = length(F);

    A_sym = sym(zeros(n_expr, n_var));
    for j = 1:n_var
        % basis vector e_j: F(j)=1, others=0
        basis = sym(zeros(1, n_var));   % sym to avoid float->sym warnings
        basis(j) = sym(1);
        A_sym(:, j) = subs(maxima_sorted, F, basis);
    end

    A = double(A_sym);

    if verbose
        fprintf('Coefficient matrix A * F <= 0:\n');
        disp(A);
    end

    %% ---------------------------------------------------------------
    % Search for redundant inequalities
    %% ---------------------------------------------------------------
    keep = true(n_expr, 1);

    for k = 1:n_expr
        rows = keep(:);
        rows(k) = false;

        M_other = A(rows, :);   % (n_other) x n_var
        m_k = A(k, :).';        % n_var x 1

        if isempty(M_other)
            if verbose
                fprintf('Inequality %d appears essential (no others).\n', k);
            end
            continue;
        end

        % Want: M_other' * lambda = m_k, sum(lambda) <= 1, lambda >= 0
        Aeq = M_other.';        % n_var x n_lambda
        beq = m_k;              % n_var x 1
        n_lambda = size(Aeq, 2);

        f = ones(n_lambda, 1);          % arbitrary objective
        Aineq = ones(1, n_lambda);
        bineq = 1;
        lb = zeros(n_lambda, 1);

        % MATLAB allows ub = []; some Octave builds are happier with Inf vector
        if isOctave
            ub = Inf(n_lambda, 1);
        else
            ub = [];
        end

        lambda = [];
        try
            % MATLAB: lambda = linprog(...)
            % Octave optim: linprog exists; outputs may vary but first is x
            lambda = linprog(f, Aineq, bineq, Aeq, beq, lb, ub);
        catch
            % If your linprog requires options or different signature, fail loudly
            error(['linprog failed. In Octave, ensure "pkg load optim" and that linprog is available. ', ...
                   'Original error: %s'], lasterr());
        end

        redundant = false;
        if ~isempty(lambda)
            req = norm(Aeq * lambda - beq, Inf);
            sum_lambda = sum(lambda);

            if req <= tol && sum_lambda <= 1 + tol && all(lambda >= -tol)
                redundant = true;
            end
        end

        if redundant
            keep(k) = false;
            if verbose
                fprintf('Inequality %d is redundant. sum(lambda) = %.6g\n', k, sum_lambda);
            end
        else
            if verbose
                if ~isempty(lambda)
                    fprintf('Inequality %d appears essential.\n', k);
                else
                    fprintf('Inequality %d appears essential (no feasible lambda).\n', k);
                end
            end
        end
    end

    %% ---------------------------------------------------------------
    % Reduced system outputs
    %% ---------------------------------------------------------------
    maxima_reduced = maxima_sorted(keep);

    if verbose
        fprintf('\nEssential maxima (kept):\n');
        disp(find(keep).');   % indices in the *sorted* list
        
        fprintf('\nReduced symbolic maxima:\n');
        disp(maxima_reduced);
    end
end