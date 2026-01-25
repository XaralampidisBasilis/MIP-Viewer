
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