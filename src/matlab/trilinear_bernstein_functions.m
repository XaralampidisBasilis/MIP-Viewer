
pkg load symbolic % OCTAVE version
pkg load optim % OCTAVE version

%% --------------------------------------------------------------------
%% Declare functions
%% --------------------------------------------------------------------


function [cB, ij] = quadrilateralBernsteinCoeffsFromTrilinearInterpolation(F, p1, p2, p3, p4)
% quadrilateralBernsteinCoeffsFromTrilinearInterpolation
% Computes bicubic (3x3) tensor-product Bernstein coefficients of the
% composition of trilinear interpolation f(x,y,z) with the bilinear quad map.
%
% Inputs:
%   F  : trilinear corner values. Either:
%        - 2x2x2 array with F(i+1,j+1,k+1) = f_ijk   (i,j,k in {0,1})
%        - 1x8 vector in order [f000 f100 f010 f001 f011 f101 f110 f111]
%   p1,p2,p3,p4 : 1x3 quad corners in (x,y,z) inside the unit cube.
%        Parameterization corners:
%          (u,v)=(0,0)->p1, (1,0)->p2, (0,1)->p3, (1,1)->p4
%        (Must be consistent—this is the bilinear surface convention.)
%
% Outputs:
%   cB : 1x16 bicubic tensor-product Bernstein coefficients, ordered as in ij
%   ij : 16x2 pairs (i,j), with i,j in {0,1,2,3} matching cB entries
%
% Notes:
%   - The resulting surface is bicubic in (u,v) in general.

    % ---- normalize F to 8 values in standard order ----
    if isequal(size(F), [2 2 2])
        f000 = F(1,1,1); f100 = F(2,1,1); f010 = F(1,2,1); f001 = F(1,1,2);
        f011 = F(1,2,2); f101 = F(2,1,2); f110 = F(2,2,1); f111 = F(2,2,2);
        F8 = [f000 f100 f010 f001 f011 f101 f110 f111];
    else
        F8 = F(:).';
        if numel(F8) ~= 8
            error('F must be 2x2x2 or a 1x8 (or 8x1) vector.');
        end
    end

    % ---- ensure points are 1x3 ----
    p1 = p1(:).'; p2 = p2(:).'; p3 = p3(:).'; p4 = p4(:).';
    if any([numel(p1) numel(p2) numel(p3) numel(p4)] ~= 3)
        error('p1,p2,p3,p4 must each be length-3 vectors.');
    end

    % ---- decide numeric vs symbolic ----
    useSym = isa(F8,'sym') || isa(p1,'sym') || isa(p2,'sym') || isa(p3,'sym') || isa(p4,'sym');
    if useSym
        F8 = sym(F8);
        p1 = sym(p1); p2 = sym(p2); p3 = sym(p3); p4 = sym(p4);
    end

    % ---- bicubic index set (16 terms) ----
    % Order: i varies fastest or j varies fastest—choose one and stick to it.
    % Here: i runs 0..3 inside, then j increments (row-major in (j,i)).
    ij = zeros(16,2);
    idx = 1;
    for j = 0:3
        for i = 0:3
            ij(idx,:) = [i j];
            idx = idx + 1;
        end
    end
    n = 3;  % bicubic degree in u and v

    % ---- lattice points (u,v) = (i/3, j/3) ----
    if useSym
        U = sym(ij(:,1)) / n;
        V = sym(ij(:,2)) / n;
        M = sym(zeros(16,16));
        rhs = sym(zeros(16,1));
    else
        U = ij(:,1) / n;
        V = ij(:,2) / n;
        M = zeros(16,16);
        rhs = zeros(16,1);
    end

    % ---- build 16x16 matrix M: tensor Bernstein basis at lattice points ----
    % Basis term for column c: B_i^3(u) * B_j^3(v)
    % with B_i^3(u)=C(3,i) u^i (1-u)^(3-i)
    for r = 1:16
        u = U(r); v = V(r);
        for c = 1:16
            i = ij(c,1); j = ij(c,2);
            M(r,c) = bern1(n,i,u) * bern1(n,j,v);
        end
    end

    % ---- rhs: sample f at mapped lattice points ----
    for r = 1:16
        u = U(r); v = V(r);

        % bilinear quad map p(u,v)
        p = bilinearQuadPoint(p1,p2,p3,p4,u,v);
        x = p(1); y = p(2); z = p(3);

        rhs(r) = evaluateTrilinearInterpolation(F8, x, y, z);
    end

    % ---- solve for Bernstein coefficients ----
    cB = (M \ rhs).';   % 1x16
    if useSym
        cB = simplify(cB);
    end
end



function [cB, ijk] = triangleBernsteinCoeffsFromTrilinearInterpolation(F, p1, p2, p3)
% triangleBernsteinCoeffsFromTrilinearInterpolation
% Inputs:
%   F  : trilinear corner values. Either:
%        - 2x2x2 array with F(i+1,j+1,k+1) = f_ijk   (i,j,k in {0,1})
%        - 1x8 vector in order [f000 f100 f010 f001 f011 f101 f110 f111]
%   p1,p2,p3 : 1x3 triangle vertices in (x,y,z) inside the unit cube
%              (numeric or symbolic)
%
% Outputs:
%   cB  : 1x10 degree-3 triangle Bernstein coefficients, ordered as in ijk
%   ijk : 10x3 exponent triples (i,j,k) with i+j+k=3 matching cB entries

    % ---- normalize F to 8 values in standard order ----
    if isequal(size(F), [2 2 2])
        f000 = F(1,1,1); f100 = F(2,1,1); f010 = F(1,2,1); f001 = F(1,1,2);
        f011 = F(1,2,2); f101 = F(2,1,2); f110 = F(2,2,1); f111 = F(2,2,2);
        F8 = [f000 f100 f010 f001 f011 f101 f110 f111];
    else
        F8 = F(:).';
        if numel(F8) ~= 8
            error('F must be 2x2x2 or a 1x8 (or 8x1) vector.');
        end
    end

    % ---- ensure p1,p2,p3 are 1x3 ----
    p1 = p1(:).'; p2 = p2(:).'; p3 = p3(:).';
    if numel(p1)~=3 || numel(p2)~=3 || numel(p3)~=3
        error('p1,p2,p3 must each be length-3 vectors.');
    end

    % ---- decide numeric vs symbolic ----
    useSym = isa(F8,'sym') || isa(p1,'sym') || isa(p2,'sym') || isa(p3,'sym');
    if useSym
        F8 = sym(F8);
        p1 = sym(p1); p2 = sym(p2); p3 = sym(p3);
    end

    % ---- degree-3 triangle index set (10 terms) ----
    ijk = [
        3 0 0;
        2 1 0;
        2 0 1;
        1 2 0;
        1 1 1;
        1 0 2;
        0 3 0;
        0 2 1;
        0 1 2;
        0 0 3
    ];
    n = 3;

    % ---- degree-3 barycentric lattice points (i/3,j/3,k/3) ----
    if useSym
        Lpts = sym(ijk) / n;
    else
        Lpts = ijk / n;
    end

    % ---- build constant 10x10 matrix M: Bernstein basis at lattice points ----
    % B_{abc}^3(l) = 3!/(a!b!c!) * l1^a l2^b l3^c
    multinom = @(a,b,c) factorial(3)/(factorial(a)*factorial(b)*factorial(c));

    if useSym
        M = sym(zeros(10,10));
    else
        M = zeros(10,10);
    end

    for r = 1:10
        l1 = Lpts(r,1); l2 = Lpts(r,2); l3 = Lpts(r,3);
        for c = 1:10
            a = ijk(c,1); b = ijk(c,2); c3 = ijk(c,3);
            M(r,c) = multinom(a,b,c3) * (l1^a) * (l2^b) * (l3^c3);
        end
    end

    % ---- rhs: sample the trilinear function on the mapped lattice points ----
    if useSym
        rhs = sym(zeros(10,1));
    else
        rhs = zeros(10,1);
    end

    for r = 1:10
        l1 = Lpts(r,1); l2 = Lpts(r,2); l3 = Lpts(r,3);

        % affine map to xyz
        p = l1*p1 + l2*p2 + l3*p3;
        x = p(1); y = p(2); z = p(3);

        % trilinear evaluation at (x,y,z)
        rhs(r) = evaluateTrilinearInterpolation(F8, x, y, z);
    end

    % ---- solve for Bernstein coefficients ----
    cB = (M \ rhs).';   % 1x10
    cB = simplify(cB);

end



% === helper: trilinear eval from 8 corners in standard order ===
function val = evaluateTrilinearInterpolation(F8, x, y, z)
% F8 order: [f000 f100 f010 f001 f011 f101 f110 f111]
    f000 = F8(1); f100 = F8(2); f010 = F8(3); f001 = F8(4);
    f011 = F8(5); f101 = F8(6); f110 = F8(7); f111 = F8(8);

    val = f000*(1-x)*(1-y)*(1-z) + ...
          f100*(  x)*(1-y)*(1-z) + ...
          f010*(1-x)*(  y)*(1-z) + ...
          f001*(1-x)*(1-y)*(  z) + ...
          f011*(1-x)*(  y)*(  z) + ...
          f101*(  x)*(1-y)*(  z) + ...
          f110*(  x)*(  y)*(1-z) + ...
          f111*(  x)*(  y)*(  z);
end



% === helper: bilinear quad map from (u,v) in [0,1]^2 ===
function p = bilinearQuadPoint(p1,p2,p3,p4,u,v)
    p = (1-u)*(1-v)*p1 + u*(1-v)*p2 + (1-u)*v*p3 + u*v*p4;
end



% === helper: 1D Bernstein basis ===
function val = bern1(n, i, t)
    val = nchoosek(n,i) * (t^i) * ((1-t)^(n-i));
end



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