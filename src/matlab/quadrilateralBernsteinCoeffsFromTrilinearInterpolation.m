
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
