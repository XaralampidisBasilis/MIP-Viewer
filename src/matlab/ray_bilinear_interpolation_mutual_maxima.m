clc, clear

pkg load symbolic % OCTAVE version
pkg load optim    % OCTAVE version (not used here)

%% --------------------------------------------------------------------
%% 1. Declare symbols (2D)
%% --------------------------------------------------------------------
syms x y tA tB real

% corner values (bilinear: 4 corners)
syms f00 f10 f20 f01 f11 f21 real
Fij = [f00 f10 f20 f01 f11 f21];

%% --------------------------------------------------------------------
%% 2. Bilinear interpolation f(x,y)
%% --------------------------------------------------------------------
FA_xy = f00 * (1-x) * (1-y) ...
      + f10 *    x  * (1-y) ...
      + f01 * (1-x) *    y  ...
      + f11 *    x  *    y;

FB_xy = f10 * (1-x) * (1-y) ...
      + f20 *    x  * (1-y) ...
      + f11 * (1-x) *    y  ...
      + f21 *    x  *    y;

%% --------------------------------------------------------------------
%% 3. Ray substitution r(t) = a + d*t  (equivalently a(1-t)+b*t with d=b-a)
%% --------------------------------------------------------------------

% Ray endpoints / direction in 2D
syms px py dxA dxB sy real

xA_tA = px - dxA*tA;
yA_tA = py - dxA*sy*tA;

xB_tB = px + dxB*tB;
yB_tB = py + dxB*sy*tB;

FA_tA = simplify( subs(FA_xy, [x y], [xA_tA, yA_tA]) );
FA_tA = collect(FA_tA, tA);

FB_tB = simplify( subs(FB_xy, [x y], [xB_tB, yB_tB]) );
FB_tB = collect(FB_tB, tB);

GA_tA = factor(simplify(FA_tA - subs(FA_tA, tA, 0)));
GB_tB = factor(simplify(FB_tB - subs(FB_tB, tB, 0)));
