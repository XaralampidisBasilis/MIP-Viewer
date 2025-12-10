clc,clear

%% --------------------------------------------------------------------
%% Include trilinear interpolation equations
%% --------------------------------------------------------------------
run('ray_trilinear_interpolation.m')

%% --------------------------------------------------------------------
%% Convert ray trilinear interpolation equation into bernstein form
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
%% Convert ray trilinear interpolation symmetric equation into bernstein form
%% --------------------------------------------------------------------

% We assume f(t) is a cubic polynomial in t:
% f(t) = c0 + c1*t + c2*t^2 + c3*t^3

Cv0 = simplify( v_t_coeffs(4) );
Cv1 = simplify( v_t_coeffs(3) );
Cv2 = simplify( v_t_coeffs(2) );  
Cv3 = simplify( v_t_coeffs(1) );  

% Bernstein coefficients for degree-3 polynomial on [0,1]:
Bv0 = simplify(Cv0);
Bv1 = simplify(Cv0 + Cv1*1/3);
Bv2 = simplify(Cv0 + Cv1*2/3 + Cv2*1/3);
Bv3 = simplify(Cv0 + Cv1 + Cv2 + Cv3);

% Collect coefficients
Bv0 = collect(Bv0, V);
Bv1 = collect(Bv1, V);
Bv2 = collect(Bv2, V);
Bv3 = collect(Bv3, V);
BV = [Bv0, Bv1, Bv2, Bv3];

% Bernstein-form polynomial (for verification / export)
Bv_t = simplify( ...
    Bv0 * 1 * t^0 * (1 - t)^3 ...
  + Bv1 * 3 * t^1 * (1 - t)^2 ...
  + Bv2 * 3 * t^2 * (1 - t)^1 ...
  + Bv3 * 1 * t^3 * (1 - t)^0);

disp("Bernstein coefficients");
disp(Bv0);
disp(Bv1);
disp(Bv2);
disp(Bv3);

disp("Sanity check v_t - Bv_t, should be 0:");
disp(simplify(v_t - Bv_t));



