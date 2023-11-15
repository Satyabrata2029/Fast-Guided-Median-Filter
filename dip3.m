%MATLAB implementation for "Fast Guided Median Filter"
%using O(r) sliding window approach
%By- Satyabrata Pradhan(23EE65R20)
%Contributed by- Akash Deep(23EE65R21)

% Load grayscale image 'g'
f = double(imread("cat.bmp"));
g = imread("cat.bmp");
g = imnoise(g,"gaussian");
% Convert image to double
g = im2double(g);

% Set the window radius 'r'
r = 2;

%Vary the window radius for time complexity comparision
R = [1 2 3 4 5 6 7 8 9 10 12 15];
T = 0*R;



% Calculate c and d matrices
[c, d] = CalculateCd(g, r);
t1 = cputime;
% Apply the Fast Guided Median Filter
f_star = FastGuidedMedianFilter(g, r, c, d);
t_FGM = cputime-t1;

for i=1:12
    r = R(i);
    t0 = cputime;
    f_star1 = FastGuidedMedianFilter(g,r,c,d);
    T(i) = cputime - t0;
end

plot(R,T); 
xlabel('Window Radius(r)')
ylabel('Computational time(s)')
title('O(r) window search method')


% Display the filtered image
figure()
subplot(1,3,1)
imshow(g);
title('Input Image');

subplot(1,3,3)
imshow(f_star,[]);
title('Filtered Output')

subplot(1,3,2)
imshow(f,[])
title('Guide Image')




function f_star = FastGuidedMedianFilter(g, r, c, d)
    [M, N] = size(g);
    f_star = zeros(M, N);

    for t = 1:M
        for s = 1:N
            W2 = InitializeWindow(g, s, t, r);
            f_star(t, s) = SearchWeightedMedian(W2, c(t, s), d(t, s));
        end
    end
end

function W = InitializeWindow(g, s, t, r)
    [M, N] = size(g);
    window_size = 2 * r + 1;
    
    start_s = max(1, s - r);
    end_s = min(N, s + r);
    start_t = max(1, t - r);
    end_t = min(M, t + r);
    
    F = zeros(1, 256);
    G = zeros(1, 256);
    f_down = 0;
    g_down = 0;
    k = 1;
    
    for j = start_s:end_s
        for i = start_t:end_t
            g_x = round(g(i, j) * 255) + 1;
            
            F(g_x) = F(g_x) + 1;
            G(g_x) = G(g_x) + g(i, j);
            
            if g_x <= k
                f_down = f_down + 1;
                g_down = g_down + g(i, j);
            end
        end
    end
    
    W = {F, G, f_down, g_down, k};
end

function fx_star = SearchWeightedMedian(W, c_x, d_x)
    F = W{1};
    G = W{2};
    f_down = W{3};
    g_down = W{4};
    k = W{5};
    
    h_down = c_x * g_down + d_x * f_down;
    
    while k < 256 && h_down < 0.5
        k = k + 1;
        f_down = f_down + F(k);
        g_down = g_down + G(k);
        h_down = c_x * g_down + d_x * f_down;
    end

    while k > 1 && h_down >= 0.5
        k = k - 1;
        f_down = f_down - F(k);
        g_down = g_down - G(k);
        h_down = c_x * g_down + d_x * f_down;
    end

    fx_star = (k - 1) / 255; % Convert back to [0, 1] range
end

function [c, d] = CalculateCd(g, r)
    [M, N] = size(g);
    c = zeros(M, N);
    d = zeros(M, N);

    for t = 1:M
        for s = 1:N
            % Calculate window boundaries
            start_s = max(1, s - r);
            end_s = min(N, s + r);
            start_t = max(1, t - r);
            end_t = min(M, t + r);

            % Extract the local window centered at pixel (s, t)
            window = g(start_t:end_t, start_s:end_s);

            % Calculate mean and variance of the local window
            mean_g = mean(window(:));
            var_g = var(window(:));

            % Calculate c_x and d_x based on the formulas
            c(t, s) = (g(t, s) - mean_g) / (var_g * numel(window));
            d(t, s) = (1 / numel(window)) - (mean_g * c(t, s));
        end
    end
end
