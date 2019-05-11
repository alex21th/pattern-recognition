
% Populates a data set for the SLNN problem.

function [X,y] = om_uo_nn_dataset(seed, ncol, target, freq, noise_freq)

global N;

N = [
    0 0 1 0 0, 0 1 1 0 0, 0 0 1 0 0, 0 0 1 0 0, 0 0 1 0 0, 0 0 1 0 0, 0 1 1 1 0; % 1
    0 1 1 1 0, 1 0 0 0 1, 0 0 0 0 1, 0 0 0 1 0, 0 0 1 0 0, 0 1 0 0 0, 1 1 1 1 1; % 2
    0 1 1 1 0, 1 0 0 0 1, 0 0 0 0 1, 0 0 1 1 0, 0 0 0 0 1, 1 0 0 0 1, 0 1 1 1 0; % 3
    0 0 1 1 0, 0 1 0 1 0, 1 0 0 1 0, 1 0 0 1 0, 1 1 1 1 1, 0 0 0 1 0, 0 0 0 1 0; % 4
    1 1 1 1 1, 1 0 0 0 0, 1 1 1 1 0, 0 0 0 0 1, 0 0 0 0 1, 1 0 0 0 1, 0 1 1 1 0; % 5
    0 0 1 1 1, 0 1 0 0 0, 1 0 0 0 0, 1 1 1 1 0, 1 0 0 0 1, 1 0 0 0 1, 0 1 1 1 0; % 6
    1 1 1 1 1, 0 0 0 0 1, 0 0 0 0 1, 0 0 0 1 0, 0 0 1 0 0, 0 1 0 0 0, 1 0 0 0 0; % 7
    0 1 1 1 0, 1 0 0 0 1, 1 0 0 0 1, 0 1 1 1 0, 1 0 0 0 1, 1 0 0 0 1, 0 1 1 1 0; % 8
    0 1 1 1 0, 1 0 0 0 1, 1 0 0 0 1, 0 1 1 1 1, 0 0 0 0 1, 0 0 0 0 1, 0 1 1 1 0; % 9
    0 1 1 1 0, 1 0 0 0 1, 1 0 0 0 1, 1 0 0 0 1, 1 0 0 0 1, 1 0 0 0 1, 0 1 1 1 0; % 0
    ];

N = N';
rng(seed); nT = size(target,2); nPixels = size(N,1);

% Generation
for j=1:ncol
    if rand() < (freq-0.1*nT)/(1-0.1*nT) i = target(randi([1,nT]));
    else                                 i = randi(10);
    end
    X(:,j) = N(:,i);
    if any(i == target) y(j) = 1;
    else                y(j) = 0;
    end
end

% Noise
for j=1:ncol
    for k = 1:round(noise_freq*nPixels)
        ii = randi([1 nPixels]);
        X(ii,j) = mod(1,X(ii,j));
    end
end
end