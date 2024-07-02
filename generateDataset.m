%% filename : generateDataset.m

%% PLANT
%                               1     
% y(k) = ------------------------------------------------
%        (1 + y(k-1)^2)) - (0.25 * u(k)) - (0.3 * u(k-1))

%% Generate Dataset

% Data Random
y = zeros(12000, 1);                
u = (1 + 1) * rand(12000, 1) - 1;

[u_row, u_column] = size(u);
[y_row, y_column] = size(y);

for k =1:y_row
    if k == 1
        y(k) = (1 / (1 + 0)) - (0.25 * u(k)) - (0.3 * 0);
    else 
        y(k) = (1 / (1 + y(k-1)^2)) - (0.25 * u(k)) - (0.3 * u(k-1));
    end
end

% INPUT
uk = zeros(u_row, 1);
uk_min1 = zeros(u_row, 1);
uk_min2 = zeros(u_row, 1);
yk = zeros(y_row, 1);
yk_min1 = zeros(y_row, 1);
yk_min2 = zeros(y_row, 1);


for i = 1:u_row
    uk(i) = u(i);
    yk(i) = y(i);
end

for i = 1:(u_row - 1)
    uk_min1(i + 1) = u(i);
    yk_min1(i + 1) = y(i);
end

for i = 1:(u_row - 2)
    uk_min2(i + 2) = u(i);
    yk_min2(i + 2) = y(i); 
end

dataset = cat(2, uk, uk_min1, uk_min2, yk_min1, yk_min2, yk);

dataTable = array2table(dataset);
dataTable = renamevars(dataTable, ["dataset1", "dataset2", "dataset3", "dataset4", "dataset5", "dataset6"], ["u(k)", "u(k-1)", "u(k-2)", "y(k-1)", "y(k-2)", "y(k)/output"]);

writetable(dataTable,'dataset.xlsx','Sheet',1);