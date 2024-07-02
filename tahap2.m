clc;
clear;

% Impor data
dataTable = readtable('dataset.xlsx', 'sheet', 'Sheet1');

% Penentuan feature dan target dari dataTable
feature = dataTable(:, 1:end-1);
target = dataTable(:, end);

% Normalisasi data feature
feature = normalize(feature, 'range', [-1, 1]);
target = normalize(target, 'range', [-1 1]);

% Mencari tahu jumlah baris dan kolom dari feature
[feature_num_row, feature_num_column] = size(feature);

% Mencari tahu jumlah baris dan kolom dari target
[target_num_row, target_num_column] = size(target);

% Pemisahan Data Training, Data Validasi, dan Data Testing
X_train = table2array(feature(1:5000, :));
y_train = table2array(target(1:5000, :));
vektor_y_plant_train = y_train;

X_val = table2array(feature(5000 + 1:7000, :));
y_val = table2array(target(5000 + 1:7000, :));
vektor_y_plant_val = y_val;

X_test = table2array(feature(7000 + 1:end, :));
y_test = table2array(target(7000 + 1:end, :));
vektor_y_plant_test = y_test;


% Model BPNN
% Inisialisasi jumlah input layer, hidden layer, dan output layer
n = feature_num_column;             % input layer
p = 10;                             % hidden layer 
m = target_num_column;              % output layer


% Penentuan nilai bobot secara random dalam skala -0.5 sampai 0.5
a = -0.5;
b = 0.5;

V = rand(n, p) + a;     % Bobot V
W = rand(p, m) - b;     % Bobot W

% Optimasi nilai bobot menggunakan metode Nguyen-Widrow
beta_V = 0.7 * (p) .^ (1/n);    % Nilai beta V
beta_W = 0.7 * (m) .^ (1/p);    % Nilai beta W

V_0j = -beta_V + (beta_V - (-beta_V)) .* rand(1,p);     % Inisialisasi nilai bobot V0j
W_0k = -beta_W + (beta_W - (-beta_W) .* rand(1,m));     % Inisialisasi nilai bobot W0k

V_j = sqrt(sum(V.^2));                                  % Inisialisasi nilai bobot V_j
W_k = sqrt(sum(W.^2));                                  % Inisialisasi nilai bobot W_k    

Vij_new = (beta_V .* V) ./ V_j;                         % Inisialisasi nilai bobot Vij_new
Wjk_new = (beta_W .* W) ./ W_k;                         % Inisialisasi nilai bobot Wjk_new

V = Vij_new;                                            % Bobot Nguyen-Widrow
W = Wjk_new;                                            % Bobot Nguyen-Widrow

% Inisialisasi nilai lama
W_jk_lama = 0;
W_0k_lama = 0;
V_ij_lama = 0;
V_0j_lama = 0;

% Algoritma BPNN
% Penentuan nilai parameter iterasi
iterasi = 2000;
iter = 0;
iter_val = 0;
Ep_stop = 1;
alpha = 0.5;
miu = 0.5;

dummy = 1;
MSE_val = zeros(iterasi, 1);

% Training data
while Ep_stop == 1 && iter < iterasi
    iter = iter + 1;
    for a = 3:length(X_train)
        % ==================== PROSES FEEDFORWARD ====================
        % Menghitung semua sinyal input dengan bobotnya
        feature_train = [X_train(a, 1) X_train(a, 2) X_train(a, 3) X_train(a, 4) X_train(a, 5)];
        z_inj = V_0j + feature_train * V;
        
        % Proses aktivasi menggunakan fungsi bipolar sigmoid
        for j = 1:p
            zj(1,j) = -1 + 2 / (1 + exp(-z_inj(1, j)));
        end
        
        % Menghitung semua sinyal input dengan bobotnya
        y_ink = W_0k + zj * W;
        % Proses aktivasi menggunakan fungsi bipolar sigmoid
        for r = 1:m
            y_k(a,r) = -1 + 2 / (1 + exp(-y_ink(1, r)));
        end
        
        for s = 1:m
            train_pred(a, s) = y_k(a);
        end
        
        % Menghitung nilai error
        E(1, a) = abs(y_train(a,:) - y_k(a));
        
        % Menghitung nilai total error kuadratik (MSE) data validasi
        E_mse(1, a) = (y_train(a, :) - y_k(a)).^2;
        
        
        if a < length(X_train)
            X_train(a + 1, 4) = train_pred(a, 1);
        end
        
        if a < length(X_train)
            X_train(a + 1, 5) = train_pred(a - 1, 1);
        end
        
        % ==================== PROSES BACKPROPAGATION ====================
        % Menghitung informasi error
        do_k = (y_train(a,:) - y_k(a)) .* ((1 + y_k(a)) * (1 - y_k(a)) / 2);
        % Menghitung besarnya koreksi bobot unit output
        w_jk = alpha * zj' * do_k + miu * W_jk_lama;
        % Menghitung besarnya koreksi bias output
        w_0k = alpha * do_k + miu * W_0k_lama;
        
        W_jk_lama = w_jk;
        W_0k_lama = w_0k;
        
        % Menghitung semua koreksi error
        do_inj  = do_k * W';
        % Menghitung nilai aktivasi koreksi error
        do_j    = do_inj .* ((1 + zj) .* (1 - zj) / 2);
        % Menghtiung koreksi bobot unit hidden
        v_ij    = alpha * X_train(a,:)' * do_j + miu * V_ij_lama;
        % Menghitung koreksi error bias unit hidden
        v_0j    = alpha * do_j + miu * V_0j_lama;
        
        V_ij_lama = v_ij;
        V_0j_lama = v_0j;
        
        % Menng-update bobot dan bias untuk setiap unit output
        W = W + w_jk;
        W_0k = W_0k + w_0k;
        
        % Mengupdate bobot dan bias untuk setiap unit hidden
        V = V + v_ij;
        V_0j = V_0j + v_0j;
    end
    
    % Melakukan validasi data dengan bobot yang diperoleh dari training
    % setiap 10 epoch
    if mod(iter, 50) == 0 
        while Ep_stop == 1 && iter_val <= iter
            iter_val = iter_val + 1;
            for a = 3:length(X_val)
                % ==================== PROSES FEEDFORWARD ====================
                % Menghitung semua sinyal input dengan bobotnya
                feature_val = [X_val(a, 1) X_val(a, 2) X_val(a, 3) X_val(a, 4) X_val(a, 5)];
                z_inj_val = V_0j + feature_val * V;

                % Proses aktivasi menggunakan fungsi bipolar sigmoid
                for j = 1:p
                    zj_val(1,j) = -1 + 2 / (1 + exp(-z_inj_val(1, j)));
                end

                % Menghitung semua sinyal input dengan bobotnya
                y_ink_val = W_0k + zj_val * W;
                % Proses aktivasi menggunakan fungsi bipolar sigmoid
                for r = 1:m
                    y_k_val(a,r) = -1 + 2 / (1 + exp(-y_ink_val(1, r)));
                end

                for s = 1:m
                    val_pred(a, s) = y_k_val(a);
                end

                % Menghitung nilai error
                E_val(1, a) = abs(y_val(a,:) - y_k_val(a));

                % Menghitung nilai total error kuadratik (MSE) data validasi
                E_mse_val(1, a) = (y_val(a, :) - y_k_val(a)).^2;

                if a < length(X_val)
                    X_val(a + 1, 4) = val_pred(a, 1);
                end

                if a < length(X_val)
                    X_val(a + 1, 5) = val_pred(a - 1, 1);
                end
            end
            % Menghitung nilai error validasi pada tiap epoch
            Ep_val(1, iter) = sum(E_val) / length(X_val);
            
            % Menghitung nilai total error kuadratik (MSE) pada tiap epoch
            MSE_val(iter_val, 1) = sum(E_mse_val) / length(X_val);
        end
    end
    
    % Menghitung nilai error training pada tiap epoch
    Ep(1, iter) = sum(E) / length(X_train);
    
    % Menghitung nilai total error kuadratik (MSE) pada tiap epoch
    MSE_train(iter, 1) = sum(E_mse) / length(X_train);
    
    if MSE_val(iter, 1) > MSE_train(iter, 1) && dummy == 1
        V_best = V;
        W_best = W;
        V_0j_best = V_0j;
        W_0k_best = W_0k;
        dummy = 0;
    end
    
    acc_p(iter, 1) = 1 - MSE_train(iter, 1);
end

MSE_train_total = sum(MSE_train) / length(MSE_train);
MSE_val_total = sum(MSE_val) / length(MSE_val);


% Melakukan testing
E_test = zeros(length(X_test), 1);
for a = 3:length(X_test)
    % ==================== PROSES FEEDFORWARD ====================
    % Menghitung semua sinyal input dengan bobotnya
    feature_test = [X_test(a, 1) X_test(a, 2) X_test(a, 3) X_test(a, 4) X_test(a, 5)];
    z_inj_test = V_0j_best + feature_test * V_best;

    % Proses aktivasi menggunakan fungsi bipolar sigmoid
    for j = 1:p
        zj_test(1,j) = -1 + 2 / (1 + exp(-z_inj_test(1, j)));
    end

    % Menghitung semua sinyal input dengan bobotnya
    y_ink_test = W_0k_best + zj_test * W_best;
    % Proses aktivasi menggunakan fungsi bipolar sigmoid
    for r = 1:m
        y_k_test(a,r) = -1 + 2 / (1 + exp(-y_ink_test(1, r)));
    end

    for s = 1:m
        test_pred(a, s) = y_k_test(a);
    end

    % Menghitung nilai error
    E_test(a, 1) = abs(y_test(a,:) - y_k_test(a));

    % Menghitung nilai total error kuadratik (MSE) data validasi
    E_mse_test(a, 1) = (y_test(a, :) - y_k_test(a)).^2;

    if a < length(X_test)
        X_test(a + 1, 4) = test_pred(a, 1);
    end

    if a < length(X_test)
        X_test(a + 1, 5) = test_pred(a - 1, 1);
    end
end

Ep_test = sum(E_test) / length(X_test);

MSE_test = sum(E_mse_test) / length(X_test);


figure;
plot(MSE_train, 'r-', 'Linewidth', 2);
hold on
plot(MSE_val, 'b-', 'LineWidth', 2);
hold off
ylabel('Total error kuadratik'); xlabel('Epoch');
legend("MSE data training", "MSE data validasi");

figure;
plot(Ep, 'r-', 'Linewidth', 2);
legend("Error data training", "Error data validasi");

x = 1:5000;
y = 1:2000;
z = 1:5000;

figure;
scatter(x, y_train, 'Linewidth', 1.5);
hold on
scatter(x, y_k, 'Linewidth', 1.5);
legend("Output Plant Training", "Output ANN Training");
hold off

figure;
scatter(y, y_val, 'Linewidth', 1.5);
hold on
scatter(y, y_k_val, 'Linewidth', 1.5);
legend("Output Plant Validasi", "Output ANN Validasi");
hold off

figure;
scatter(z, y_test, 'Linewidth', 1.5);
hold on
scatter(z, y_k_test, 'Linewidth', 1.5);
legend("Output Plant Testing", "Output ANN Testing");
hold off


disp("Error final Data Training = " + (Ep(end)));
disp("Error final Data Validasi = " + (Ep_val(end)));
disp("Error final Data Testing = " + (Ep_test));

disp("MSE final Data Training = " + (MSE_train(end)));
disp("MSE final Data Validasi = " + (MSE_val(end)));

disp("MSE total Data Training = " + (MSE_train_total));
disp("MSE total Data Validasi = " + (MSE_val_total));
disp("MSE total Data Testing = " + (MSE_test));