clc;clear
load('CI_Project_data.mat');
DATA = zeros(30,384,160);
for i = 1:120
    DATA(:,:,i) = TrainData(:,:,i);
end
for i = 1:40
    DATA(:,:,120+i) = TestData(:,:,i);
end
clean_data = mapstd(DATA);
TrainData = clean_data(:,:,1:120);
TestData = clean_data(:,:,121:160);
data_length = size(TrainData,3);
Fs = 256;
[t,~,~] = pca(TrainData(:,:,1));

TrainData_1 = zeros(size(t,2),size(t,1),data_length);
for i = 1:data_length
    [t,~,~] = pca(TrainData(:,:,i));
    TrainData_1(:,:,i) = t';
end
Normalized_Train_Features =feature_extractor1(TrainData_1,data_length,Fs,size(t,2),size(t,1));
Normalized_Train_Features = mapstd(Normalized_Train_Features);
[Normalized_Train_Features,xPS] = mapminmax(Normalized_Train_Features ) ;

%%
F = 14;
w = CSP(TrainData_1,F,TrainLabel);
X_CSP = zeros(data_length, 2*F);
for j = 1:data_length
    s = TrainData_1(:, :, j);
    Y = w'*s;
    X_CSP(j, :) = log(var(Y, [], 2)');
%     Normalized_Train_Features(1742+j,:) = 

end
tt = size(Normalized_Train_Features,1);
Normalized_Train_Features(tt+1:tt+2*F,:) = mapstd(X_CSP');
%%
%emd features
[emd_basis] = emd_base(TrainData_1,TrainLabel,1);
emd_basis_1 = emd_basis(1,:);
emd_basis_2 = emd_basis(2,:);
[emd_basis] = emd_base(TrainData_1,TrainLabel,2);
emd_basis_3 = emd_basis(1,:);
emd_basis_4 = emd_basis(2,:);
[emd_basis] = emd_base(TrainData_1,TrainLabel,3);
emd_basis_5 = emd_basis(1,:);
emd_basis_6 = emd_basis(2,:);

Normalized_Train_Features = feature_emd(Normalized_Train_Features,emd_basis_1,TrainData_1);
Normalized_Train_Features = feature_emd(Normalized_Train_Features,emd_basis_2,TrainData_1);
Normalized_Train_Features = feature_emd(Normalized_Train_Features,emd_basis_3,TrainData_1);
Normalized_Train_Features = feature_emd(Normalized_Train_Features,emd_basis_4,TrainData_1);
Normalized_Train_Features = feature_emd(Normalized_Train_Features,emd_basis_5,TrainData_1);
Normalized_Train_Features = feature_emd(Normalized_Train_Features,emd_basis_6,TrainData_1);

%%
data_length_1 = size(TestData,3);
[t,~,~] = pca(TestData(:,:,1));

TestData_1 = zeros(size(t,2),size(t,1),data_length_1);
for i = 1:data_length_1
    [t,~,~] = pca(TestData(:,:,i));
    TestData_1(:,:,i) = t';
end

Normalized_Test_Features =feature_extractor1(TestData_1,data_length_1,Fs,size(t,2),size(t,1));
Normalized_Test_Features = mapstd(Normalized_Test_Features);
Normalized_Test_Features = mapminmax('apply',Normalized_Test_Features,xPS) ;

X_CSP = zeros(data_length_1, 2*F);
for j = 1:data_length_1
    s = TestData_1(:, :, j);
    Y = w'*s;
    X_CSP(j, :) = var(Y, [], 2)';
%     Normalized_Train_Features(1742+j,:) = 

end
tt = size(Normalized_Test_Features,1);
Normalized_Test_Features(tt+1:tt+2*F,:) = mapstd(X_CSP');

Normalized_Test_Features = feature_emd(Normalized_Test_Features,emd_basis_1,TestData_1);
Normalized_Test_Features = feature_emd(Normalized_Test_Features,emd_basis_2,TestData_1);
Normalized_Test_Features = feature_emd(Normalized_Test_Features,emd_basis_3,TestData_1);
Normalized_Test_Features = feature_emd(Normalized_Test_Features,emd_basis_4,TestData_1);
Normalized_Test_Features = feature_emd(Normalized_Test_Features,emd_basis_5,TestData_1);
Normalized_Test_Features = feature_emd(Normalized_Test_Features,emd_basis_6,TestData_1);

%% Fischer Feature Selection
Test_data_len = 40;
Fs = 256; %Hz
channel_len = size(t,2);
Features_len = size(Normalized_Train_Features,1);
Train_data_len = 120;
right_Mov_indices = find(TrainLabel==1) ;
left_Mov_indices = find(TrainLabel==2) ;
J = zeros(Features_len,1);
for i = 1:Features_len
    u1 = mean(Normalized_Train_Features(i,right_Mov_indices)) ;
    S1 = (Normalized_Train_Features(i,right_Mov_indices)-u1)*(Normalized_Train_Features(i,right_Mov_indices)-u1)' ; 
    u2 = mean(Normalized_Train_Features(i,left_Mov_indices)) ;
    S2 = (Normalized_Train_Features(i,left_Mov_indices)-u2)*(Normalized_Train_Features(i,left_Mov_indices)-u2)' ; 
    Sw = S1/length(right_Mov_indices)+S2/length(left_Mov_indices) ;
    u0 = mean(Normalized_Train_Features(i,:)) ; 
    Sb = (u1-u0)^2 + (u2-u0)^2 ;
    J(i) = Sb/Sw ;
end
%% Genetic Algorithm
lenFeatures = 45;
Num_of_features = 130;
[mxx, ind] = maxk(J, Num_of_features);
ind(size(ind,1)+1:size(ind,1)+2*F) = 1770:-1:1770-2*F+1;
ChosenFeatures = Normalized_Train_Features(ind, :);
Num_of_features = Num_of_features + 28;
%%
N = 20; % Generation Size
p = 1/lenFeatures; % mutation prob
max_gen = 2e2;
%initialize the population
P = zeros(N, lenFeatures);
P2 = zeros(N, lenFeatures);
for i = 1:N
    P(i,:) = randperm(Num_of_features, lenFeatures);
    P2(i,:) = randperm(Num_of_features, lenFeatures);
end
P = sort(P,2);
P2 = sort(P2,2);
P_best = P(N,:);
P_best_test = P_best;
gen = 1;
fit_best_gen = 0;
fitbest = zeros(max_gen,1);
%%
fit = zeros(N, 1);
hidden_size = 25;
while gen < max_gen
   for i = 1:N 
       fit(i) = mlp_fitness(ChosenFeatures(P(i,:), :),  TrainLabel-1,hidden_size);
   end 
   [tmp, idx] = max(fit);
   if(fit_best_gen < tmp)
       P_best = P(idx,:);
       fit_best_gen = tmp;
       fitbest(gen,1) = fit_best_gen;
   end
   if(fit_best_gen == tmp)
       P_best_test = P(idx,:);
   end
   % roulette-wheel selection
   couples = roulette_wheel(fit - min(fit) - 0.01/(1+gen),N-4);
   % Mutation
   P = mutate_natural(P, p,Num_of_features);
   % Cross-over
   P2(1:N-4, :) = crossover_1Point_correction(P, couples);
   % Correction
   P2(N,:) = P_best;
   P2(N-3:N-1,:) = P(roulette_wheel2(fit,3),:);
   P = P2; 
   P = correction(P, Num_of_features); 
   disp(fit_best_gen);%disp(P_best);disp(fit);
   gen = gen + 1; 
%    if(gen < 10)
%        p = 0.02;
%    elseif(gen < 20)
%        p = 0.01;
%    elseif(gen < 30)
%        p = 0.005;
%    end
end
%%
%best index we founded 
s = [ 498,1200,1814,1058,65,1806,444,999,1231,113,402,1246,1059,255,189,1408,538,1775,925,1230,1101,179,1822,896,188,1237,1239,1787,489,409,234, 1701,51,461,34,146, 1051,138,719,1754,1752,1751,1750,1749 ,1748];
hidden_size = 25;
%% MLP - Training GA
% clc

% ind2 = P_best

acc = zeros(100,1);
for j = 1 :100
    total_err = 0 ; 
% using 5-fold cross-validation
    for k=1:5
        train_indices = [1:(k-1)*24,k*24+1:120] ;
        valid_indices = (k-1)*24+1:k*24 ;

%         TrainX = ChosenFeatures(ind2,train_indices) ;
        TrainX = Normalized_Train_Features(s,train_indices) ;
        ValX = Normalized_Train_Features(s,valid_indices) ;
%         ValX = ChosenFeatures(ind2,valid_indices) ;
        TrainY = TrainLabel(:,train_indices)-1 ;
        ValY = TrainLabel(:,valid_indices)-1 ;
        net = patternnet(hidden_size);
        net.trainParam.max_fail = 10;
        net.trainParam.showWindow = false;
        net = train(net,TrainX,TrainY);
        predict_y = net(ValX);
        predictedLabel = predict_y > 0.5;
        err = sum(abs(predictedLabel - ValY));
        total_err = total_err + err;

    end
    accuracy_mlp_ga = 1 - total_err / 120;
    acc(j) = accuracy_mlp_ga;
%     disp(accuracy_mlp_ga);
end
hist(acc)
%% RBF - Training GA
% ind2 = P_best

spread = 19.5;
Maxnumber = 19;
err = 0 ; 
% using 5-fold cross-validation
% acc_1 = zeros(100,1);
% for j = 1 :100
%     total_err_1 = 0 ; 
for k = 1:5
    train_indices = [1:(k-1)*24,k*24+1:120] ;
    valid_indices = (k-1)*24+1:k*24 ;
%     TrainX = ChosenFeatures(ind2,train_indices) ;
%     ValX = ChosenFeatures(ind2,valid_indices) ;
    TrainX = Normalized_Train_Features(s,train_indices) ;
    ValX = Normalized_Train_Features(s,valid_indices) ;
    TrainY = TrainLabel(:,train_indices)-1 ;
    ValY = TrainLabel(:,valid_indices)-1 ;

    net = newrb(TrainX,TrainY,10^-6,spread,Maxnumber,1) ;

    predict_y = net(ValX);
    Thr = 0.5 ;
    predict_y = predict_y > Thr ;
    err = err + sum(abs(predict_y - ValY)) ;
%     total_err_1 = total_err_1 + err;
end
accuracy_rbf_ga = 1 - err / 120;
%     acc_1(j) = accuracy_rbf_ga;
% end
disp(accuracy_rbf_ga);
% hist(acc_1)
%% Tests on MLP
ChosenFeatures_Test = Normalized_Test_Features(ind, :);
% Classification
N = hidden_size ; % Best parameter found in training step

% TrainX = ChosenFeatures(ind2, :) ;

TrainX = Normalized_Train_Features(s,:) ;
%         ValX = Normalized_Train_Features(s,valid_indices) ;
TrainY = TrainLabel-1 ;
% TestX = ChosenFeatures_Test(ind2, :) ;

TestX = Normalized_Test_Features(s,:) ;
%         ValX = Normalized_Train_Features(s,valid_indices) ;
cumm = zeros(1,40);
for i = 1:100
    net = patternnet(N);
    net.trainParam.max_fail = 10;
    net = train(net,TrainX,TrainY);
    predict_y = net(TestX);
    Test_MLP_GA_Result = predict_y > 0.5;
    cumm = cumm + Test_MLP_GA_Result;
end
% cumm
[~,b] = sort(cumm);
cumm(b(1:20)) = 1;
cumm(b(21:40)) = 2;
save('MLP_predicted_Genetic_algorythm_reported','cumm')
%%
%% Tests on RBF
spread = 19.5;
Maxnumber = 19;
% TrainX = Normalized_Train_Features(ind, :) ;
TrainX = Normalized_Train_Features(s,:) ;
TrainY = TrainLabel-1 ;
% TestX = Normalized_Test_Features(ind, :) ;
TestX = Normalized_Test_Features(s,:) ;
net = newrb(TrainX,TrainY,10^-5,spread,Maxnumber, 2) ;
predict_y = net(TestX);
% predict_y
% Thr = 0.5 ;
% Test_RBF_GA_Result = predict_y > Thr ;
[~,b] = sort(predict_y);
predict_y(b(1:20)) = 1;
predict_y(b(21:40)) = 2;


save('RBF_predicted_Genetic_algorythm_reported','predict_y')

%%
%phase 1 index we found
%35 feature
%30 neuron

s = [498 1200 525  65 1806 492 1245 541 113 1059 429 1120 1408 1101 229 258 1247 1822 1006 1800 1062 1024 850 1008 705 544 138 1010 1795 1766 1762 1759 1756 1755 1748];
hidden_size = 30;

%% Tests on MLP
ChosenFeatures_Test = Normalized_Test_Features(ind, :);
% Classification
N = hidden_size ; % Best parameter found in training step

% TrainX = ChosenFeatures(ind2, :) ;

TrainX = Normalized_Train_Features(s,:) ;
%         ValX = Normalized_Train_Features(s,valid_indices) ;
TrainY = TrainLabel-1 ;
% TestX = ChosenFeatures_Test(ind2, :) ;

TestX = Normalized_Test_Features(s,:) ;
%         ValX = Normalized_Train_Features(s,valid_indices) ;
cumm = zeros(1,40);
for i = 1:100
    net = patternnet(N);
    net.trainParam.max_fail = 10;
    net = train(net,TrainX,TrainY);
    predict_y = net(TestX);
    Test_MLP_GA_Result = predict_y > 0.5;
    cumm = cumm + Test_MLP_GA_Result;
end
% cumm
[~,b] = sort(cumm);
cumm(b(1:20)) = 1;
cumm(b(21:40)) = 2;
save('MLP_predicted_Genetic_algorythm_reported_phase_1','cumm')

%% Tests on RBF
spread = 19.5;
Maxnumber = 19;
% TrainX = Normalized_Train_Features(ind, :) ;
TrainX = Normalized_Train_Features(s,:) ;
TrainY = TrainLabel-1 ;
% TestX = Normalized_Test_Features(ind, :) ;
TestX = Normalized_Test_Features(s,:) ;
net = newrb(TrainX,TrainY,10^-5,spread,Maxnumber, 2) ;
predict_y = net(TestX);
% predict_y
% Thr = 0.5 ;
% Test_RBF_GA_Result = predict_y > Thr ;
[~,b] = sort(predict_y);
predict_y(b(1:20)) = 1;
predict_y(b(21:40)) = 2;


save('RBF_predicted_Genetic_algorythm_reported_phase_1','predict_y')
%%
%CSP
function w = CSP(DATA,F,TrainLabel)
    p_index = find(TrainLabel == 2);
    n_index = find(TrainLabel == 1);

    TrainData_zero_mean = zeros(size(DATA));
    DATA_len = size(DATA,3);
    for j = 1:DATA_len
        TrainData_zero_mean(:, :, j) = DATA(:, :, j) - mean(DATA(:, :, j)); 
    end
    channel_len = size(DATA,1);
    C0 = zeros(channel_len, channel_len);
    C1 = zeros(channel_len, channel_len);
    for j = p_index
        s = DATA(:, :, j);
        C1 = C1 + s*s'/trace(s*s');
    end
    for j = n_index
        s = DATA(:, :, j);
        C0 = C0 + s*s'/trace(s*s');
    end
    [V, D] = eig(C1, C0);
    [~, index] = sort(diag(D), "descend");
    V = V(:, index);
%     F = 14;
    W_CSP = [V(:, 1:F), V(:, end-F+1:end)];
    w = W_CSP;
end
%%
function emd_base_1 = emd_base(DATA,TrainLabel,basis_num)
    index_1 = find(TrainLabel == 1);
    index_0 = find(TrainLabel == 2);
    x1 = size(DATA,1);
    x2 = size(DATA,2);
    test = zeros(x1,x2);

    for i = index_0
        counter = 1;
        X_1_temp = DATA(:,:,i);
%         disp(size(X_1_temp))
    %     [F1,~,~]=COM2R(X_1_temp,30);
    %     Z1 = pinv(F1)*X_1_temp;
    %     test(counter,1:end) = (Z1(1,1:end))/2;
        for j=1:x1
%             disp(j)
            [t,~,~] = emd(X_1_temp(j,:),'Display',0);
    %         disp(size(t(:,1)'))
            test(counter,1:end) = test(counter,1:end) + t(:,basis_num)';
            counter =counter + 1;
        end
    %     test(1,1:end) = test(1,1:end)/30;
        
    end
    test_signal_0 = sum(test,1);
    test_signal_0 = (test_signal_0-mean(test_signal_0))/norm(test_signal_0);
% offset = max(abs(test(:)));
% feq = 256;
% disp_eeg(test,offset,feq);

    test = zeros(x1,384);

    for i = index_1
        counter = 1;
        X_1_temp = DATA(1:end,1:end,i);
    %     [F1,~,~]=COM2R(X_1_temp,30);
    %     Z1 = pinv(F1)*X_1_temp;
    %     test(counter,1:end) = (Z1(1,1:end))/2;
        for j=1:x1
            
            [t,~,~] = emd(X_1_temp(j,:),'Display',0);
    %         disp(size(t(:,1)'))
            test(counter,1:end) = test(counter,1:end) + t(:,basis_num)';
            counter =counter + 1;
        end
    %     test(1,1:end) = test(1,1:end)/30;
        
    end
    test_signal_1 = sum(test,1);
    test_signal_1 = (test_signal_1-mean(test_signal_1))/norm(test_signal_1);
% offset = max(abs(test(:)));
% feq = 256;
% disp_eeg(test,offset,feq);
    emd_base_1 = [test_signal_0;test_signal_1];
end
function Feature = feature_emd(Feature,emdbasis,Train_data)
    temp = size(Train_data,1);
    temp_2 = size(Train_data,3);
    corr_matrix_0 = zeros(temp,temp_2);
    for i = 1 :temp_2
        for j = 1:temp
            [t,~,~] = emd(Train_data(j,:,i),'Display',0);
            
            corr_matrix_0(j,i) = sum(t(:,1)'.*emdbasis);    
        end
    end
    tt = size(Feature,1);
    Feature(tt+1:tt+temp,:) = mapstd(corr_matrix_0);
end
