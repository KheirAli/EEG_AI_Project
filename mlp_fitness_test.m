function acc = mlp_fitness_test(ChosenFeatures, TrainLabel)
    hidden_size = 13;
%load('Tf.mat')
    total_err = 0 ; 
    % 5-fold cross-validation
    %for k = 1:5
    k = 5;
    len_train = length(TrainLabel);
    len_train_k = floor(len_train/k);
    t = randperm(len_train);
    for j = 1:3
        train_indices = [1:(j-1)*len_train_k,j*len_train_k+1:len_train] ;
        valid_indices = (j-1)*len_train_k+1:j*len_train_k ;
        train_indices = t(train_indices);
        valid_indices = t(valid_indices);

        TrainX = ChosenFeatures(:,train_indices) ;
        ValX = ChosenFeatures(:,valid_indices) ;
        TrainY = TrainLabel(:,train_indices) ;
        ValY = TrainLabel(:,valid_indices) ;

        % feedforwardnet, newff, paternnet
        % patternnet(hiddenSizes,trainFcn,performFcn)

        net = patternnet(hidden_size);
        net.trainParam.showWindow = false;
        net.trainParam.max_fail = 10;
        net = train(net,TrainX,TrainY);
%        net.layers{1}.transferFcn = 'logsig';
%        net.layers{3}.transferFcn = 'purelin';

        predict_y = net(ValX);
        predictedLabel = predict_y > 0.5;
        err = sum(abs(predictedLabel - ValY));
        total_err = total_err + err;
        
    end

    acc = 1 - total_err /(j*len_train_k);
    
end