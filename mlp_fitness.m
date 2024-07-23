function acc = mlp_fitness(ChosenFeatures, TrainLabel,hidden_size)
    
%load('Tf.mat')
    total_err = 0 ; 
    % 5-fold cross-validation
    %for k = 1:5
    for j = 1:5
        k = 5;
        len_train = length(TrainLabel);
        len_train_k = floor(len_train/k);
        t = randperm(len_train);
        train_indices = t(len_train_k+1:end);
        valid_indices = t(1:len_train_k);

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