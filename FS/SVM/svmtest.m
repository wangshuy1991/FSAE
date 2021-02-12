training_label(1:514)=1;
training_label(515:end)=0;
testing_label(1:514)=1;
testing_label(515:end)=0;

model = svmtrain(training_label, training_feature, '-c 1 -t 0 -g 0.1');

[predict_label, accuracy, dec_values] = svmpredict(testing_label, testing_feature, model);
