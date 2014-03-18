addpath ..\libsvm-3.17\windows\

% Read the particle data without scaling, and train an svm using default
% settings.  Then test the learned model.
trDataFn = '..\databases\libsvm_guide\train.1.txt';
[labels, trData] = libsvmread(trDataFn);
model_nscd = svmtrain(labels, trData);
tsDataFn = '..\databases\libsvm_guide\test.1.txt';
[tsLabels, tsData] = libsvmread(tsDataFn);
[pLabels, acy, dv] = svmpredict(tsLabels, tsData, model_nscd);
fprintf('Prediction accuracy: %2.2f %%\n', acy(1, 1));

% Read the particle data, which have been scaled using the tools
% svm-scale.exe under the directory of the <libsvm_root>\windows
% Learn another model and test it.
trdsFn = '..\databases\libsvm_guide\train_scale.1';
[labels, trDataS] = libsvmread(trdsFn);
model_scd = svmtrain(labels, trDataS);
tsdsFn = '..\databases\libsvm_guide\test_scale.1';
[tsLabels, tsDataS] = libsvmread(tsdsFn);
[pLabelsScd, acyScd, dvScd] = svmpredict(tsLabels, tsDataS, model_scd);
fprintf('Prediction accuracy with scaled data: %2.2f %%\n', acyScd(1, 1));


% Read the scaled data and perform grid search to get the model parameters
% for C and Gamma and then train a model and test it
expR = -3:2:13;
expC = -3:2:15;
C = 2.^expC;
R = 2.^expR;
cvAcy = zeros(size(C, 2), size(R, 2));
disp('Automatic cross validation for model selection.');
for cIdx = 1:size(C, 2)
    for rIdx = 1:size(R, 2)
        optStr = sprintf('-v 5 -c %.2f -g %.2f', C(1, cIdx), R(1, rIdx));
        cvAcy(cIdx, rIdx) = svmtrain(labels, trDataS, optStr);
        fprintf('Cross validation accurarcy: %.2f %%, param:%s\n', cvAcy(cIdx, rIdx), optStr);
    end
end
contour(cvAcy); xlabel('Gamma'); ylabel('C'); 
title('Cross validation accuracy'); colorbar;

% Picke the maximum and start the new parameter selection
[x, y]= find(cvAcy == max(max(cvAcy)));

CFine = (C(1, x)-3):0.1:(C(1, x)+3);
RFine = (R(1, y)-3):0.1:(R(1, y)+3);
disp('Automatic cross validation for model selection.');
cvAcyFine = zeros(size(CFine, 2), size(RFine, 2));
for cIdx = 1:size(CFine, 2)
    for rIdx = 1:size(RFine, 2)
        optStr = sprintf('-v 5 -c %.2f -g %.2f', CFine(1, cIdx), RFine(1, rIdx));
        cvAcyFine(cIdx, rIdx) = svmtrain(labels, trDataS, optStr);
        fprintf('Cross validation accurarcy: %.2f %%, param:%s\n', cvAcyFine(cIdx, rIdx), optStr);
    end
end
figure; contour(cvAcyFine); xlabel('Gamma'); ylabel('C'); 
title('Cross validation accuracy with fine parameter selection'); colorbar;

[x, y]= find(cvAcyFine == max(max(cvAcyFine)));
COptimal = CFine(1, x);
ROptimal = RFine(1, y);
optStr = sprintf('-c %.2f -g %.2f', COptimal, ROptimal);
modelSelect = svmtrain(labels, trDataS, optStr);
[pLabelsScdMdSel, acyScdMdSel, dvScdMdSel] = svmpredict(tsLabels, tsDataS, modelSelect);
fprintf('Prediction accuracy with scaled data after model selection: %2.2f %%\n', acyScdMdSel(1, 1));


        
        
        