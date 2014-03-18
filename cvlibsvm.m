function [COptimal, ROptimal] = cvlibsvm(labels, trDataS)


% Read the scaled data and perform grid search to get the model parameters
% for C and Gamma and then train a model and test it
expR = -5:13;
expC = -5:15;
C = 2.^expC;
R = 2.^expR;
cvAcy = zeros(size(C, 2), size(R, 2));
disp('Automatic cross validation for coarse model selection.');
% for cIdx = 1:size(C, 2)
%     for rIdx = 1:size(R, 2)
%         optStr = sprintf('-v 5 -c %.2f -g %.2f', C(1, cIdx), R(1, rIdx));
%         cvAcy(cIdx, rIdx) = svmtrain(labels, trDataS, optStr);
%         fprintf('Cross validation accurarcy: %.2f %%, param:%s\n', cvAcy(cIdx, rIdx), optStr);
%     end
% end
contour(cvAcy); xlabel('Gamma'); ylabel('C'); 
title('Cross validation accuracy'); colorbar;

% Picke the maximum and start the new parameter selection
[x, y]= find(cvAcy == max(max(cvAcy)));
fprintf('Coarse parameter search C=%f, R=%f\n', C(1, x), R(1, y));

% CFine = max((C(1, x)-3), 1e-5):0.1:(C(1, x)+3);
% RFine = max((R(1, y)-3), 1e-5):0.1:(R(1, y)+3);

CFine = 2.000000-1:0.1:2.000000+2;
RFine = 0.031250:0.1:0.031250+2;
disp('Automatic cross validation for fine model selection.');
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
% optStr = sprintf('-c %.2f -g %.2f', COptimal, ROptimal);
% modelSelect = svmtrain(labels, trDataS, optStr);
% [pLabelsScdMdSel, acyScdMdSel, dvScdMdSel] = svmpredict(tsLabels, tsDataS, modelSelect);
% fprintf('Prediction accuracy with scaled data after model seelction: %2.2f %%\n', acyScdMdSel(1, 1));