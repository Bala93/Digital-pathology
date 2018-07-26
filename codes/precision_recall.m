function [result] = precision_recall(groundtruth_json_path,predicted_json_path)

ground_truth_text = fileread(groundtruth_json_path);
predicted_text    = fileread(predicted_json_path);

ground_truth_json = jsondecode(ground_truth_text);
predicted_json = jsondecode(predicted_text);

file_names = fieldnames(ground_truth_json);
numImages = numel(file_names);
predicted(numImages) = struct('Boxes',[],'Scores',[]);
groundtruth(numImages) = struct('Boxes',[]);
confidence_score = 0.5;

for ii = 1 : numImages
    % Predicted 
    info_pred_temp = predicted_json.(file_names{ii});
    info_pred_temp.boxes(:,3) = info_pred_temp.boxes(:,3) - info_pred_temp.boxes(:,1);
    info_pred_temp.boxes(:,4) = info_pred_temp.boxes(:,4) - info_pred_temp.boxes(:,2);
    indices  = info_pred_temp.scores > confidence_score ;
    predicted(ii).Boxes = info_pred_temp.boxes(indices,:);
    predicted(ii).Scores = info_pred_temp.scores(indices);

    
    % Ground truth update
    info_gt_temp   = ground_truth_json.(file_names{ii});
    info_gt_temp(:,3) = info_gt_temp(:,3) - info_gt_temp(:,1);
    info_gt_temp(:,4) = info_gt_temp(:,4) - info_gt_temp(:,2);
    groundtruth(ii).Boxes = info_gt_temp;
    
end

predicted = struct2table(predicted);
groundtruth = struct2table(groundtruth);


%%
% Evaluate the results against the ground truth data. Get the precision 
% statistics.
[ap,recall,precision] = evaluateDetectionPrecision(predicted,groundtruth,0.5);

scores = [];

for kk = 1:numel(predicted.Scores)
    scores = [scores; predicted.Scores{kk}];
end

confidence = sort(scores,'descend');
precision = precision(1:end-1);
recall    = recall(2:end);
f1_score = 2 * ((precision .* recall) ./(precision + recall));
result = table(precision,recall,f1_score,confidence);

%best_f1_score = max(f1_score);%
%ind = find(f1_score == best_f1_score);

% fprintf('Recall = %f\n',recall(ind));
% fprintf('Precision = %f\n',precision(ind));
% fprintf('F1-score = %f\n',best_f1_score);

%%
% Plot the precision-recall curve.
figure
plot(recall,precision)
grid on
title(sprintf('Average Precision = %.2f',ap))

end