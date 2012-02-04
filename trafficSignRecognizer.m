function [vector_predicted, error]=trafficSignRecognizer(im,Window_Candidates)
%% Traffic Sign recognizer
load('OVA_SVM_ECOC.mat');
vector_predicted=zeros(1,length(Window_Candidates));
error=zeros(1,length(Window_Candidates));
  if (isempty(Window_Candidates(1).x)) 
      return;
  end
for l=1:1:length(Window_Candidates)
    sub_window = im(int16(Window_Candidates(l).y):int16(Window_Candidates(l).y+Window_Candidates(l).h), int16(Window_Candidates(l).x):int16(Window_Candidates(l).x+Window_Candidates(l).w), :);
    ECOC = eye(14)*2-1;     %One Against All Matrix
    [predicted, errorRate] = ova_svm(OVA_SVM,ECOC,sub_window);
    vector_predicted(l)=predicted;
    error(l)=errorRate;
end
end

function [predicted,errorRate] = ova_svm(OVA_SVM,ECOC, im)
    %Multiclass classifier 
    im = imresize(im, [64 64]);   % We resize the input test image to compare with the model (this is important)
    imgray = double(rgb2gray(im));
    H = HOG_9_12(imgray);
    % now we will test this image for each of the OVA SVM classifiers and obtain it's probabilities
    [num_labels,nClassifiers] = size(ECOC);        
     for k=1:nClassifiers                    %for each of the OVA SVM classifiers
        model_SVM = OVA_SVM(k);             %Load classifier into model_SVM
       [predicted_label, ~, prob_estimates] = svmpredict( 1, H', model_SVM, '-b 1' ); %,'b -1'
        %prediction is a 2 column matrix and nClassifiers rows.
        %1st col is the predicted class (A or B) of each classifier
        %2nd col is the probability of the prediction of each classifier
        prediction(k,1) = predicted_label;
        prediction(k,2) = max(prob_estimates);
    end 
    %DECODIFICATION: Now take each row of ECOC matrix and see the error respect the prediction, the lowest error will be the label prediction
    for k=1:14
        aux = abs(ECOC(k,:)) .* prediction(:,1)' .* prediction(:,2)' ; 
        error(k) = sqrt(sum((ECOC(k,:) - aux) .^ 2));   %euclidean distance using probabilities
    end
    [~,index] = sort (error);  %We sort the errors 
    errorRate=error(index(1));
    predicted = index(1);       %we take the lowest error as predicted label
end