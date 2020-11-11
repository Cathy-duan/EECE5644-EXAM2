%% ================== Generate D10 train samples ================== %%
clear all; close all; clc;

n = 2;      % number of feature dimensions
N1 = 10;   % number of iid samples

% parallel distributions
mu(:,1) = [-2;0]; Sigma(:,:,1) = [1 -0.9;-0.9 2];
mu(:,2) = [2;0]; Sigma(:,:,2) = [2 0.9;0.9 1]; 

% Class priors for class 0 and 1 respectively
p = [0.9,0.1]; 

% Generating true class labels
label1 = (rand(1,N1) >= p(1))';
Nc1 = [sum(label1==0),sum(label1==1)];

% Draw samples from each class pdf
x1 = zeros(N1,n); 
for L = 0:1
    x1(label1==L,:) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc1(L+1));
end
%save('Exam2.mat','n','N1','label','Nc','x');
%% ================== Generate D100 train samples ================== %%
n = 2;      % number of feature dimensions
N2 = 100;   % number of iid samples

% parallel distributions
mu(:,1) = [-2;0]; Sigma(:,:,1) = [1 -0.9;-0.9 2];
mu(:,2) = [2;0]; Sigma(:,:,2) = [2 0.9;0.9 1]; 

% Class priors for class 0 and 1 respectively
p = [0.9,0.1]; 

% Generating true class labels
label2 = (rand(1,N2) >= p(1))';
Nc2 = [sum(label2==0),sum(label2==1)];

% Draw samples from each class pdf
x2 = zeros(N2,n); 
for L = 0:1
    x2(label2==L,:) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc2(L+1));
end
%save('Exam2.mat','n','N2','label','Nc','x');
%% ================== Generate D1000 train samples ================== %%

n = 2;      % number of feature dimensions
N3 = 1000;   % number of iid samples

% parallel distributions
mu(:,1) = [-2;0]; Sigma(:,:,1) = [1 -0.9;-0.9 2];
mu(:,2) = [2;0]; Sigma(:,:,2) = [2 0.9;0.9 1]; 

% Class priors for class 0 and 1 respectively
p = [0.9,0.1]; 

% Generating true class labels
label3 = (rand(1,N3) >= p(1))';
Nc3 = [sum(label3==0),sum(label3==1)];

% Draw samples from each class pdf
x3 = zeros(N3,n); 
for L = 0:1
    x3(label3==L,:) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc3(L+1));
end
%save('Exam2.mat','n','N3','label','Nc','x');
%% ================== Generate D10K Validate samples ================== %%
n = 2;      % number of feature dimensions
N4 = 10000;   % number of iid samples

% parallel distributions
mu(:,1) = [-2;0]; Sigma(:,:,1) = [1 -0.9;-0.9 2];
mu(:,2) = [2;0]; Sigma(:,:,2) = [2 0.9;0.9 1]; 

% Class priors for class 0 and 1 respectively
p = [0.9,0.1]; 

% Generating true class labels
label4 = (rand(1,N4) >= p(1))';
Nc4 = [sum(label4==0),sum(label4==1)];

% Draw samples from each class pdf
x4 = zeros(N4,n); 

for L = 0:1
    x4(label4==L,:) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc4(L+1));
    
end
save('Exam2.mat','n','N1','N2','N3','N4','label1','label2','label3','label4','Nc1','Nc2','Nc3','Nc4','x1','x2','x3','x4');
%% ================== Draw ROC on Validate samples ================== %%
x5=x4.';
label5 = label4.';
discriminantScore = log(evalGaussian(x5,mu(:,2),Sigma(:,:,2))./evalGaussian(x5,mu(:,1),Sigma(:,:,1)));
y = log(sort(discriminantScore(discriminantScore >=0)));

%Find midpoints of y to use as threshold values
mid_y= [y(1)-100 y(1:end-1) + diff(y)./2 y(length(y))+100];

%Make decision for every threshold and calculate error values
for i=1:length(mid_y)
    decision=(discriminantScore >=mid_y(i));
    pFA(i) = sum(decision==1 & label5==0)/Nc4(1); %False alarm Prob.
    pCD(i) = sum(decision==1 & label5==1)/Nc4(2); %Correct detection Prob.
    pE(i) = pFA(i)*p(1) +(1-pCD(i))*p(2);       %Total error Prob.
end

%Find minimum error and corresponding threshold
[min_error, min_index] = min(pE);
min_decision = (discriminantScore >=mid_y(min_index));
min_FA = pFA(min_index); min_CD = pCD(min_index);

%Plot ROC curve with minumum error point labeled
figure(1); plot(pFA, pCD, '-', min_FA,min_CD,'o');
title('Minimum Expected Risk ROC Curve'); legend('ROC Curve','Calculated Min Error');
xlabel('P_{False Alarm}'); ylabel('P_{Correct Detection}');

lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(x5,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x5,mu(:,1),Sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));

ind00 = find(decision==0 & label5==0); p00 = length(ind00)/Nc4(1); % probability of true negative
ind10 = find(decision==1 & label5==0); p10 = length(ind10)/Nc4(1); % probability of false positive
ind01 = find(decision==0 & label5==1); p01 = length(ind01)/Nc4(2); % probability of false negative
ind11 = find(decision==1 & label5==1); p11 = length(ind11)/Nc4(2); % probability of true positive
p_error = [p10,p01]*Nc4'/N4;
disp(p_error);
% plot correct and incorrect decisions
figure(2), % class 0 circle, class 1 +, correct green, incorrect red
plot(x5(1,ind00),x5(2,ind00),'og'); hold on,
plot(x5(1,ind10),x5(2,ind10),'or'); hold on,
plot(x5(1,ind01),x5(2,ind01),'+r'); hold on,
plot(x5(1,ind11),x5(2,ind11),'+g'); hold on,
axis equal,
% Draw the decision boundary
horizontalGrid = linspace(floor(min(x5(1,:))),ceil(max(x5(1,:))),101);
verticalGrid = linspace(floor(min(x5(2,:))),ceil(max(x5(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
discriminantScoreGridValues = log(evalGaussian([h(:)';v(:)'],mu(:,2),Sigma(:,:,2)))-log(evalGaussian([h(:)';v(:)'],mu(:,1),Sigma(:,:,1))) - log(gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
figure(2), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
% including the contour at level 0 which is the decision boundary
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Equilevel contours of the discriminant function' ), 
title('Data and their classifier decisions versus true labels'),
xlabel('x_1'), ylabel('x_2'), 
%% ======================== Logistic Regression for D10 ======================= %%
% Initialize fitting parameters
x1 = [ones(N1, 1) x1];
initial_theta1 = zeros(n+1, 1);
label1=double(label1);

% Compute gradient descent to get theta values
[theta1, cost1] = fminsearch(@(t)(cost_func(t, x1, label1, N1)), initial_theta1);

% Choose points to draw boundary line
plot_x11 = [min(x1(:,2))-2,  max(x1(:,2))+2];                      
plot_x21(2,:) = (-1./theta1(3)).*(theta1(2).*plot_x11 + theta1(1)); % fminsearch

%% ========================= Test Classifier for D10 and D10 ========================== %%
% Coefficients for decision boundary line equation
coeff1(2,:) = polyfit([plot_x11(1), plot_x11(2)], [plot_x21(2,1), plot_x21(2,2)], 1); %fminsearch
% Decide based on which side of the line each point is on

    if coeff1(2,1) >= 0
        decision1(:,2) = (coeff1(2,1).*x1(:,2) + coeff1(2,2)) > x1(:,3);
    else
        decision1(:,2) = (coeff1(2,1).*x1(:,2) + coeff1(2,2)) < x1(:,3);
    end

error1 = plot_test_data(decision1(:,2), label1, Nc1, p, 3, x1(:,2:3), plot_x11, plot_x21(2,:));
title('Test Data Classification (D10 train on D10)');
%fprintf('Total error (D10 train on D10): %.2f%%\n',error1);
%% ======================== Logistic Regression for D100 ======================= %%
% Initialize fitting parameters
x2 = [ones(N2, 1) x2];
initial_theta2 = zeros(n+1, 1);
label2=double(label2);

% Compute gradient descent to get theta values
[theta2, cost2] = fminsearch(@(t)(cost_func(t, x2, label2, N2)), initial_theta2);

% Choose points to draw boundary line
plot_x12 = [min(x2(:,2))-2,  max(x2(:,2))+2];                      
plot_x22(2,:) = (-1./theta2(3)).*(theta2(2).*plot_x12 + theta2(1)); % fminsearch

%% ========================= Test Classifier for D100 and D100 ========================== %%
% Coefficients for decision boundary line equation
coeff2(2,:) = polyfit([plot_x12(1), plot_x12(2)], [plot_x22(2,1), plot_x22(2,2)], 1); %fminsearch
% Decide based on which side of the line each point is on

    if coeff2(2,1) >= 0
        decision2(:,2) = (coeff2(2,1).*x2(:,2) + coeff2(2,2)) > x2(:,3);
    else
        decision2(:,2) = (coeff2(2,1).*x2(:,2) + coeff2(2,2)) < x2(:,3);
    end

error2 = plot_test_data(decision2(:,2), label2, Nc2, p, 4, x2(:,2:3), plot_x12, plot_x22(2,:));
title('Test Data Classification (D100 train on D100)');
%fprintf('Total error (D100 train on D100): %.2f%%\n',error2);
%% ======================== Logistic Regression for D1000 ======================= %%
% Initialize fitting parameters
x3 = [ones(N3, 1) x3];
initial_theta3 = zeros(n+1, 1);
label3=double(label3);

% Compute gradient descent to get theta values
[theta3, cost3] = fminsearch(@(t)(cost_func(t, x3, label3, N3)), initial_theta3);

% Choose points to draw boundary line
plot_x13 = [min(x3(:,2))-2,  max(x3(:,2))+2];                      
plot_x23(2,:) = (-1./theta3(3)).*(theta3(2).*plot_x13 + theta3(1)); % fminsearch

%% ========================= Test Classifier for D1000 and D1000 ========================== %%
% Coefficients for decision boundary line equation
coeff3(2,:) = polyfit([plot_x13(1), plot_x13(2)], [plot_x23(2,1), plot_x23(2,2)], 1); %fminsearch
% Decide based on which side of the line each point is on

    if coeff3(2,1) >= 0
        decision3(:,2) = (coeff3(2,1).*x3(:,2) + coeff3(2,2)) > x3(:,3);
    else
        decision3(:,2) = (coeff3(2,1).*x3(:,2) + coeff3(2,2)) < x3(:,3);
    end

error3 = plot_test_data(decision3(:,2), label3, Nc3, p, 5, x3(:,2:3), plot_x13, plot_x23(2,:));
title('Test Data Classification (D1000 train on D1000)');
%fprintf('Total error (D1000 train on D1000): %.2f%%\n',error3);
%% ======================== Logistic Regression for theta1 and D10000 ======================= %%
% Choose points to draw boundary line
plot_x14 = [min(x4(:,1))-2,  max(x4(:,1))+2];                      
plot_x24(2,:) = (-1./theta1(3)).*(theta1(2).*plot_x14 + theta1(1)); % fminsearch
%% ========================= Test Classifier for D10 and D10000 ========================== %%
% Coefficients for decision boundary line equation
coeff4(2,:) = polyfit([plot_x14(1), plot_x14(2)], [plot_x24(2,1), plot_x24(2,2)], 1); %fminsearch
% Decide based on which side of the line each point is on

    if coeff4(2,1) >= 0
        decision4(:,2) = (coeff4(2,1).*x4(:,1) + coeff4(2,2)) > x4(:,2);
    else
        decision4(:,2) = (coeff4(2,1).*x4(:,1) + coeff4(2,2)) < x4(:,2);
    end

error4 = plot_test_data(decision4(:,2), label4, Nc4, p, 6, x4, plot_x14, plot_x24(2,:));
title('Test Data Classification (D10 train on D10K)');
fprintf('Total error (D10 train on D10K): %.2f%%\n',error4);
%% ======================== Logistic Regression for theta2 and D10000 ======================= %%
% Choose points to draw boundary line
plot_x15 = [min(x4(:,1))-2,  max(x4(:,1))+2];                      
plot_x25(2,:) = (-1./theta2(3)).*(theta2(2).*plot_x15 + theta2(1)); % fminsearch
%% ========================= Test Classifier for D100 and D10000 ========================== %%
% Coefficients for decision boundary line equation
coeff5(2,:) = polyfit([plot_x15(1), plot_x15(2)], [plot_x25(2,1), plot_x25(2,2)], 1); %fminsearch
% Decide based on which side of the line each point is on

    if coeff5(2,1) >= 0
        decision5(:,2) = (coeff5(2,1).*x4(:,1) + coeff5(2,2)) > x4(:,2);
    else
        decision5(:,2) = (coeff5(2,1).*x4(:,1) + coeff5(2,2)) < x4(:,2);
    end

error5 = plot_test_data(decision5(:,2), label4, Nc4, p, 7, x4, plot_x15, plot_x25(2,:));
title('Test Data Classification (D100 train on D10K)');
fprintf('Total error (D100 train on D10K): %.2f%%\n',error5);
%% ======================== Logistic Regression for theta3 and D10000 ======================= %%
% Choose points to draw boundary line
plot_x16 = [min(x4(:,1))-2,  max(x4(:,1))+2];                      
plot_x26(2,:) = (-1./theta3(3)).*(theta3(2).*plot_x16 + theta3(1)); % fminsearch

%% ========================= Test Classifier for D1000 and D10000 ========================== %%
% Coefficients for decision boundary line equation
coeff6(2,:) = polyfit([plot_x16(1), plot_x16(2)], [plot_x26(2,1), plot_x26(2,2)], 1); %fminsearch
% Decide based on which side of the line each point is on

    if coeff6(2,1) >= 0
        decision6(:,2) = (coeff6(2,1).*x4(:,1) + coeff6(2,2)) > x4(:,2);
    else
        decision6(:,2) = (coeff6(2,1).*x4(:,1) + coeff6(2,2)) < x4(:,2);
    end

error6 = plot_test_data(decision6(:,2), label4, Nc4, p, 8, x4, plot_x16, plot_x26(2,:));
title('Test Data Classification (D1000 train on D10K)');
fprintf('Total error (D1000 train on D10K): %.2f%%\n',error6);
%% ======================== Logistic-quadratic-function for theta1 and D10000 ======================= %%
% Initialize fitting parameters
x1 = [x1 x1(:,2).^2 x1(:,2).*x1(:,3) x1(:,3).^2];
initial_theta1 = [theta1; zeros(n+1, 1)]; 

% Compute gradient descent to get theta values
%[theta1, cost1] = fminsearch(@(t)(cost_func(t, x1, label1, N1)), initial_theta1);
[theta1, cost1] = gradient_descent(x1,N1,label1,initial_theta1,1,10);

decision2 = theta1(1)+ theta1(2).*x4(:,1)+theta1(3).*x4(:,2)+theta1(4).*(x4(:,1).^2)+theta1(5).*x4(:,1).*x4(:,2)+theta1(6).*(x4(:,2).^2) >0;
ind00 = find(decision2==0 & label4==0); % true negative
ind10 = find(decision2==1 & label4==0); p10 = length(ind10)/Nc4(1); % false positive
ind01 = find(decision2==0 & label4==1); p01 = length(ind01)/Nc4(2); % false negative
ind11 = find(decision2==1 & label4==1); % true positive
error7 = (p10*p(1) + p01*p(2))*100;
    
% Plot decisions and decision boundary
figure(9);
plot(x4(ind00,1),x4(ind00,2),'og'); hold on,
plot(x4(ind10,1),x4(ind10,2),'or'); hold on,
plot(x4(ind01,1),x4(ind01,2),'+r'); hold on,
plot(x4(ind11,1),x4(ind11,2),'+g'); hold on,
func1 = @(x,y) theta1(1)+theta1(2).*x+theta1(3).*y+theta1(4).*(x.^2)+theta1(5).*x.*y+theta1(6).*(y.^2);
fimplicit(func1, [min(x4(:,1))-10, max(x4(:,1))+10, min(x4(:,2))-10, max(x4(:,2))+10]);
title('Test Data Classification (D10 train on D10K)');
fprintf('Total quadratic function error (D10 train on D10K): %.2f%%\n',error7);
legend('Class 0 Correct Decisions','Class 0 Wrong Decisions','Class 1 Wrong Decisions','Class 1 Correct Decisions');
%% ======================== Logistic-quadratic-function for theta2 and D10000 ======================= %%
% Initialize fitting parameters
x2 = [x2 x2(:,2).^2 x2(:,2).*x2(:,3) x2(:,3).^2];
initial_theta2 = [theta2; zeros(n+1, 1)]; 

% Compute gradient descent to get theta values
%[theta2, cost2] = fminsearch(@(t)(cost_func(t, x2, label2, N2)), initial_theta2);
[theta2, cost2] = gradient_descent(x2,N2,label2,initial_theta2,1,100);
decision3 = theta2(1)+ theta2(2).*x4(:,1)+theta2(3).*x4(:,2)+theta2(4).*(x4(:,1).^2)+theta2(5).*x4(:,1).*x4(:,2)+theta2(6).*(x4(:,2).^2) >0;
ind00 = find(decision3==0 & label4==0); % true negative
ind10 = find(decision3==1 & label4==0); p10 = length(ind10)/Nc4(1); % false positive
ind01 = find(decision3==0 & label4==1); p01 = length(ind01)/Nc4(2); % false negative
ind11 = find(decision3==1 & label4==1); % true positive
error8 = (p10*p(1) + p01*p(2))*100;
    
% Plot decisions and decision boundary
figure(10);
plot(x4(ind00,1),x4(ind00,2),'og'); hold on,
plot(x4(ind10,1),x4(ind10,2),'or'); hold on,
plot(x4(ind01,1),x4(ind01,2),'+r'); hold on,
plot(x4(ind11,1),x4(ind11,2),'+g'); hold on,
func2 = @(x,y) theta2(1)+theta2(2).*x+theta2(3).*y+theta2(4).*(x.^2)+theta2(5).*x.*y+theta2(6).*(y.^2);
fimplicit(func2, [min(x4(:,1))-10, max(x4(:,1))+10, min(x4(:,2))-10, max(x4(:,2))+10]);
title('Test Data Classification (D100 train on D10K)');
fprintf('Total quadratic function error (D100 train on D10K): %.2f%%\n',error8);
legend('Class 0 Correct Decisions','Class 0 Wrong Decisions','Class 1 Wrong Decisions','Class 1 Correct Decisions');

%% ======================== Logistic-quadratic-function for theta3 and D10000 ======================= %%
% Initialize fitting parameters
x3 = [x3 x3(:,2).^2 x3(:,2).*x3(:,3) x3(:,3).^2];
initial_theta3 = [theta3; zeros(n+1, 1)]; 

% Compute gradient descent to get theta values
%[theta3, cost3] = fminsearch(@(t)(cost_func(t, x3, label3, N3)), initial_theta3);
[theta3, cost3] = gradient_descent(x3,N3,label3,initial_theta3,1,1000);


decision4 = theta3(1)+ theta3(2).*x4(:,1)+theta3(3).*x4(:,2)+theta3(4).*(x4(:,1).^2)+theta3(5).*x4(:,1).*x4(:,2)+theta3(6).*(x4(:,2).^2) >0;
ind00 = find(decision4==0 & label4==0); % true negative
ind10 = find(decision4==1 & label4==0); p10 = length(ind10)/Nc4(1); % false positive
ind01 = find(decision4==0 & label4==1); p01 = length(ind01)/Nc4(2); % false negative
ind11 = find(decision4==1 & label4==1); % true positive
error9 = (p10*p(1) + p01*p(2))*100;
    
% Plot decisions and decision boundary
figure(11);
plot(x4(ind00,1),x4(ind00,2),'og'); hold on,
plot(x4(ind10,1),x4(ind10,2),'or'); hold on,
plot(x4(ind01,1),x4(ind01,2),'+r'); hold on,
plot(x4(ind11,1),x4(ind11,2),'+g'); hold on,
func3 = @(x,y) theta3(1)+theta3(2).*x+theta3(3).*y+theta3(4).*(x.^2)+theta3(5).*x.*y+theta3(6).*(y.^2);
fimplicit(func3, [min(x4(:,1))-10, max(x4(:,1))+10, min(x4(:,2))-10, max(x4(:,2))+10]);
title('Test Data Classification (D1000 train on D10K)');
fprintf('Total quadratic function error (D1000 train on D10K): %.2f%%\n',error9);
legend('Class 0 Correct Decisions','Class 0 Wrong Decisions','Class 1 Wrong Decisions','Class 1 Correct Decisions');
%% ======================== Logistic-quadratic-function for D10 ======================= %%
decision5 = theta1(1)+ theta1(2).*x1(:,2)+theta1(3).*x1(:,3)+theta1(4).*(x1(:,2).^2)+theta1(5).*x1(:,2).*x1(:,3)+theta1(6).*(x1(:,3).^2) >0;
ind00 = find(decision5==0 & label1==0); % true negative
ind10 = find(decision5==1 & label1==0); p10 = length(ind10)/Nc1(1); % false positive
ind01 = find(decision5==0 & label1==1); p01 = length(ind01)/Nc1(2); % false negative
ind11 = find(decision5==1 & label1==1); % true positive
error10 = (p10*p(1) + p01*p(2))*100;
    
% Plot decisions and decision boundary
figure(12);
plot(x1(ind00,2),x1(ind00,3),'og'); hold on,
plot(x1(ind10,2),x1(ind10,3),'or'); hold on,
plot(x1(ind01,2),x1(ind01,3),'+r'); hold on,
plot(x1(ind11,2),x1(ind11,3),'+g'); hold on,
func = @(x,y) theta1(1)+theta1(2).*x+theta1(3).*y+theta1(4).*(x.^2)+theta1(5).*x.*y+theta1(6).*(y.^2);
fimplicit(func, [min(x1(:,2))-10, max(x1(:,2))+10, min(x1(:,3))-10, max(x1(:,3))+10]);
title('Test Data Classification (D10 train on D10)');
fprintf('Total quadratic function error (D10 train on D10): %.2f%%\n',error10);
legend('Class 0 Correct Decisions','Class 0 Wrong Decisions','Class 1 Wrong Decisions','Class 1 Correct Decisions');
%% ======================== Logistic-quadratic-function for D100 ======================= %%
decision6 = theta2(1)+ theta2(2).*x2(:,2)+theta2(3).*x2(:,3)+theta2(4).*(x2(:,2).^2)+theta2(5).*x2(:,2).*x2(:,3)+theta2(6).*(x2(:,3).^2) >0;
ind00 = find(decision6==0 & label2==0); % true negative
ind10 = find(decision6==1 & label2==0); p10 = length(ind10)/Nc2(1); % false positive
ind01 = find(decision6==0 & label2==1); p01 = length(ind01)/Nc2(2); % false negative
ind11 = find(decision6==1 & label2==1); % true positive
error11 = (p10*p(1) + p01*p(2))*100;
    
% Plot decisions and decision boundary
figure(13);
plot(x2(ind00,2),x2(ind00,3),'og'); hold on,
plot(x2(ind10,2),x2(ind10,3),'or'); hold on,
plot(x2(ind01,2),x2(ind01,3),'+r'); hold on,
plot(x2(ind11,2),x2(ind11,3),'+g'); hold on,
func = @(x,y) theta2(1)+theta2(2).*x+theta2(3).*y+theta2(4).*(x.^2)+theta2(5).*x.*y+theta2(6).*(y.^2);
fimplicit(func, [min(x2(:,2))-10, max(x2(:,2))+10, min(x2(:,3))-10, max(x2(:,3))+10]);
title('Test Data Classification (D100 train on D100)');
fprintf('Total quadratic function error (D100 train on D100): %.2f%%\n',error11);
legend('Class 0 Correct Decisions','Class 0 Wrong Decisions','Class 1 Wrong Decisions','Class 1 Correct Decisions');
%% ======================== Logistic-quadratic-function for D1000 ======================= %%
decision7 = theta3(1)+ theta3(2).*x3(:,2)+theta3(3).*x3(:,3)+theta3(4).*(x3(:,2).^2)+theta3(5).*x3(:,2).*x3(:,3)+theta3(6).*(x3(:,3).^2) >0;
ind00 = find(decision7==0 & label3==0); % true negative
ind10 = find(decision7==1 & label3==0); p10 = length(ind10)/Nc3(1); % false positive
ind01 = find(decision7==0 & label3==1); p01 = length(ind01)/Nc3(2); % false negative
ind11 = find(decision7==1 & label3==1); % true positive
error12 = (p10*p(1) + p01*p(2))*100;
    
% Plot decisions and decision boundary
figure(14);
plot(x3(ind00,2),x3(ind00,3),'og'); hold on,
plot(x3(ind10,2),x3(ind10,3),'or'); hold on,
plot(x3(ind01,2),x3(ind01,3),'+r'); hold on,
plot(x3(ind11,2),x3(ind11,3),'+g'); hold on,
func = @(x,y) theta3(1)+theta3(2).*x+theta3(3).*y+theta3(4).*(x.^2)+theta3(5).*x.*y+theta3(6).*(y.^2);
fimplicit(func, [min(x3(:,2))-10, max(x3(:,2))+10, min(x3(:,3))-10, max(x3(:,3))+10]);
title('Test Data Classification (D1000 train on D1000)');
fprintf('Total quadratic function error (D1000 train on D1000): %.2f%%\n',error12);
legend('Class 0 Correct Decisions','Class 0 Wrong Decisions','Class 1 Wrong Decisions','Class 1 Correct Decisions');

%% ============================ Functions ============================= %%
function [theta, cost] = gradient_descent(x, N, label, theta, alpha, num_iters)
    cost = zeros(num_iters, 1);
    for i = 1:num_iters % while norm(cost_gradient) > threshold
        h = 1 ./ (1 + exp(-x*theta));	% Sigmoid function   
        cost(i) = (-1/N)*((sum(label' * log(h)))+(sum((1-label)' * log(1-h))));
        cost_gradient = (1/N)*(x' * (h - label));
        theta = theta - (alpha.*cost_gradient); % Update theta
    end
end

function cost = cost_func(theta, x, label,N)
    h = 1 ./ (1 + exp(-x*theta));	% Sigmoid function
    cost = (-1/N)*((sum(label' * log(h)))+(sum((1-label)' * log(1-h))));
end
function error = plot_test_data(decision, label, Nc, p, fig, x, plot_x1, plot_x2)
    ind00 = find(decision==0 & label==0); % true negative
    ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % false positive
    ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % false negative
    ind11 = find(decision==1 & label==1); % true positive
    error = (p10*p(1) + p01*p(2))*100;

    % Plot decisions and decision boundary
    figure(fig);
    plot(x(ind00,1),x(ind00,2),'og'); hold on,
    plot(x(ind10,1),x(ind10,2),'or'); hold on,
    plot(x(ind01,1),x(ind01,2),'+r'); hold on,
    plot(x(ind11,1),x(ind11,2),'+g'); hold on,
    plot(plot_x1, plot_x2);
    axis([plot_x1(1), plot_x1(2), min(x(:,2))-2, max(x(:,2))+2])
    legend('Class 0 Correct Decisions','Class 0 Wrong Decisions','Class 1 Wrong Decisions','Class 1 Correct Decisions','Classifier');
end