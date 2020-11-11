clear; close all; clc;
%% Input and initializations

%squarede = zeros(100,100); %  squared-error values
gammaArray = logspace(-5,5,100);    % Array of gamma values
%gammaB = -50:1:49;
SigmaV = 0.5; % Standard - 0-mean Gaussian noise

w = [1,0.5,-1,1]'; %Pick truePolynomial parameters wTrue with three roots in [-1,1]
%log10gammaList = linspace(-3,3,5)

for Experiment = 1:100   
    %gamma = 10.^gammaB;
    v = normrnd(0,SigmaV^2,1,10);  %Generate N samples of v~Gaussian(0,sigma^2)
    x = rand(1,10)*2-1;    %Generate N samples of x~Uniform[-1,1]
    
    % Define z vectors for linear models
    zL = [ones(1,size(x,2)); x; x.^2;x.^3];

    % Compute z*z^T for linear models
    for i = 1:10
        zzTL(:,:,i) = zL(:,i)*zL(:,i)';
    end 
    
    % Calculate y: linear in x + additive 0-mean Gaussian noise
    yTruth{1,Experiment} = w(1).*x.^3 + w(2).*x.^2 + w(3).*x + w(4);
    y = yTruth{1,Experiment} + v;  
    
    for indGamma = 1:length(gammaArray)
      gamma = gammaArray(indGamma)
      %Calculate wMAP(gamma) using the analytical solution (you need to put in several lines of code here of course)
      wMAP= (sum(zzTL,3)+SigmaV^2/gamma^2*eye(size(zL,1)))^-1*sum(repmat(y,size(zL,1),1).*zL,2);
      %Calculate parameterErrorL2norm(Experiment,indgamma) = norm(wTrue-wMAP,2)
      %avMsqError(Experiment,indGamma) = length(w)\sum((w - wMAP).^2);
      parameterErrorL2norm(Experiment,indGamma) = length(w)\norm(w-wMAP,2).^2;
    end
% 
%         %the MAP estimate for the parameter vector w
%         p1 = zeros(4,4);
%         p2 = zeros(4,1); 
%         
%         for j = 1:10
%             p1 = z(:,j)*(z(:,j))'+p1;
%             p2 = z(:,j)*y(j,:)+p2;
%         end
%        
%         p3 = gamma(1,a)^(-2)*eye(4);
%         e = (p1/(SigmaV)^2+p3)^(-1);
%         f = p2/(SigmaV)^2;
%         wMAP = e*f;
% 
%         % the squared L2 distance between the true parameter vector and this estimate
%         squarede(a,i) = (w'*w-wMAP'*wMAP)^2;
  end

% % plot 
% plot(gamma,minimum,'+r');  hold on, 
% plot(gamma,twentyfive,'+b');  hold on, 
% plot(gamma,median,'or');  hold on, 
% plot(gamma,seventyfive,'ob');  hold on, 
% plot(gamma,maximum,'xr');  hold on, 
% ax = gca;
% ax.XScale = 'log';
% ax.YScale = 'log';
% legend('minimum','25%','median','75%','maximum'), 
% title('Different Values and Gamma'),
% xlabel('gamma'), ylabel('squared-error values'),

%% Plot results - MAP Ensemble: mean squared error
fig = figure; fig.Position([1,2]) = [50,100];
fig.Position([3 4]) = 1.5*fig.Position([3,4]);
percentileArray = [0,25,50,75,100];

ax = gca; hold on; box on;
prctlMsqError = prctile(parameterErrorL2norm,percentileArray,1);
p=plot(ax,gammaArray,prctlMsqError,'LineWidth',2);
xlabel('gamma'); ylabel('average mean squared error of parameters'); ax.XScale = 'log';
lgnd = legend(ax,p,[num2str(percentileArray'),...
    repmat(' percentile',length(percentileArray),1)]); lgnd.Location = 'southwest';
%% show the minimum, 25%, median, 75%, and maximum values of these squared-error values
prctlMsqError1 = prctlMsqError';
minimum = prctlMsqError1(:,1);
twentyfive = prctlMsqError1(:,2);
median = prctlMsqError1(:,3);
seventyfive = prctlMsqError1(:,4);
maximum = prctlMsqError1(:,5);
disp('The minimum values are:');
disp(minimum);
disp('The 25% values are:');
disp(twentyfive);
disp('The median values are:');
disp(median);
disp('The 75% values are:');
disp(seventyfive);
disp('The maximum values are:');
disp(maximum);