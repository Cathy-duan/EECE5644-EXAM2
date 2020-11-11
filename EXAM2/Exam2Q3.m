close all, clear all,
N = 1000; % N samples
N2 = 100;
N3 = 10;
initializations1=0;
initializations2=0;
initializations3=0;
delta = 0.5; % tolerance for EM stopping criterion
regWeight = 1e-3; % regularization parameter for covariance estimates
B=10;
% Generate samples from a 4-component GMM
alpha_true = [0.2,0.25,0.25,0.3];
mu_true = [-10 1 2 1;1 1 1 2];
Sigma_true(:,:,1) = [3 2;1 20];
Sigma_true(:,:,2) = [7 1;2 2];
Sigma_true(:,:,3) = [4 2;1 16];
Sigma_true(:,:,4) = [5 1;2 5];
[d,~] = size(mu_true); % determine dimensionality of samples and number of GMM components
%% 1000 samples
for m = 1:100
    x1000 = randGMM(N,alpha_true,mu_true,Sigma_true);
    
    logLike = zeros(B,6);
    for b=1:B
        Dtrain = x1000(:,randi([1,1000],1,500))+mvnrnd([0; 0],1e-5.*eye(2),500)';
        Dvalidate = x1000(:,randi([1,1000],1,500));
        %clear temp;
        % Initialize the GMM to randomly selected samples
        for M=1:6
             Converged = 0; % Not converged at the beginning
             while ~Converged
            alpha = ones(1,M)/M;
            shuffledIndices = randperm(size(Dtrain,2));
            mu = Dtrain(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates
            [~,assignedCentroidLabels] = min(pdist2(mu',Dtrain'),[],1); % assign each sample to the nearest mean
            for a = 1:M % use sample covariances of initial assignments as initial covariance estimates
                Sigma(:,:,a) = cov(Dtrain(:,find(assignedCentroidLabels==a))') + regWeight*eye(d,d);
            end
            %t = 0; %displayProgress(t,x,alpha,mu,Sigma);
            initializations1=initializations1+1;
            
            number=0;
            while (~Converged & number<101) 
            
               %for i=200
                for l = 1:M
                    temp(l,:) = repmat(alpha(l),1,size(Dtrain,2)).*evalGaussian(Dtrain,mu(:,l),Sigma(:,:,l));
                end
                plgivenx = temp./sum(temp,1);
                clear temp;
                alphaNew = mean(plgivenx,2);
                w = plgivenx./repmat(sum(plgivenx,2),1,size(Dtrain,2));
                muNew = Dtrain*w';
                for l = 1:M
                    v = Dtrain-repmat(muNew(:,l),1,size(Dtrain,2));
                    u = repmat(w(l,:),d,1).*v;
                    SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
                end
                Dalpha = sum(abs(alphaNew-alpha'));
                Dmu = sum(sum(abs(muNew-mu)));
                DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
                Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
                %if Converged
                 %   break
                %end
                alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
                %t = t+1;
                %displayProgress(t,Dtrain,alpha,mu,Sigma);
    
                number=number+1;
            end
                
             end
            %clf(1);
            logLike(b,M) = sum(log(evalGMM(Dvalidate,alpha,mu,Sigma)));%%%%%%%%%%%
        end
        %AveragelogLike(1,M) = mean(logLike(:,M)); 
        
    end
    AveragelogLike = mean(logLike, 2);
end

[~, ind]= max(max(AveragelogLike));
fprintf('WinnerCounts1 = %d', ind);
fprintf('initializations1 = %d', initializations1);
%% 100 samples
for m = 1:100
    x100 = randGMM(N2,alpha_true,mu_true,Sigma_true);
    
    logLike = zeros(B,6);
    for b=1:B
        Dtrain = x100(:,randi([1,100],1,500))+mvnrnd([0; 0],1e-5.*eye(2),500)';;
        Dvalidate = x100(:,randi([1,100],1,500));
        clear temp;
        % Initialize the GMM to randomly selected samples
        for M=1:6
            Converged = 0; % Not converged at the beginning
            while ~Converged
            alpha = ones(1,M)/M;
            shuffledIndices = randperm(size(Dtrain,2));
            mu = Dtrain(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates
            [~,assignedCentroidLabels] = min(pdist2(mu',Dtrain'),[],1); % assign each sample to the nearest mean
            for a = 1:M % use sample covariances of initial assignments as initial covariance estimates
                Sigma(:,:,a) = cov(Dtrain(:,find(assignedCentroidLabels==a))') + regWeight*eye(d,d);
            end
            %t = 0; %displayProgress(t,x,alpha,mu,Sigma);

            initializations2=initializations2+1;
            number=0;
           
                %for i=1:200
            while (~Converged & number<101) 
                for l = 1:M
                    temp(l,:) = repmat(alpha(l),1,size(Dtrain,2)).*evalGaussian(Dtrain,mu(:,l),Sigma(:,:,l));
                end
                plgivenx = temp./sum(temp,1);
                clear temp;
                alphaNew = mean(plgivenx,2);
                w = plgivenx./repmat(sum(plgivenx,2),1,size(Dtrain,2));
                muNew = Dtrain*w';
                for l = 1:M
                    v = Dtrain-repmat(muNew(:,l),1,size(Dtrain,2));
                    u = repmat(w(l,:),d,1).*v;
                    SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
                end
                Dalpha = sum(abs(alphaNew-alpha'));
                Dmu = sum(sum(abs(muNew-mu)));
                DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
                Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
                %if Converged
                  %  break
                %end
                
                alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
                %t = t+1;
                %displayProgress(t,Dtrain,alpha,mu,Sigma);
                number=number+1;
                end
            end
             %clf(1);
            logLike(b,M) = sum(log(evalGMM(Dvalidate,alpha,mu,Sigma)));%%%%%%%%%%%
        end
       %AveragelogLike(1,M) = mean(logLike(:,M)); 
       
    end
    AveragelogLike = mean(logLike, 2);
end

[~, ind]= max(max(AveragelogLike));
fprintf('WinnerCounts2 = %d', ind);
fprintf('initializations2 = %d', initializations2);
%% 10 samples
for m = 1:100
    x10 = randGMM(N3,alpha_true,mu_true,Sigma_true);
    
    logLike = zeros(B,6);
    for b=1:B
        Dtrain = x10(:,randi([1,10],1,500))+mvnrnd([0; 0],1e-5.*eye(2),500)';;
        Dvalidate = x10(:,randi([1,10],1,500));
        clear temp;
        
        % Initialize the GMM to randomly selected samples
        for M=1:6
            Converged = 0; % Not converged at the beginning
            while ~Converged
            alpha = ones(1,M)/M;
            
            shuffledIndices = randperm(size(Dtrain,2));
            mu = Dtrain(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates
            [~,assignedCentroidLabels] = min(pdist2(mu',Dtrain'),[],1); % assign each sample to the nearest mean
            for a = 1:M % use sample covariances of initial assignments as initial covariance estimates
                Sigma(:,:,a) = cov(Dtrain(:,find(assignedCentroidLabels==a))') + regWeight*eye(d,d);
            end
           
            %t = 0; %displayProgress(t,x,alpha,mu,Sigma);

            initializations3=initializations3+1;
            number=0;
            %for i=1:200
            while (~Converged & number<101) 
                for l = 1:M
                    temp(l,:) = repmat(alpha(l),1,size(Dtrain,2)).*evalGaussian(Dtrain,mu(:,l),Sigma(:,:,l));
                end
                plgivenx = temp./sum(temp,1);
                %clear temp;
                alphaNew = mean(plgivenx,2);
                w = plgivenx./repmat(sum(plgivenx,2),1,size(Dtrain,2));
                muNew = Dtrain*w';
                for l = 1:M
                    v = Dtrain-repmat(muNew(:,l),1,size(Dtrain,2));
                    u = repmat(w(l,:),d,1).*v;
                    SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
                end
                Dalpha = sum(abs(alphaNew-alpha'));
                Dmu = sum(sum(abs(muNew-mu)));
                DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
                Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
                %if Converged
                 %   break
                %end
                alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
                %t = t+1;
                %displayProgress(t,Dtrain,alpha,mu,Sigma);
                number=number+1;
               end
            end
             %clf(1);
            logLike(b,M) = sum(log(evalGMM(Dvalidate,alpha,mu,Sigma)));%%%%%%%%%%%
        end
        %AveragelogLike(1,M) = mean(logLike(:,M)); 
        
    end
    AveragelogLike = mean(logLike, 2);
end

[~, ind]= max(max(AveragelogLike));
fprintf('WinnerCounts3 = %d', ind);
fprintf('initializations3 = %d', initializations3);
%% functions
function x = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % dimensionality of samples
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
for m = 1:length(alpha)
    ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1));
    x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
end
end

function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end

function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end

function displayProgress(t,x,alpha,mu,Sigma)
figure(1),
if size(x,1)==2
    subplot(1,2,1), cla,
    plot(x(1,:),x(2,:),'b.'); 
    xlabel('x_1'), ylabel('x_2'), title('Data and Estimated GMM Contours'),
    axis equal, hold on;
    rangex1 = [min(x(1,:)),max(x(1,:))];
    rangex2 = [min(x(2,:)),max(x(2,:))];
    [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2);
    contour(x1Grid,x2Grid,zGMM); axis equal, 
    subplot(1,2,2), 
end
logLikelihood = sum(log(evalGMM(x,alpha,mu,Sigma)));%%%%%%%%%%%
plot(t,logLikelihood,'b.'); hold on,
xlabel('Iteration Index'), ylabel('Log-Likelihood of Data'),
drawnow; pause(0.1),
end

function [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2)
x1Grid = linspace(floor(rangex1(1)),ceil(rangex1(2)),101);
x2Grid = linspace(floor(rangex2(1)),ceil(rangex2(2)),91);
[h,v] = meshgrid(x1Grid,x2Grid);
GMM = evalGMM([h(:)';v(:)'],alpha, mu, Sigma);
zGMM = reshape(GMM,91,101);
%figure(1), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
end