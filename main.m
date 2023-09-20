function [F, T, obj]=main(X,numClusters,r1Temp,r2Temp)
%X:each row is a sample
knn0=15;%number of neihorhood
maxIter=10;
numViews=length(X);
numSamples=size(X{1},1);
H=cell(numViews,1);
R=cell(numViews,1);

lambda = ones(numViews,1)/(numViews);
HR_sum=zeros(numSamples,numClusters);
for v=1:numViews
    W{v}=constructW_PKN(X{v}',knn0);
    S{v}=(W{v}+W{v}')./2;
    [H{v},~,~]=spectral_embedding(S{v},numClusters);
    R{v}=eye(numClusters);
    HR_sum = HR_sum + lambda(v)*H{v}*R{v};
end
%init F
[Uf,~,Vf]=svd(HR_sum,'econ');
F=Uf*Vf';


%L=eye(numSamples) - 1/numSamples*ones(numSmaples);
 L=eye(numSamples)-1/numSamples*ones(numSamples);
K_sum=zeros(numSamples,numSamples);
K=cell(numViews,1);
gamma = ones(numViews,1)/(numViews);
for v=1:numViews
   Dist=pdist2(X{v},X{v}, 'squaredeuclidean');
   K_base=exp(-Dist/(2*max(Dist(:))));
   K{v} = L * K_base * L;
   K_sum=K_sum+gamma(v)*K{v};
end
% H
opt.disp = 0;
[T, ~] = eigs(K_sum, numClusters, 'la', opt);
flag=1;
iter=0;
while flag
    iter=iter+1;

    
    %% update gama
       f1 = zeros(1,numViews);
       for v=1:numViews
        f1(v) = trace(T' * K{v} * T) + eps; 
       end
    gamma = f1./norm(f1,2);
    
     %% update lambda
       f2 = zeros(1,numViews);
       for v=1:numViews
        f2(v) = trace(F' * H{v} * R{v}) + eps; 
       end
    lambda = f2./norm(f2,2);
    
     %% update K_sum
     K_sum=zeros(numSamples,numSamples);
     for v=1:numViews
        K_sum=K_sum+gamma(v)*K{v};
     end
       
     %% update R^v
     for v=1:numViews
        [Ur,~,Vr]=svd(lambda(v)*H{v}'*F);
        R{v}=Ur*Vr';
     end
     
      %% update F
     HR_sum=zeros(numSamples,numClusters);
     for v=1:numViews
        HR_sum=HR_sum+lambda(v)*H{v}*R{v};
     end
     [Uf,~,Vf]=svd(r1Temp*T + r2Temp*HR_sum,'econ');
     F=Uf*Vf';
     
     %% update T
         opts = [];  opts.info = 1;
    opts.gtol = 1e-5;
     X=T;
     A=-K_sum;
     G=-r1Temp*F;
     [T, ~] = FOForth(X,G,@fun,opts,A,G);
      %% update calculate obj
      obj(iter)=trace(T'*K_sum*T + r1Temp*F'*T + r2Temp*F'*HR_sum);
      
    if (iter>2) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-6 || iter>maxIter)
        flag =0;
    end
%     if  (iter>maxIter)
%         flag =0;
%     end
end
%  % plot
%  figure();
%   x=1:length(obj);
%   plot(x,obj,'->r','linewidth', 1.2);
% %ylim([1.4e5 2.3e5])
%    xlabel('Iteration', 'FontName', 'YaHei Consolas Hybrid', 'FontSize', 15);
%    ylabel(' Objective value', 'FontName', 'YaHei Consolas Hybrid', 'FontSize', 15);
%  set(gca,'FontName', 'YaHei Consolas Hybrid','FontSize', 15,'FontWeight','bold');
%   set(gca,'xcolor',[0 0 0],'FontWeight','bold');
%   set(gca,'ycolor',[0 0 0],'FontWeight','bold');
%   PicPath = ['Caltech101','.jpg'];
%   print('-djpeg','-r600', PicPath);
    function [funX, F] = fun(X,A,G)
        F = A * X + G;
        funX = sum(sum(X.* F));
    end
end