function [Z,Clus,ACC,NMI,AR,purity,Fscore,precision,recall]=train(dataset,k_number,lambda_e_list,lambda_r_list)
load(dataset);
X=fea';
gt=gnd;
N=size(gt,1);
A=cell(1,2);
A{1}=double(W);
iter = 100;
M=1;
nmix = zeros(M,1);
accr = zeros(M,1);
pur = zeros(M,1);
Fscore = zeros(M,1);
Prec = zeros(M,1);
Rec = zeros(M,1);
ar = zeros(M,1);
err=zeros(iter,1);
lambda_c=1;
r=1.1;
w=1/2;
options = [];
options.NeighborMode = 'KNN';
options.k =k_number;          
options.WeightMode = 'HeatKernel';
A{2}=X;
A{2}=constructW(A{2}',options);
DCol=cell(1,2);
D=cell(1,2);
L=cell(1,2);
for i=1:2
    DCol{i}=full(sum(A{i},2));
    D{i}=spdiags(DCol{i},0,N,N);
    L{i}=D{i}-A{i};
end
lambda_e=lambda_e_list;
lambda_r=lambda_r_list;
tol=10^-5;
clusters = size(unique(gt),1);
mu = 10^-3;
rho=1.1;
maxmu=10^6;
C = zeros(N,N);
K = zeros(N,N);
Y2 = zeros(N,N);
X1=X;
R=zeros(N,N);
E=zeros(size(X1));
Y1=zeros(size(X1));
xx=0;
s=1;
W=[];
while s<=iter
    %% caculate C
    fprintf('s=%f\n',s);
    AC=mu*eye(N)+mu*(X1')*X1;
    BC=2*(w^r)*L{1}+2*((1-w)^r)*L{2};
    CC=mu*(X1')*(Y1/mu+X1-X1*R-E)-Y2+mu*K;
    AC=full(AC);
    BC=full(BC);
    CC=full(CC);
    C=sylvester(AC,BC,CC);
    %% caculate K
    [U,S,V] = svd(C + Y2./mu);
    a = diag(S)-lambda_c/mu;
    a(a<0)=0;
    T = diag(a);
    K = U*T*V';
    %% caculate R
    R=(2*lambda_r*eye(N)+mu*(X1')*X1)\(mu*(X1')*(X1-X1*C-E+Y1/mu));
    %% caculate E
    temp=X1-X1*C-X1*R+Y1/mu;
    E= solve_l1l2(temp,lambda_e/mu);
    %% caculate w
    w=(trace(C*L{2}*C'))^(1/(r-1))/((trace(C*L{1}*C'))^(1/(r-1))+(trace(C*L{2}*C'))^(1/(r-1)));
    W(s)=w;
    %% check if converge
    P=norm(X1-X1*C-X1*R-E,'fro')^2;
    eq2=X1-X1*C-X1*R-E;
    eq1=C-K;
    err(s)=abs(xx-P);
    if P<tol
        break;
    else
        Y1=Y1+mu*eq2;
        Y2=Y2+mu*eq1;
        mu=min(maxmu,mu*rho);
    end
    s = s+1;
end
%% clustering
Z = (abs(C)+abs(C'))/2+(abs(R)+abs(R'))/2;
Clus = SpectralClustering(Z,clusters);
%% evaluation
result = ClusteringMeasure(gt,Clus);
accr(M) = result(1);
pur(M) = result(3);
nmix(M) = result(2);
ar(M) = result(4);
Fscore(M)=result(5);
Prec(M)=result(6);
Rec(M)=result(7);
ACC=mean(accr);
NMI=mean(nmix);
AR=mean(ar);
purity=mean(pur);
Fscore=mean(Fscore);
precision=mean(Prec);
recall=mean(Rec);