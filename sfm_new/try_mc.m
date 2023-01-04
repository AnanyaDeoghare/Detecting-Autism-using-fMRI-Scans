% Varargin = mean connectivity matrix of autistic and neurotypical
% nargin = number of classes
%Rsum is total connectivity matrix
%Evalsum is stylish X
%EvacSum is B
%W is weight matrix
%S is C_tilda_A and C_tilda_N
%Bu is U
%D is beta_A and beta_N

S = cell(2,1);
R = cell(2,1);
nargin = 2;
%mean_conn = cell(90,90,2);
s1 = load('m_cxa.mat');
s2 = load('m_cxn.mat');

varargin{1} = s1.m_cxa;
varargin{2} = s2.m_cxn;
 
if (nargin ~= 2) 
    disp('Must have 2 classes for CSP!')
end
Rsum=0;
%finding the covariance of each class and composite covariance
for i = 1:nargin
    R{i}=varargin{i}; 
    Rsum=Rsum+R{i};
end

%   Find Eigenvalues and Eigenvectors of RC
%   Sort eigenvalues in descending order
[EVecsum,EValsum] = eig(Rsum);
[EValsum,ind] = sort(diag(EValsum),'descend');
EVecsum = EVecsum(:,ind);

%   Find Whitening  Matrix 
W1 = sqrt(inv(diag(EValsum))) * EVecsum';
for k = 1:nargin
    S{k} = W1 * R{k} * W1'; %      
end

%generalized eigenvectors
[Bu,D] = eig(S{1},S{2});
[D,ind]=sort(diag(D));
Bu=Bu(:,ind);

%Resulting Projection Matrix
result = (Bu'*W1);

