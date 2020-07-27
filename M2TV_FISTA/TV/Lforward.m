function X=Lforward(P)

[m2,n2]=size(P{1});
[m1,n1]=size(P{2});

if (n2~=n1+1)
    error('dimensions are not consistent')
end
if(m1~=m2+1)
    error('dimensions are not consistent')
end

m=m2+1;
n=n2;

X=zeros(m,n);
X(1:m-1,:)=P{1};
X(:,1:n-1)=X(:,1:n-1)+P{2};
X(2:m,:)=X(2:m,:)-P{1};
X(:,2:n)=X(:,2:n)-P{2};


