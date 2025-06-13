function [Gbest_val,fit_record,diversity_position]=CEC_LDPSO(fhd,MaxFES,N,D,Xmax,Xmin,elite_max,elite_min,Jump,alpha,varargin)
rand('state',sum(100*clock));
Vmax = Xmax/2;
Vmin = -Vmax;
fit_record=zeros(1,MaxFES);
diversity_position=zeros(1,ceil(MaxFES/N));
fitness=zeros(1,N);
X = rand(N,D).*(Xmax-Xmin)+Xmin;
V = rand(N,D).*(Vmax-Vmin)+Vmin;
for i =1:N
    fitness(i) = feval(fhd,X(i,:)',varargin{:});
end
fitcount=N;
[best_fitness,best_index] = min(fitness);
Gbest = X(best_index,:);
fit_record(1:N)=best_fitness;
Pbest = X;
Pbest_val = fitness;
Gbest_val = best_fitness;
iter=0;
Record=zeros(1,N);
while fitcount<MaxFES
    w=0.9-0.5*fitcount/MaxFES;
    c1=2.5-2*fitcount/MaxFES;
    c2=0.5+2*fitcount/MaxFES;
    elite_num=ceil((elite_max-(elite_max-elite_min)*(fitcount/MaxFES))*N);
    iter=iter+1;
    [~,Index]=sort(Pbest_val);
    diversity_position(iter)=sum(sqrt(sum((X-mean(X)).^2,2)))/N;
    Elite_ID=Index(1:elite_num);
    Ordinary_ID=setdiff(cumsum(ones(1,N)),Elite_ID);
    SE=cover_based_exemplar(Pbest(Elite_ID,:),Pbest_val(Elite_ID),fitness(Elite_ID));
    for i = 1:N
        if Record(i)<=Jump
            if ismember(i,Elite_ID)
                k=find(Elite_ID==i);
                XE1=diag(Pbest(Elite_ID(k+ceil(rand(1,D)*(numel(Elite_ID)-k))),cumsum(ones(1,D))))';
                XE=diag(Pbest(Elite_ID(ceil(rand(1,D)*k)),cumsum(ones(1,D))))';
                V(i,:)=w.*V(i,:)+c1*rand(1,D).*(XE1-X(i,:))+c2*rand(1,D).*(XE-X(i,:));
            else
                XE1=diag(Pbest(Elite_ID(ceil(rand(1,D)*numel(Elite_ID))),cumsum(ones(1,D))))';
                V(i,:)=alpha*rand(1,D)*w.*V(i,:)+c1*rand(1,D).*(XE1-X(i,:))+c2*rand(1,D).*(SE-X(i,:));
            end
            V(i,V(i,:)>Vmax) = Vmax;
            V(i,V(i,:)<Vmin) = Vmin;
            X(i,:) = X(i,:) + V(i,:);
            X(i,X(i,:)>Xmax) = Xmax;
            X(i,X(i,:)<Xmin) = Xmin;
            fitness(i) = feval(fhd,X(i,:)',varargin{:});
            fitcount=fitcount+1;
            fit_record(fitcount)=min(fit_record(fitcount-1),fitness(i));
            Record(i)=Record(i)*(fitness(i) >= Pbest_val(i))+(fitness(i) >= Pbest_val(i));
            Pbest(i,:)= (fitness(i) < Pbest_val(i))* X(i,:)+(fitness(i) >= Pbest_val(i))*Pbest(i,:);
            Pbest_val(i)=(fitness(i) < Pbest_val(i))*fitness(i)+(fitness(i) >= Pbest_val(i))*Pbest_val(i);
            Gbest= (fitness(i) < Gbest_val)* X(i,:)+(fitness(i) >= Gbest_val)*Gbest;
            Gbest_val=(fitness(i) < Gbest_val)*fitness(i)+(fitness(i) >= Gbest_val)*Gbest_val;
        else
            Xa=Pbest(i,:);
            XO=diag(Pbest(Ordinary_ID(ceil(rand(1,D)*numel(Ordinary_ID))),cumsum(ones(1,D))))';
            XE=diag(Pbest(Elite_ID(ceil(rand(1,D)*numel(Elite_ID))),cumsum(ones(1,D))))';
            Xa=Xa+rand(1,D).*(XO-Xa)+rand(1,D).*(XE-Xa);
            Xa(Xa>Xmax) = Xmax;   Xa(Xa<Xmin) = Xmin;
            Xa_fit=feval(fhd,Xa',varargin{:});
            fitcount=fitcount+1;
            fit_record(fitcount)=min(fit_record(fitcount-1),Xa_fit);
            Record(i)=0;
            X(i,:)=Xa;
            fitness(i)=Xa_fit;
            Pbest(i,:)= (fitness(i) < Pbest_val(i))* X(i,:)+(fitness(i) >= Pbest_val(i))*Pbest(i,:);
            Pbest_val(i)=(fitness(i) < Pbest_val(i))*fitness(i)+(fitness(i) >= Pbest_val(i))*Pbest_val(i);
            Gbest= (fitness(i) < Gbest_val)* X(i,:)+(fitness(i) >= Gbest_val)*Gbest;
            Gbest_val=(fitness(i) < Gbest_val)*fitness(i)+(fitness(i) >= Gbest_val)*Gbest_val;
        end
    end
end
end

function social_exemplar=cover_based_exemplar(Pbest,Pbest_val,fitness)
[Number_of_Member,D]=size(Pbest);
[~,index]=sort(Pbest_val);
weight=(1./index);
weight=(weight/sum(weight));
social_exemplar=sum(weight'.*Pbest((index),:));
end
