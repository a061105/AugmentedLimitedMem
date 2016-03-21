dim = 5;
N = 100;
sigma = 0.1;

% simulate
w = rand(dim,1);
X = rand(N,dim);

noise = randn(N,1)*sigma;

y = X*w + noise;

% write to file

% sim_train (X,y)
f = fopen('sim_train','w');
fprintf(f, '%d %d\n', N, dim);
for i=1:N
	fprintf(f, '%g ', y(i) );
	fprintf(f, '%g ', X(i,:));
	fprintf(f, '\n');
end
fclose(f);

% train to find w_train
w_train = inv(X'*X)*X'*y;

% model_true (w) and model_train (w_train)
f = fopen('model_true', 'w');
f2 = fopen('model_train', 'w');
fprintf(f, '%g\n', w);
fprintf(f2,'%g\n',w_train);
fclose(f);
fclose(f2);
