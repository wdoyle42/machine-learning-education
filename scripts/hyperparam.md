### Hyperparameter Tuning Options

First: What's going to be tuned? What are to be tuned? 

So far: penalty and mixture

Second: What measure of loss will be used to evaluate proposed hyperparameters? 

Continuous: RMSE, MAE, R2

Binary: AUC, Accuracy, Sensitivity, Specificity

Multinomial: AUC Hand-Till, Accuracy 

Third: How will you tune? 

Grid Search: Propose grid of combinations of hyperparameters--> how will you set distances
how many combinations? 

Simulated Annealing: Size of cooling coefficient and the number of iterations overall, and 
iterations in suboptimal space are the key control factors

Bayesian Optimization: No improvement control is key: how many iterations without improvemeent? 

