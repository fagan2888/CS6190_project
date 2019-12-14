# CS6190_project

arima.py: attempt to use arima for predicting converged J-integral values
gaussian_process_Keras.py: test data with scikit-learn Gaussian process model and Keras neural network
nn_all.py: test split where all data is included in one model
nn_BC_param_crack_size.py: run BNN with models for each combination of boudnary condition, crack size, and RVE_MSC parameter
nn_crackSize.py: run BNN with models for each crack size
nn_crackSize_param.py: run BNN with models for each crack size and RVE_MSC parameter combination
nn_param.py: run BNN with models for each RVE_MSC parameter
nn_param_hyperparameterSelection.py: run hyperparameter selection on model for each RVE_MSC parameter
nn_param_smallEps.py: run BNN with modified epsilon sampling where better results are predicted
nonlinear_regression.py: perform nonlinear regressions on raw data
nonlinear_regression_get_c.py: extract exponent fitting coefficient to use on other data sets
nonlinear_regression_set_c.py: use extracted exponent fitting coefficient on new cases
plot_hyper_parameter.py: plot project results
prep_data.py: normalize and subsample data for use in BNN
