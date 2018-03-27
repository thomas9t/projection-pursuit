# Copyright 2018 Anthony H. Thomas
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib
matplotlib.use('Agg')

import numpy as np
import numpy.linalg as alg
import scipy.interpolate as ipl
import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNetCV

class PPRNode:
    """
    Simple container to hold the parameters associated with a PPR node
    """

    def __init__(self, weights=None, spline=None, mu=0):
        self.weights = weights
        self.spline = spline
        self.mu = 0

class ProjectionPursuitRegressor:
    """
    Estimate Friedman's Projection Pursuit Regression.

    Parameters:

    knots - The number of knots to use in splines. The number of parameters in 
    the spline will be knots + 2. The location of knots are chosen as quantiles
    of the data
    max_terms - The maximum number of terms to fit in the model. The final number
    of terms will be the configuration which minimized test error.
    use_bias - Should each ridge function include a bias term?
    debug - Print detailed convergence information for inner iterations
    l2_reg - Regularization strength for ridge solve
    center - Should data be centered before fit?
    inner_iter_tol - Tolerance to use on the optimization routine for fitting
    weights of a ridge function.
    tolerance - Tolerance to use on adding new ridge function to the model. If
    the reduction in MSE is less than this value then the model has converged.
    spline_method: How to fit the spline function - 'least_squares' uses quantiles
    as the knots, 'smoother' uses the standard cubic smoothing spline. Use of
    'smoother' for prediction tasks is discouraged as it is severly prone to
    overfitting.
    """

    def __init__(self, knots=2, max_terms=15,
                 train_split=0.80, tolerance=1.0, center=True,
                 verbose=True, inner_iter_tol=1e-5,
                 use_bias=True, stabilizer=3.2, categorical_cols=None,
                 l1_ratio=[.25, .5, .75, .95, 1], use_backfitting=True,
                 spline_method='least_squares', smoothing_param=None):

        if knots < 1:
            raise StandardError('Must have at least one knot')

        self._stabilizer = stabilizer
        self._smoother = self._smoothing_spline if spline_method == 'smoother' \
             else self._least_squares_spline
        self._smoothing_param = smoothing_param
        self._knots = knots
        self._use_bias = use_bias
        self._train_pct = train_split
        self._outer_tolerance = tolerance
        self._debug_backfitting = False
        self._debug_inner_iterations = False
        self._mu = None
        self._sigma = None
        self._params = []
        self._cv_gof = []
        self._train_gof = []
        self._l1_ratio = l1_ratio
        self._is_fit = False
        self._inner_iter_tol = inner_iter_tol
        self._max_terms = max_terms
        self._should_center = center
        self._should_backfit = use_backfitting
        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None
        self._X = None
        self._l1_ratio_cv = None
        self._pos_variance = None
        self._categorical_cols = [] if categorical_cols is None else categorical_cols
        self._is_verbose = verbose

    def fit(self, X, y, X_test=None, y_test=None):
        """
        Fit the model. The model will add terms until either `max_terms` is 
        reached, or reduction in MSE between two iterations drops below `tolerance`.

        The final model returned is the one which yielded minimum error on the
        validation set.

        Parameters:
        X - Independent Variables - may not contain missing values
        y - Dependent variable - may not contain missing values.
        """

        y = y.ravel()
        self._X = X
        if X_test is None:
            split = self._train_test_split(X, y, self._train_pct)
            self._X_train, self._y_train = split[0:2]
            self._X_test, self._y_test = split[2:]
        else:
            self._X_train = X
            self._X_test = X_test
            self._y_train = y
            self._y_test = y_test

        if self._should_center:
            self._X_train = self._center_indepvars(self._X_train)
            self._X_test = self._center_new_X(self._X_test)
        self._X = self._X[:,self._pos_variance]

        if self._use_bias:
            self._add_biases()

        num_terms = 0
        mse = np.inf
        self._all_zeros = False
        r = self._y_train  # We begin the fit using the raw dependent variable
        while num_terms < self._max_terms:
            if self._is_verbose:
                print 'Fitting term: {} => INITAL GOF: {}'.format(num_terms, mse)

            self._fit(self._X_train, r)
            if num_terms > 0 and self._should_backfit:
                self._backfit()

            mse_new_train = self._compute_gof(self._X_train, self._y_train)
            mse_new_test = self._compute_gof(self._X_test, self._y_test)
            self._cv_gof.append(mse_new_test)
            self._train_gof.append(mse_new_train)

            if self._is_verbose:
                msg = 'Term: {} fitted => TRAINING GOF: {} => TESTING GOF: {}'
                print msg.format(num_terms, mse_new_train, mse_new_test)

            # Check to see if the drop in MSE was below user defined threshold
            if np.abs(mse_new_train - mse) < self._outer_tolerance:
                if self._is_verbose:
                    print 'Converged!'
                break

            # if not add a new direction and continue
            r = self._y_train - self._predict(self._X_train)  # we fit the next direction using this direction's residuals
            mse = mse_new_train
            num_terms += 1

        self._is_fit = True
        if num_terms == self._max_terms-1:
            print 'WARNING: Maximum number of directions reached'
            print '   Using more directions may improve fit'

        self._params = self._params[:np.argmin(self._cv_gof)+1]

    def _center_indepvars(self, X):
        """
        Internal method used to mean center the independent variables.
        Variables are rescaled to mean zero and variance one.
        """

        self._mu = X.mean(axis=0)
        self._sigma = np.sqrt(X.var(axis=0))

        # if the first column is all ones assume it is an intercept
        if self._mu[0] == 1.0:
            self._sigma[0] == 0
            self._sigma[0] = 1.0
        
        if self._categorical_cols is not None:
            self._mu[self._categorical_cols] = 0.0
            self._sigma[self._categorical_cols] = 1.0

        # drop columns with zero variance
        null_variance = (self._sigma == 0)
        if null_variance.any():
            print 'WARNING: Dropping {} columns for null variance'.format(
                null_variance.sum())
        
        self._pos_variance = np.logical_not(null_variance)
        self._mu = self._mu[self._pos_variance]
        self._sigma = self._sigma[self._pos_variance]
        X = X[:,self._pos_variance]

        shape = (1,X.shape[1])
        X_rsc = (X-self._mu.reshape(shape)) / self._sigma.reshape(shape)
        return X_rsc

    def _center_new_X(self, X):
        """
        Internal method used to center a new data matrix.
        """

        X = X[:,self._pos_variance]
        shape = (1,X.shape[1])
        return (X-self._mu.reshape(shape)) / self._sigma.reshape(shape)

    def _add_biases(self):
        """
        Internal method to add bias columns to the independent variables
        """
        N_test, N_train = self._X_test.shape[0], self._X_train.shape[0]
        self._X_train = np.concatenate((np.ones((N_train,1)), self._X_train), axis=1)
        self._X_test = np.concatenate((np.ones((N_test,1)), self._X_test), axis=1)

    @staticmethod
    def _train_test_split(X, y, split_pct):
        is_test_set = (np.random.rand(X.shape[0],1) > split_pct).ravel()
        X_test = X[is_test_set,:]
        y_test = y[is_test_set]
        X_train = X[np.logical_not(is_test_set),:]
        y_train = y[np.logical_not(is_test_set)]
        return X_train, y_train, X_test, y_test

    def _fit(self, X, y):
        """
        Internal workhorse function to fit an individual direction using
        iteratively reweighted least squares.
        """

        node = PPRNode()
        self._params.append(node)

        if not self._all_zeros:
            w = self._elastic_net_solve(X, y)
        else:
            w = np.zeros((X.shape[1],))

        # if more than 50% of weights are zero choose 50% of nonzero weights to jitter at random
        if (((w != 0).sum() / np.float64(w.size)) < 0.5):
            jitter = np.random.choice(np.where(w == 0.0)[0], size=w.size/2, replace=False)
            w[jitter] = np.random.rand(jitter.size,1).ravel()
        
        converged = False

        cost = np.Inf
        iteration = 0
        MAX_ITER = 100
        iter_costs = np.zeros((10,1))
        counter = 0
        while not converged:
            if self._debug_inner_iterations:
                print 'Inner iteration: {} => GOF: {}'.format(iteration, cost)

            v = X.dot(w)                               
            g, g_prime, spline = self._smoother(v, y)
            eps = y - g                                
            g_prime_squared = np.power(g_prime,2)   
            m = v + np.divide(eps, g_prime)            
            w = self._wls(X, g_prime_squared, m)

            node.spline = spline                       
            node.weights = w
            ix = counter % iter_costs.size

            iter_costs[ix] = self._compute_gof(X, y)
            cost_new = iter_costs.mean()     

            if np.abs(cost-cost_new) < self._inner_iter_tol:
                converged = True
            if iteration > MAX_ITER:
                break
            cost = cost_new
            iteration += 1
            counter += 1

    def _backfit(self):
        converged = False
        max_iters = 100
        cntr = 0
        cost = np.inf
        cost_orig = self._compute_gof(self._X_train, self._y_train)
        while not converged:
            node_counter = 0
            for node in self._params:
                v = self._X_train.dot(node.weights)
                z = self._predict(self._X_train, leave_out=node)
                residuals = self._y_train - z
                g, _, new_spline = self._smoother(v, residuals)
                node.spline = new_spline
            
            cost_new = self._compute_gof(self._X_train, self._y_train)
            delta_cost = np.abs(cost - cost_new)
            if self._debug_backfitting:
                msg = 'Backfitting Iter {} => Cost change: {}'
                print msg.format(cntr, delta_cost)
            cost = cost_new
            if delta_cost < 1e-3:    
                break
            if cntr > max_iters:
                break
            cntr += 1
        cost_after = self._compute_gof(self._X_train, self._y_train)
        if self._debug_backfitting:
            msg = 'Backfitting complete. Cost reduction: {}'
            print msg.format(cost_orig - cost_after)

    def _compute_gof(self, X, y):
        """
        Compute the goodness of fit for this model (as measured by MSE)
        """

        y_hat = self._predict(X)
        return np.mean(np.power(y - y_hat, 2))

    def predict(self, X=None, cv=False):
        """
        Compute the fitted values for this model. Set the CV flag to ``true''
        to use the test set. Optionally pass a new ``X'' matrix at which to
        evaluate the fit. May not contain missing values.
        """

        if X is not None and cv is True:
            raise StandardError('Cross validating a new sample is meaningless')
        if not self._is_fit:
            raise StandardError('Must fit model before predicting')
        if X is not None:
            _X = self._center_new_X(X)
            if self._use_bias:
                _X = np.concatenate((np.ones((X.shape[0],1)), _X), axis=1)
            return self._predict(_X)
        if cv:
            return self._predict(self._X_test)
        return self._predict(self._X_train)

    def _predict(self, X, leave_out=None):
        """
        Internal workhorse function to form predicted values for the current
        state of the model
        """

        directions = np.array(
            [p.spline(X.dot(p.weights)) for p in self._params if not (p is leave_out)])
        y = np.zeros(directions[0].shape)
        for direction in directions:
            y = y + direction
        return y

    def get_train_set(self):
        """
        Returns the training data set used by the model
        """
        return self._X_train, self._y_train

    def get_test_set(self):
        """
        Returns the test data set used by the model
        """
        return self._X_test, self._y_test

    def get_num_params(self):
        """
        Returns the number of parameters in the model
        """

        np = 0
        for p in self._params:
            np += p.weights.size
            np += p.spline.get_coeffs().size
        return np

    def plot_convergence(self, saving, xline=None):
        """
        Plots the convergence of the model as new ridge functions are included.
        Plots the train error and test error.
        """
        
        plt.plot(self._train_gof, color='blue', label='Training Error')
        plt.plot(self._cv_gof, color='green', label='Test Error')
        plt.ylabel('Mean Squared Error')
        plt.xlabel('Number of Terms in Model')
        if xline is not None:
            plt.axvline(x=xline, color='red')
        plt.legend()
        plt.savefig(saving)
        plt.close()

        base, _ = saving.rsplit('.',1)
        with open(base + '.csv', mode='w') as fh:
            fh.write('test,train\n')
            for tup in zip(self._cv_gof, self._train_gof):
                fh.write('{0},{1}\n'.format(*tup))

    def get_paths(self):
        """
        Returns the train loss and test loss path as each
        new ridge function is included in the model
        """
        
        return (self._train_gof, self._cv_gof)

    def _wls(self, X, w, y):
        """
        Internal method to fit a weighted least squares model
        """

        XTW = X.T*w
        return self._solve_linear_system(XTW.dot(X), XTW.dot(y))

    def _solve_linear_system(self, A, b):
        """
        Solves a system of equations Ax = b using ridge regularization
        """

        AA = A.copy()
        np.fill_diagonal(AA, self._stabilizer+np.diag(A))
        return alg.solve(AA, b)

    def _elastic_net_solve(self, X, y):
        l1_ratio_actual = self._l1_ratio if self._l1_ratio_cv is None else self._l1_ratio_cv
        E = ElasticNetCV(l1_ratio=l1_ratio_actual, fit_intercept=False)
        E.fit(X, y)
        w = E.coef_.ravel()
        if self._l1_ratio_cv is None:
            self._l1_ratio_cv = E.l1_ratio_
        if w.sum() == 0:
            self._all_zeros = True
            jitter = np.random.normal(0, 1, w.shape)
        else:
            jitter = 0
        return w + jitter
    
    def _least_squares_spline(self, v, y):
        """
        Fits y = g(v) where g is a cubic spline with the stipulated knots. 
        """
        ixs = np.argsort(v)
        _v = v[ixs]
        _y = y[ixs]
        if type(self._knots) is int:
            if self._knots > 1000:
                print 'WARNING: Large number of knots requested.'
                print 'Consider using fewer knots'
            knots_actual = np.percentile(_v, np.linspace(0,100,self._knots))
            if np.abs(knots_actual[0] - _v[0]) < 1e-6:
                knots_actual = knots_actual[1:]
            if np.abs(knots_actual[-1] - _v[-1]) < 1e-6:
                knots_actual = knots_actual[:-1]
        else:
            knots_actual = self._knots
        try:
            spline = ipl.LSQUnivariateSpline(_v, _y, knots_actual, k=3)
        except ValueError as e:
            msg = "Spline fit returned error\nTry reducing the number of knots"
            print e
            raise StandardError(msg)
        deriv = spline.derivative()
        return (spline(v), deriv(v), spline)

    def _smoothing_spline(self, v, y):
        ixs = np.argsort(v)
        _v = v[ixs]
        _y = y[ixs]

        spline = ipl.UnivariateSpline(_v, _y, s=self._smoothing_param)
        deriv = spline.derivative()
        return (spline(v), deriv(v), spline)

if __name__=='__main__':
    np.random.seed(19832)
    X = np.random.rand(1000,10)*100
    XR = X.copy()
    X = np.concatenate((X, (X[:,1]*X[:,2]).reshape(1000,1),
                       (X[:,1]*X[:,3]).reshape(1000,1)), axis=1)
    y = X.dot(np.random.rand(X.shape[1],1)) + np.random.normal(0,1,(1000,1))

    # increase the number of knots with caution!
    P = ProjectionPursuitRegressor(knots=2)
    P._debug_backfitting = True
    P.fit(XR, y)
    P.plot_convergence('ppr_convergence.png', xline=3)

    # compare with OLS
    sample = (np.random.rand(1000,1) > .80).ravel()
    X_train = XR[np.logical_not(sample),:]
    y_train = XR[np.logical_not(sample)]
    b = alg.solve(X_train.T.dot(X_train), X_train.T.dot(y_train))

    X_test = XR[sample,:]
    y_test = y[sample]

    y_hat = P.predict(X_test)
    eps = y_test.ravel() - y_hat
    print 'MSE (PPR): ', np.mean(np.power(eps,2))

    y_hat = X_test.dot(b)
    eps = y_test - y_hat
    print 'MSE (OLS): ', np.mean(np.power(eps,2))
