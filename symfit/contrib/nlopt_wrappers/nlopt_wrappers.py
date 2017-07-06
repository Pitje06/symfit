# -*- coding: utf-8 -*-

from ...core.minimizers import (BaseMinimizer, ConstrainedMinimizer,
                                BoundedMinimizer, GradientMinimizer)
from ...core.support import keywordonly, key2str

import warnings

import nlopt
import numpy as np
import sympy

DEFAULT_OPTIONS = dict(max_time=None, max_eval=None, xtol=1e-9, ftol=1e-15, verbose=False)
nlopt_default_kwarg_decorator = keywordonly(**DEFAULT_OPTIONS)


class NLOptMinimizer(object):
    EXIT_VALUES = {nlopt.SUCCESS: "Success",
                   nlopt.STOPVAL_REACHED: "Stopval reached",
                   nlopt.FTOL_REACHED: "Ftol reached",
                   nlopt.XTOL_REACHED: "Xtol reached",
                   nlopt.MAXEVAL_REACHED: "Maxeval reached",
                   nlopt.MAXTIME_REACHED: "Maxtime reached",
                   nlopt.FAILURE: "Failure",
                   nlopt.INVALID_ARGS: "Invalid arguments",
                   nlopt.OUT_OF_MEMORY: "Out of memory",
                   nlopt.ROUNDOFF_LIMITED: "Roundoff limited",
                   nlopt.FORCED_STOP: "Forced termination",
                   }

    def __init__(self, *args, **kwargs):
        self.evals = 0
        self.jac_evals = 0
        self.verbose = False
        super(NLOptMinimizer, self).__init__(*args, **kwargs)

    def wrapped_objective(self, xs, jac):
        self.evals += 1
        param_vals = {p.name: x for p, x in zip(self.params, xs)}
        f_val = self.objective(**param_vals)
        if jac.shape != (0,):
            self.jac_evals += 1
            jac[:] = self.jacobian(**param_vals)
        if self.verbose:
            out_str = 'x: {}; f_val: {}'.format(xs, f_val)
            if jac.shape != (0,):
                out_str += '; jac: {}'.format(jac)
            print(out_str)
        return f_val

    def execute(self, opt, x0):
        fit_results = {'popt': x0,
                       'ier': 0}
        try:
            xopt = opt.optimize(x0)
            fit_results['popt'] = xopt
        except Exception as error:
            warnings.warn("Your fit failed!", RuntimeWarning)
            raise
        finally:
            fit_results['value'] = opt.last_optimum_value()
            result = opt.last_optimize_result()
            fit_results['mesg'] = NLOptMinimizer.EXIT_VALUES[result]
            fit_results['infodic'] = {'nfev': self.evals, 'njev': self.jac_evals}
        return fit_results

    def set_options(self, opt, **options):
        max_time = options['max_time']
        max_eval = options['max_eval']
        xtol = options['xtol']
        ftol = options['ftol']
        self.verbose = options['verbose']
        opt.set_min_objective(self.wrapped_objective)
        if xtol is not None:
            opt.set_xtol_abs(xtol)
        if ftol is not None:
            opt.set_ftol_abs(ftol)
        if max_eval is not None:
            opt.set_maxeval(max_eval)
        if max_time is not None:
            opt.set_maxtime(max_time)
        super(NLOptMinimizer, self).set_options(opt, **options)


class NLOptConstrainedMinimizer(ConstrainedMinimizer):
    def wrap_constraint(self, cons):
        def con(xs, jac):
            param_vals = {p.name: x for p, x in zip(self.params, xs)}
            var_vals = {var.name: 0 for var in cons.model.vars}
            if jac.shape != (0, ):
                jac[:] = [-component(**param_vals, **var_vals) for component in cons.numerical_jacobian[0]]
#                jac[:] = cons.eval_jacobian(**param_vals, x=1, y=1, sigma_y=1)[0]
#                jac[:] *= -1
            con_val = -(cons(**param_vals, **var_vals)[0])
            if self.verbose:
                expr = list(cons.values())[0]
                comp = '<=' if cons.constraint_type is sympy.Ge else '=='
                cons_str = '{} {} 0'.format(expr, comp)
                out_str = '{:>25}: x: {}; val: {}'.format(cons_str, xs, con_val)
                if jac.shape != (0,):
                    out_str += '; jac: {}'.format(jac)
                print(out_str)
            return con_val
        return con

    def set_options(self, opt, tolerance=1e-8, **options):
        for cons in self.constraints:
            if cons.constraint_type is sympy.Eq:
                opt.add_equality_constraint(self.wrap_constraint(cons), tolerance)
            elif cons.constraint_type is sympy.Ge:
                opt.add_inequality_constraint(self.wrap_constraint(cons), tolerance)

        super(NLOptConstrainedMinimizer, self).set_options(opt, **options)


class NLOptBoundedMinimizer(BoundedMinimizer):
    def __init__(self, *args, **kwargs):
        super(NLOptBoundedMinimizer, self).__init__(*args, **kwargs)
        lower_bounds, upper_bounds = zip(*self.bounds)
        self.nlopt_lower_bounds = [-np.inf if bound is None else bound for bound in lower_bounds]
        self.nlopt_upper_bounds = [np.inf if bound is None else bound for bound in upper_bounds]

    def set_options(self, opt, **options):
        opt.set_lower_bounds(self.nlopt_lower_bounds)
        opt.set_upper_bounds(self.nlopt_upper_bounds)
#        super(NLOptBoundedMinimizer, self).set_options(opt, **options)

##################################
# BEGIN SPECIFIC IMPLEMENTATIONS #
##################################

# ----------------------
# Gradient based methods
# ----------------------

class MMA(NLOptMinimizer, GradientMinimizer, NLOptBoundedMinimizer):
    @nlopt_default_kwarg_decorator
    def execute(self, **options):
        opt = nlopt.opt(nlopt.LD_MMA, len(self.params))
        self.set_options(opt, **options)
        return super(MMA, self).execute(opt, self.initial_guesses)


class SLSQP(NLOptMinimizer, GradientMinimizer, NLOptConstrainedMinimizer, NLOptBoundedMinimizer):
    @nlopt_default_kwarg_decorator
    def execute(self, **options):
        opt = nlopt.opt(nlopt.LD_SLSQP, len(self.params))
        self.set_options(opt, **options)
        return super(SLSQP, self).execute(opt, self.initial_guesses)


class LBFGS(GradientMinimizer, NLOptMinimizer, NLOptBoundedMinimizer):
    @nlopt_default_kwarg_decorator
    def execute(self, **options):
        opt = nlopt.opt(nlopt.LD_LBFGS, len(self.params))
        self.set_options(opt, **options)
        return super(LBFGS, self).execute(opt, self.initial_guesses)


class TNewtonPR(NLOptMinimizer, GradientMinimizer, NLOptBoundedMinimizer):
    @nlopt_default_kwarg_decorator
    def execute(self, **options):
        opt = nlopt.opt(nlopt.LD_TNEWTON_PRECOND_RESTART, len(self.params))
        self.set_options(opt, **options)
        return super(TNewtonPR, self).execute(opt, self.initial_guesses)


class VariableMetric(NLOptMinimizer, GradientMinimizer, NLOptBoundedMinimizer):
    @nlopt_default_kwarg_decorator
    def execute(self, **options):
        opt = nlopt.opt(nlopt.LD_VAR2, len(self.params))
        self.set_options(opt, **options)
        return super(VariableMetric, self).execute(opt, self.initial_guesses)


# -----------------------
# Derivative free methods
# -----------------------

class COBYLA(NLOptMinimizer, NLOptConstrainedMinimizer, NLOptBoundedMinimizer):
    @nlopt_default_kwarg_decorator
    def execute(self, **options):
        opt = nlopt.opt(nlopt.LN_COBYLA, len(self.params))
        self.set_options(opt, **options)
        return super(COBYLA, self).execute(opt, self.initial_guesses)


class BOBYQA(NLOptMinimizer, NLOptBoundedMinimizer):
    @nlopt_default_kwarg_decorator
    def execute(self, **options):
        opt = nlopt.opt(nlopt.LN_BOBYQA, len(self.params))
        self.set_options(opt, **options)
        return super(BOBYQA, self).execute(opt, self.initial_guesses)


class PRAXIS(NLOptMinimizer, NLOptBoundedMinimizer):
    @nlopt_default_kwarg_decorator
    def execute(self, **options):
        opt = nlopt.opt(nlopt.LN_PRAXIS, len(self.params))
        self.set_options(opt, **options)
        return super(PRAXIS, self).execute(opt, self.initial_guesses)


class NelderMead(NLOptMinimizer, NLOptBoundedMinimizer):
    @nlopt_default_kwarg_decorator
    def execute(self, **options):
        opt = nlopt.opt(nlopt.LN_NELDERMEAD, len(self.params))
        self.set_options(opt, **options)
        return super(NelderMead, self).execute(opt, self.initial_guesses)


# Not called Subplex by original author's request.
# http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms#Local_derivative-free_optimization
class Sbplex(NLOptMinimizer, NLOptBoundedMinimizer):
    @nlopt_default_kwarg_decorator
    def execute(self, **options):
        opt = nlopt.opt(nlopt.LN_SBPLX, len(self.params))
        self.set_options(opt, **options)
        return super(Sbplex, self).execute(opt, self.initial_guesses)


# --------------
# Global methods
# --------------

class MLSL_MMA(NLOptMinimizer, GradientMinimizer, NLOptBoundedMinimizer):
    @nlopt_default_kwarg_decorator
    def execute(self, **options):
        opt = nlopt.opt(nlopt.GD_MLSL_LDS, len(self.params))
        self.set_options(opt, **options)
        x0 = self.initial_guesses
        local_opt = nlopt.opt(nlopt.LD_MMA, len(self.params))
        self.set_options(local_opt, **options)
        opt.set_local_optimizer(local_opt)
        return super(MLSL_MMA, self).execute(opt, x0)


class MLSL_BOBYQA(NLOptMinimizer, NLOptBoundedMinimizer):
    @nlopt_default_kwarg_decorator
    def execute(self, **options):
        opt = nlopt.opt(nlopt.GN_MLSL_LDS, len(self.params))
        self.set_options(opt, **options)
        x0 = self.initial_guesses
        local_opt = nlopt.opt(nlopt.LN_BOBYQA, len(self.params))
        self.set_options(local_opt, **options)
        opt.set_local_optimizer(local_opt)
        return super(MLSL_BOBYQA, self).execute(opt, x0)


class DirectL(NLOptMinimizer, NLOptBoundedMinimizer):
    @nlopt_default_kwarg_decorator
    def execute(self, **options):
        opt = nlopt.opt(nlopt.GN_DIRECT_L, len(self.params))
        self.set_options(opt, **options)
        return super(DirectL, self).execute(opt, self.initial_guesses)


class CRS(NLOptMinimizer, NLOptBoundedMinimizer):
    @nlopt_default_kwarg_decorator
    def execute(self, **options):
        opt = nlopt.opt(nlopt.GN_CRS2_LM, len(self.params))
        self.set_options(opt, **options)
        return super(CRS, self).execute(opt, self.initial_guesses)


class StoGO(NLOptMinimizer, GradientMinimizer, NLOptBoundedMinimizer):
    @nlopt_default_kwarg_decorator
    def execute(self, **options):
        opt = nlopt.opt(nlopt.GD_STOGO, len(self.params))
        self.set_options(opt, **options)
        return super(StoGO, self).execute(opt, self.initial_guesses)


class ISRES(NLOptMinimizer, NLOptConstrainedMinimizer, NLOptBoundedMinimizer):
    @nlopt_default_kwarg_decorator
    def execute(self, **options):
        opt = nlopt.opt(nlopt.GN_ISRES, len(self.params))
        self.set_options(opt, **options)
        return super(ISRES, self).execute(opt, self.initial_guesses)


class ESCH(NLOptMinimizer, NLOptBoundedMinimizer):
    @nlopt_default_kwarg_decorator
    def execute(self, **options):
        opt = nlopt.opt(nlopt.GN_ESCH, len(self.params))
        self.set_options(opt, **options)
        return super(ESCH, self).execute(opt, self.initial_guesses)
