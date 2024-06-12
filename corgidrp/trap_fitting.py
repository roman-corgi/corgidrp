import numpy as np
import warnings
from scipy.optimize import curve_fit

import check as check

def trap_fit_const_old(scheme, amps, times, num_pumps, fit_thresh, tau_min,
    tau_max, pc_min, pc_max, offset_min, offset_max, both_a = None):
    """For a given temperature, scheme, and pixel, this function examines data
    for amplitude vs phase time and fits for release time constant (tau) and
    the probability for capture (pc).  It tries fitting for a single trap in
    the pixel, and if the goodness of fit is not high enough (if less than
    fit_thresh), then the function attempts to fit for two traps in the pixel.
    The exception is the case where a 'both' type pixel is input; then only a
    two-trap fit is attempted.  The function assumes a constant for pc rather
    than its actual time-dependent form.
    Parameters
    ----------
    scheme : int, {1, 2, 3, 4}
        The scheme under consideration.  Only certain probability functions are
        valid for different schemes.
    amps : array
        Amplitudes of bright pixel of the dipole Units: e-.
    times : array
        Phase times in same order as amps.  Units:  seconds.
    num_pumps : int, > 0
        Number of cycles for trap pumping.
    fit_thresh : (0, 1)
        The minimum value required for adjusted coefficient of determination
        (adjusted R^2) for curve fitting for the release time constant
        (tau) using data for dipole amplitude vs phase time.  The closer to 1,
        the better the fit. Must be between 0 and 1.
    tau_min : float, >= 0
        Lower bound value for tau (release time constant) for curve fitting,
        in seconds.
    tau_max : float, > tau_min
        Upper bound value for tau (release time constant) for curve fitting,
        in seconds.
    pc_min : float, >= 0
        Lower bound value for pc (capture probability) for curve fitting,
        in e-.
    pc_max : float, > pc_min
        Upper bound value for pc (capture probability) for curve fitting,
        in e-.
    offset_min : float
        Lower bound value for the offset in the fitting of data for amplitude
        vs phase time.  Acts as a nuisance parameter.  Units of e-.
    offset_max : float, > offset_min
        Upper bound value for the offset in the fitting of data for amplitude
        vs phase time.  Acts as a nuisance parameter.  Units of e-.
    both_a : None or dict, optional
        Use None if you are fitting for a dipole that is of the 'above' or
        'below' kind.  Use the dictionary corresponding to
        rc_both[(row,col)]['above'] if you are fitting for a dipole that is of
        the 'both' kind.  See doc string of trap_id() for more details on this
        dictionary.  Defaults to None.
    Returns
    -------
    Dictionary of fit data.  Uses the following:
        prob: {1, 2, 3}; denotes probability function that gave the best fit
        pc: constant capture probability, best-fit value
        pc_err: standard deviation of pc
        tau: release time constant, best-fit value
        tau_err: standard deviation of tau
        If both_a = None and one trap is the best fit,
        the return is of the following form:
        {prob: [[pc, pc_err, tau, tau_err]]}
        If both_a = None and two traps is the best fit,
        the return is of the following form:
        {prob1: [[pc, pc_err, tau, tau_err]],
        prob2: [[pc2, pc2_err, tau2, tau2_err]]}
        If both traps are of prob1 type, then the return is like this:
        {prob1: [[pc, pc_err, tau, tau_err], [pc2, pc2_err, tau2, tau2_err]]}
        If both_a is not None, then the return is of the following form:
        {type1:{1: [[pc, pc_err, tau, tau_err]]},
        type2:{2: [[pc2, pc2_err, tau2, tau2_err]]}}
        where type1 is either 'a' for above or 'b' for below, and type2 is
        whichever of those type1 is not.
    """
    # TODO time-dep pc and pc2:  assume constant charge packet throughout the
    # different phase times across all num_pumps, when in reality, the charge
    # packet changes a little with each pump (a la 'express=num_pumps' in
    # arcticpy).  The fit function would technically be a sum of num_pumps
    # terms in which tauc is a slightly different value in each.  So the value
    # of tauc may not be that useful; could compare its uncertainty to that of
    # the values in the literature.  But it's generally good for large charge
    # packet values (perhaps it nominally corresponds to the average between
    # the starting charge packet value for trap pumping and the ending value
    # read off the highest-amp phase time frame.

    # check inputs
    if scheme != 1 and scheme != 2 and scheme != 3 and scheme != 4:
        raise TypeError('scheme must be 1, 2, 3, or 4.')
    try:
        amps = np.array(amps).astype(float)
    except:
        raise TypeError("amps elements should be real numbers")
    check.oneD_array(amps, 'amps', TypeError)
    try:
        times = np.array(times).astype(float)
    except:
        raise TypeError("times elements should be real numbers")
    check.oneD_array(times, 'times', TypeError)
    if len(np.unique(times)) < 6:
        raise IndexError('times must have a number of unique phase times '
        'longer than the number of fitted parameters.')
    if len(amps) != len(times):
        raise ValueError("times and amps should have same number of elements")
    check.positive_scalar_integer(num_pumps, 'num_pumps', TypeError)
    check.real_nonnegative_scalar(fit_thresh, 'fit_thresh', TypeError)
    if fit_thresh > 1:
        raise ValueError('fit_thresh should fall in (0,1).')
    check.real_nonnegative_scalar(tau_min, 'tau_min', TypeError)
    check.real_nonnegative_scalar(tau_max, 'tau_max', TypeError)
    if tau_max <= tau_min:
        raise ValueError('tau_max must be > tau_min')
    check.real_nonnegative_scalar(pc_min, 'pc_min', TypeError)
    check.real_nonnegative_scalar(pc_max, 'pc_max', TypeError)
    if pc_max <= pc_min:
        raise ValueError('pc_max must be > pc_min')
    check.real_scalar(offset_min, 'offset_min', TypeError)
    check.real_scalar(offset_max, 'offset_max', TypeError)
    if offset_max <= offset_min:
        raise ValueError('offset_max must be > offset_min')
    if (type(both_a) is not dict) and (both_a is not None):
        raise TypeError('both_a must be a rc_both[(row,col)][\'above\'] '
        'dictionary.  See trap_id() doc string for more details.')
    if both_a is not None:
        try:
            x = len(both_a['amp'])
            y = len(both_a['t'])
        except:
            raise KeyError('both_a must contain \'amp\' and \'t\' keys.')
        if x != y:
            raise ValueError('both_a[\'amp\'] and both_a[\'t\'] must have the '
            'same number of elements')
    # if both_a doesn't have expected keys, exceptions will be raised

    # define all these inside this function because they depend also on
    # num_pumps, and I can't specify an unfitted parameter in the function
    # definition if I want to use curve_fit
    def P1(time_data, offset, pc, tau):
        """Probability function 1, one trap.
        """
        return offset+(num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-2*time_data/tau)))

    def P1_P1(time_data, offset, pc, tau, pc2, tau2):
        """Probability function 1, two traps.
        """
        return offset+num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-2*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-2*time_data/tau2))

    def P2(time_data, offset, pc, tau):
        """Probability function 2, one trap.
        """
        return offset+(num_pumps*pc*(np.exp(-2*time_data/tau)-
            np.exp(-3*time_data/tau)))

    def P1_P2(time_data, offset, pc, tau, pc2, tau2):
        """One trap for probability function 1, one for probability function 2.
        """
        return offset+num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-2*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-2*time_data/tau2)-np.exp(-3*time_data/tau2))

    def P2_P2(time_data, offset, pc, tau, pc2, tau2):
        """Probability function 2, two traps.
        """
        return offset+num_pumps*pc*(np.exp(-2*time_data/tau)-
            np.exp(-3*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-2*time_data/tau2)-np.exp(-3*time_data/tau2))

    def P3(time_data, offset, pc, tau):
        """Probability function 3, one trap.
        """
        return offset+(num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-4*time_data/tau)))

    def P3_P3(time_data, offset, pc, tau, pc2, tau2):
        """Probability function 3, two traps.
        """
        return offset+num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-4*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-4*time_data/tau2))

    def P2_P3(time_data, offset, pc, tau, pc2, tau2):
        """One trap for probability function 2, one for probability function 3.
        """
        return offset+num_pumps*pc*(np.exp(-2*time_data/tau)-
            np.exp(-3*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-4*time_data/tau2))


    #upper bound for pc: 1*eperdn, but our knowledge of eperdn may have error.

    # Makes sense that you wouldn't find traps at times far away from time
    # constant, so to avoid false good fits, restrict bounds between 10^-6 and
    # 10^-2
    # in order, for one trap:  offset, pc, tau
    l_bounds_one = [offset_min, pc_min, tau_min]
    u_bounds_one = [offset_max, pc_max, tau_max]
    # avoid initial guess of 0
    offset0 = (offset_min+offset_max)/2
    if (offset_min+offset_max)/2 == 0:
        offset0 = min(1 + (offset_min+offset_max)/2, offset_max)
    pc0 = 1
    if pc_min > pc0 or pc_max < pc0:
        pc0 = (pc_min+pc_max)/2
    tau0 = np.median(times)
    if tau_min > tau0 or tau_max < tau0:
        tau0 = (tau_min+tau_max)/2
    p01 = [offset0, pc0, tau0]
    #l_bounds_one = [-100000, 0, 0]
    #u_bounds_one = [100000, 1, 1]
    bounds_one = (l_bounds_one, u_bounds_one)
    # in order, for two traps:  offset, pc, tau, pc2, tau2
    l_bounds_two = [offset_min, pc_min, tau_min, pc_min, tau_min]
    u_bounds_two = [offset_max, pc_max, tau_max, pc_max, tau_max]
    p02 = [offset0, pc0, tau0, pc0, tau0]
    #l_bounds_two = [-100000, 0, 0, 0, 0]
    #u_bounds_two = [100000, 1, 1, 1, 1]
    bounds_two = (l_bounds_two, u_bounds_two)

    if scheme == 1 or scheme == 2:

        # attempt all possible probability functions
        try:
            popt1, pcov1 = curve_fit(P1, times, amps, bounds=bounds_one,
            p0 = p01, maxfev = np.inf)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit1 = P1(times, popt1[0], popt1[1], popt1[2])
        #ss_r1 = np.sum((fit1 - np.mean(amps))**2)
        #ss_e1 = np.sum((fit2 - amps)**2)
        #R_value1 = (ss_r1/(ss_e1+ss_r1))**0.5
        ssres1 = np.sum((fit1 - amps)**2)
        sstot1 = np.sum((amps - np.mean(amps))**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value1 = 1 - (ssres1/sstot1)*(len(times) - 1)/(len(times) -
            len(popt1))

        try:
            popt2, pcov2 = curve_fit(P2, times, amps, bounds=bounds_one,
            p0 = p01, maxfev = np.inf)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit2 = P2(times, popt2[0], popt2[1], popt2[2])
        ssres2 = np.sum((fit2 - amps)**2)
        sstot2 = np.sum((amps - np.mean(amps))**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value2 = 1 - (ssres2/sstot2)*(len(times) - 1)/(len(times) -
            len(popt2))

        # accept the best fit and require threshold met
        maxR1 = max(R_value1, R_value2)

        if (both_a == None) and maxR1 >= fit_thresh and maxR1 == R_value1:
            pc = popt1[1]
            tau = popt1[2]
            _, pc_err, tau_err  = np.sqrt(np.diag(pcov1))
            return {1: [[pc, pc_err, tau, tau_err]]}
        if (both_a == None) and maxR1 >= fit_thresh and maxR1 == R_value2:
            pc = popt2[1]
            tau = popt2[2]
            _, pc_err, tau_err  = np.sqrt(np.diag(pcov2))
            return {2: [[pc, pc_err, tau, tau_err]]}

        # maxR1 must have been below fit_thresh.  Now try 2 traps
        try:
            popt11, pcov11 = curve_fit(P1_P1, times, amps, bounds=bounds_two,
            p0 = p02, maxfev = np.inf)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit11 = P1_P1(times, popt11[0], popt11[1], popt11[2], popt11[3],
        popt11[4])
        ssres11 = np.sum((fit11 - amps)**2)
        sstot11 = np.sum((amps - np.mean(amps))**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value11 = 1 - (ssres11/sstot11)*(len(times) - 1)/(len(times) -
            len(popt11))

        try:
            popt12, pcov12 = curve_fit(P1_P2, times, amps, bounds=bounds_two,
            p0 = p02, maxfev = np.inf)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit12 = P1_P2(times, popt12[0], popt12[1], popt12[2], popt12[3],
            popt12[4])
        ssres12 = np.sum((fit12 - amps)**2)
        sstot12 = np.sum((amps - np.mean(amps))**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value12 = 1 - (ssres12/sstot12)*(len(times) - 1)/(len(times) -
            len(popt12))

        try:
            popt22, pcov22 = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02, maxfev = np.inf)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit22 = P2_P2(times, popt22[0], popt22[1], popt22[2], popt22[3],
            popt22[4])
        ssres22 = np.sum((fit22 - amps)**2)
        sstot22 = np.sum((amps - np.mean(amps))**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value22 = 1 - (ssres22/sstot22)*(len(times) - 1)/(len(times) -
            len(popt22))

        maxR2 = max(R_value11, R_value12, R_value22)
        if maxR2 < fit_thresh:
            warnings.warn('No curve fit gave adjusted R^2 value above '
            'fit_thresh')
            return None

        if maxR2 == R_value11:
            off = popt11[0]
            pc = popt11[1]
            tau = popt11[2]
            pc2 = popt11[3]
            tau2 = popt11[4]
            _, pc_err, tau_err, pc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov11))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                comp_1 = np.sum((amp_a - (off+num_pumps*pc*(np.exp(-t_a/tau)-
                    np.exp(-2*t_a/tau))))**2)
                comp_2 = np.sum((amp_a - (off+num_pumps*pc2*(np.exp(-t_a/tau2)-
                    np.exp(-2*t_a/tau2))))**2)
                # the sum of squared residuals that is less: better fit
                if comp_1 <= comp_2:
                    return {'a':{1: [[pc, pc_err, tau, tau_err]]},
                        'b':{1: [[pc2, pc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{1: [[pc, pc_err, tau, tau_err]]},
                        'a':{1: [[pc2, pc2_err, tau2, tau2_err]]}}
            return {1: [[pc, pc_err, tau, tau_err],
                [pc2, pc2_err, tau2, tau2_err]]}

        if maxR2 == R_value12:
            off = popt12[0]
            pc = popt12[1]
            tau = popt12[2]
            pc2 = popt12[3]
            tau2 = popt12[4]
            _, pc_err, tau_err, pc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov12))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                comp_1 = np.sum((amp_a - (off+num_pumps*pc*(np.exp(-t_a/tau)-
                    np.exp(-2*t_a/tau))))**2)
                comp_2 = np.sum((amp_a-(off+num_pumps*pc2*(np.exp(-2*t_a/tau2)-
                    np.exp(-3*t_a/tau2))))**2)
                if comp_1 <= comp_2:
                    return {'a':{1: [[pc, pc_err, tau, tau_err]]},
                        'b':{2: [[pc2, pc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{1: [[pc, pc_err, tau, tau_err]]},
                        'a':{2: [[pc2, pc2_err, tau2, tau2_err]]}}
            return {1: [[pc, pc_err, tau, tau_err]],
                2: [[pc2, pc2_err, tau2, tau2_err]]}

        if maxR2 == R_value22:
            off = popt22[0]
            pc = popt22[1]
            tau = popt22[2]
            pc2 = popt22[3]
            tau2 = popt22[4]
            _, pc_err, tau_err, pc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov22))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                # TODO v2, for trap_fit and trap_fit_const:
                # if 'amp' list for 'above' is identical to 'amp'
                #list for 'below', then earmark for assignment whenever
                # finding optimal matchings of schemes for sub-el loc; if
                #no particular assignment is better at that point, then the
                #assignment doesn't matter.
                # TODO v2: And could make process more accurate in general?
                # Calculate another adjusted R^2 for each and take the bigger?
                # The split data are not exactly the one-P curves exactly, but
                # could still be better than this?
                comp_1 = np.sum((amp_a -(off+num_pumps*pc*(np.exp(-2*t_a/tau)-
                    np.exp(-3*t_a/tau))))**2)
                comp_2 = np.sum((amp_a-(off+num_pumps*pc2*(np.exp(-2*t_a/tau2)-
                    np.exp(-3*t_a/tau2))))**2)
                # the sum of squared residuals that is less: better fit
                if comp_1 <= comp_2:
                    return {'a':{2: [[pc, pc_err, tau, tau_err]]},
                        'b':{2: [[pc2, pc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{2: [[pc, pc_err, tau, tau_err]]},
                        'a':{2: [[pc2, pc2_err, tau2, tau2_err]]}}
            return {2: [[pc, pc_err, tau, tau_err],
                [pc2, pc2_err, tau2, tau2_err]]}

    if scheme == 3 or scheme == 4:
        #attempt both probability functions (the more probable P3 listed first)
        try:
            popt3, pcov3 = curve_fit(P3, times, amps, bounds=bounds_one,
            p0 = p01, maxfev = np.inf)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit3 = P3(times, popt3[0], popt3[1], popt3[2])
        ssres3 = np.sum((fit3 - amps)**2)
        sstot3 = np.sum((amps - np.mean(amps))**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value3 = 1 - (ssres3/sstot3)*(len(times) - 1)/(len(times) -
            len(popt3))

        try:
            popt2, pcov2 = curve_fit(P2, times, amps, bounds=bounds_one,
            p0 = p01, maxfev = np.inf)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit2 = P2(times, popt2[0], popt2[1], popt2[2])
        ssres2 = np.sum((fit2 - amps)**2)
        sstot2 = np.sum((amps - np.mean(amps))**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value2 = 1 - (ssres2/sstot2)*(len(times) - 1)/(len(times) -
            len(popt2))

        # accept the best fit and require threshold met
        maxR1 = max(R_value3, R_value2)

        if (both_a == None) and maxR1 >= fit_thresh and maxR1 == R_value3:
            pc = popt3[1]
            tau = popt3[2]
            _, pc_err, tau_err  = np.sqrt(np.diag(pcov3))
            return {3: [[pc, pc_err, tau, tau_err]]}
        if (both_a == None) and maxR1 >= fit_thresh and maxR1 == R_value2:
            pc = popt2[1]
            tau = popt2[2]
            _, pc_err, tau_err  = np.sqrt(np.diag(pcov2))
            return {2: [[pc, pc_err, tau, tau_err]]}

        # maxR1 must have been below fit_thresh.  Now try 2 traps

        try:
            popt33, pcov33 = curve_fit(P3_P3, times, amps, bounds=bounds_two,
            p0 = p02, maxfev = np.inf)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit33 = P3_P3(times, popt33[0], popt33[1], popt33[2], popt33[3],
            popt33[4])
        ssres33 = np.sum((fit33 - amps)**2)
        sstot33 = np.sum((amps - np.mean(amps))**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value33 = 1 - (ssres33/sstot33)*(len(times) - 1)/(len(times) -
            len(popt33))

        try:
            popt23, pcov23 = curve_fit(P2_P3, times, amps, bounds=bounds_two,
            p0 = p02, maxfev = np.inf)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit23 = P2_P3(times, popt23[0], popt23[1], popt23[2], popt23[3],
            popt23[4])
        ssres23 = np.sum((fit23 - amps)**2)
        sstot23 = np.sum((amps - np.mean(amps))**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value23 = 1 - (ssres23/sstot23)*(len(times) - 1)/(len(times) -
            len(popt23))

        try:
            popt22, pcov22 = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02, maxfev = np.inf)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit22 = P2_P2(times, popt22[0], popt22[1], popt22[2], popt22[3],
            popt22[4])
        ssres22 = np.sum((fit22 - amps)**2)
        sstot22 = np.sum((amps - np.mean(amps))**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value22 = 1 - (ssres22/sstot22)*(len(times) - 1)/(len(times) -
            len(popt22))

        maxR2 = max(R_value33, R_value23, R_value22)

        if maxR2 < fit_thresh:
            warnings.warn('No curve fit gave adjusted R^2 value above '
            'fit_thresh')
            return None

        if maxR2 == R_value33:
            off = popt33[0]
            pc = popt33[1]
            tau = popt33[2]
            pc2 = popt33[3]
            tau2 = popt33[4]
            _, pc_err, tau_err, pc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov33))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                comp_1 = np.sum((amp_a - (off+num_pumps*pc*(np.exp(-t_a/tau)-
                    np.exp(-4*t_a/tau))))**2)
                comp_2 = np.sum((amp_a - (off+num_pumps*pc2*(np.exp(-t_a/tau2)-
                    np.exp(-4*t_a/tau2))))**2)
                if comp_1 <= comp_2:
                    return {'a':{3: [[pc, pc_err, tau, tau_err]]},
                        'b':{3: [[pc2, pc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{3: [[pc, pc_err, tau, tau_err]]},
                        'a':{3: [[pc2, pc2_err, tau2, tau2_err]]}}
            return {3: [[pc, pc_err, tau, tau_err], [pc2, pc2_err,
                tau2, tau2_err]]}

        if maxR2 == R_value23:
            off = popt23[0]
            pc = popt23[1]
            tau = popt23[2]
            pc2 = popt23[3]
            tau2 = popt23[4]
            _, pc_err, tau_err, pc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov23))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                comp_1 = np.sum((amp_a - (off+num_pumps*pc*(np.exp(-2*t_a/tau)-
                    np.exp(-3*t_a/tau))))**2)
                comp_2 = np.sum((amp_a - (off+num_pumps*pc2*(np.exp(-t_a/tau2)-
                    np.exp(-4*t_a/tau2))))**2)
                if comp_1 <= comp_2:
                    return {'a':{2: [[pc, pc_err, tau, tau_err]]},
                        'b':{3: [[pc2, pc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{2: [[pc, pc_err, tau, tau_err]]},
                        'a':{3: [[pc2, pc2_err, tau2, tau2_err]]}}
            return {2: [[pc, pc_err, tau, tau_err]],
                3: [[pc2, pc2_err, tau2, tau2_err]]}
        if maxR2 == R_value22:
            off = popt22[0]
            pc = popt22[1]
            tau = popt22[2]
            pc2 = popt22[3]
            tau2 = popt22[4]
            _, pc_err, tau_err, pc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov22))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                comp_1 = np.sum((amp_a - (off+num_pumps*pc*(np.exp(-2*t_a/tau)-
                    np.exp(-3*t_a/tau))))**2)
                comp_2 = np.sum((amp_a-(off+num_pumps*pc2*(np.exp(-2*t_a/tau2)-
                    np.exp(-3*t_a/tau2))))**2)
                if comp_1 <= comp_2:
                    return {'a':{2: [[pc, pc_err, tau, tau_err]]},
                        'b':{2: [[pc2, pc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{2: [[pc, pc_err, tau, tau_err]]},
                        'a':{2: [[pc2, pc2_err, tau2, tau2_err]]}}
            return {2: [[pc, pc_err, tau, tau_err],
                [pc2, pc2_err, tau2, tau2_err]]}

def trap_fit_old(scheme, amps, times, num_pumps, fit_thresh, tau_min, tau_max,
    tauc_min, tauc_max, offset_min, offset_max, both_a = None):
    """For a given temperature, scheme, and pixel, this function examines data
    for amplitude vs phase time and fits for release time constant (tau) and
    the probability for capture (pc).  It tries fitting for a single trap in
    the pixel, and if the goodness of fit is not high enough (if less than
    fit_thresh), then the function attempts to fit for two traps in the pixel.
    The exception is the case where a 'both' type pixel is input; then only a
    two-trap fit is attempted.  The function assumes the full time-dependent
    form for capture probability.
    Parameters
    ----------
    scheme : int, {1, 2, 3, 4}
        The scheme under consideration.  Only certain probability functions are
        valid for different schemes.
    amps : array
        Amplitudes of bright pixel of the dipole Units: e-.
    times : array
        Phase times in same order as amps.  Units:  seconds.
    num_pumps : int, > 0
        Number of cycles for trap pumping.
    fit_thresh : (0, 1)
        The minimum value required for adjusted coefficient of determination
        (adjusted R^2) for curve fitting for the release time constant
        (tau) using data for dipole amplitude vs phase time.  The closer to 1,
        the better the fit. Must be between 0 and 1.
    tau_min : float, >= 0
        Lower bound value for tau (release time constant) for curve fitting,
        in seconds.
    tau_max : float, > tau_min
        Upper bound value for tau (release time constant) for curve fitting,
        in seconds.
    tauc_min : float, >= 0
        Lower bound value for tauc (capture time constant) for curve fitting,
        in e-.
    tauc_max : float, > tau_min
        Upper bound value for tauc (capture time constant) for curve fitting,
        in e-.
    offset_min : float
        Lower bound value for the offset in the fitting of data for amplitude
        vs phase time.  Acts as a nuisance parameter.  Units of e-.
    offset_max : float, > offset_min
        Upper bound value for the offset in the fitting of data for amplitude
        vs phase time.  Acts as a nuisance parameter.  Units of e-.
    both_a : None or dict, optional
        Use None if you are fitting for a dipole that is of the 'above' or
        'below' kind.  Use the dictionary corresponding to
        rc_both[(row,col)]['above'] if you are fitting for a dipole that is of
        the 'both' kind.  See doc string of trap_id() for more details on this
        dictionary.  Defaults to None.
    Returns
    -------
    Dictionary of fit data.  Uses the following:
        prob: {1, 2, 3}; denotes probability function that gave the best fit
        pc: constant capture probability, best-fit value
        pc_err: standard deviation of pc
        tau: release time constant, best-fit value
        tau_err: standard deviation of tau
        If both_a = None and one trap is the best fit,
        the return is of the following form:
        {prob: [[pc, pc_err, tau, tau_err]]}
        If both_a = None and two traps is the best fit,
        the return is of the following form:
        {prob1: [[pc, pc_err, tau, tau_err]],
        prob2: [[pc2, pc2_err, tau2, tau2_err]]}
        If both traps are of prob1 type, then the return is like this:
        {prob1: [[pc, pc_err, tau, tau_err], [pc2, pc2_err, tau2, tau2_err]]}
        If both_a is not None, then the return is of the following form:
        {type1:{1: [[pc, pc_err, tau, tau_err]]},
        type2:{2: [[pc2, pc2_err, tau2, tau2_err]]}}
        where type1 is either 'a' for above or 'b' for below, and type2 is
        whichever of those type1 is not.
    """
    # TODO time-dep pc and pc2:  assume constant charge packet throughout the
    # different phase times across all num_pumps, when in reality, the charge
    # packet changes a little with each pump (a la 'express=num_pumps' in
    # arcticpy).  The fit function would technically be a sum of num_pumps
    # terms in which tauc is a slightly different value in each.  So the value
    # of tauc may not be that useful; could compare its uncertainty to that of
    # the values in the literature.  But it's generally good for large charge
    # packet values (perhaps it nominally corresponds to the average between
    # the starting charge packet value for trap pumping and the ending value
    # read off the highest-amp phase time frame.

    # check inputs
    if scheme != 1 and scheme != 2 and scheme != 3 and scheme != 4:
        raise TypeError('scheme must be 1, 2, 3, or 4.')
    try:
        amps = np.array(amps).astype(float)
    except:
        raise TypeError("amps elements should be real numbers")
    check.oneD_array(amps, 'amps', TypeError)
    try:
        times = np.array(times).astype(float)
    except:
        raise TypeError("times elements should be real numbers")
    check.oneD_array(times, 'times', TypeError)
    if len(np.unique(times)) < 6:
        raise IndexError('times must have a number of unique phase times '
        'longer than the number of fitted parameters.')
    if len(amps) != len(times):
        raise ValueError("times and amps should have same number of elements")
    check.positive_scalar_integer(num_pumps, 'num_pumps', TypeError)
    check.real_nonnegative_scalar(fit_thresh, 'fit_thresh', TypeError)
    if fit_thresh > 1:
        raise ValueError('fit_thresh should fall in (0,1).')
    check.real_nonnegative_scalar(tau_min, 'tau_min', TypeError)
    check.real_nonnegative_scalar(tau_max, 'tau_max', TypeError)
    if tau_max <= tau_min:
        raise ValueError('tau_max must be > tau_min')
    check.real_nonnegative_scalar(tauc_min, 'tauc_min', TypeError)
    check.real_nonnegative_scalar(tauc_max, 'tauc_max', TypeError)
    if tauc_max <= tauc_min:
        raise ValueError('tauc_max must be > tauc_min')
    check.real_scalar(offset_min, 'offset_min', TypeError)
    check.real_scalar(offset_max, 'offset_max', TypeError)
    if offset_max <= offset_min:
        raise ValueError('offset_max must be > offset_min')
    if (type(both_a) is not dict) and (both_a is not None):
        raise TypeError('both_a must be a rc_both[(row,col)][\'above\'] '
        'dictionary or None.  See trap_id() doc string for more details.')
    if both_a is not None:
        try:
            x = len(both_a['amp'])
            y = len(both_a['t'])
        except:
            raise KeyError('both_a must contain \'amp\' and \'t\' keys.')
        if x != y:
            raise ValueError('both_a[\'amp\'] and both_a[\'t\'] must have the '
            'same number of elements')
    # if input for both_a doesn't conform to expected dictionary structure,
    # exceptions will be raised

    # define all these inside this function because they depend also on
    # num_pumps, and I can't specify an unfitted parameter in the function
    # definition if I want to use curve_fit
    def P1(time_data, offset, tauc, tau):
        """Probability function 1, one trap.
        """
        pc = 1 - np.exp(-time_data/tauc)
        return offset+(num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-2*time_data/tau)))

    def P1_P1(time_data, offset, tauc, tau, tauc2, tau2):
        """Probability function 1, two traps.
        """
        pc = 1 - np.exp(-time_data/tauc)
        pc2 = 1 - np.exp(-time_data/tauc2)
        return offset+num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-2*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-2*time_data/tau2))

    def P2(time_data, offset, tauc, tau):
        """Probability function 2, one trap.
        """
        pc = 1 - np.exp(-time_data/tauc)
        return offset+(num_pumps*pc*(np.exp(-2*time_data/tau)-
            np.exp(-3*time_data/tau)))

    def P1_P2(time_data, offset, tauc, tau, tauc2, tau2):
        """One trap for probability function 1, one for probability function 2.
        """
        pc = 1 - np.exp(-time_data/tauc)
        pc2 = 1 - np.exp(-time_data/tauc2)
        return offset+num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-2*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-2*time_data/tau2)-np.exp(-3*time_data/tau2))

    def P2_P2(time_data, offset, tauc, tau, tauc2, tau2):
        """Probability function 2, two traps.
        """
        pc = 1 - np.exp(-time_data/tauc)
        pc2 = 1 - np.exp(-time_data/tauc2)
        return offset+num_pumps*pc*(np.exp(-2*time_data/tau)-
            np.exp(-3*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-2*time_data/tau2)-np.exp(-3*time_data/tau2))

    def P3(time_data, offset, tauc, tau):
        """Probability function 3, one trap.
        """
        pc = 1 - np.exp(-time_data/tauc)
        return offset+(num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-4*time_data/tau)))

    def P3_P3(time_data, offset, tauc, tau, tauc2, tau2):
        """Probability function 3, two traps.
        """
        pc = 1 - np.exp(-time_data/tauc)
        pc2 = 1 - np.exp(-time_data/tauc2)
        return offset+num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-4*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-4*time_data/tau2))

    def P2_P3(time_data, offset, tauc, tau, tauc2, tau2):
        """One trap for probability function 2, one for probability function 3.
        """
        pc = 1 - np.exp(-time_data/tauc)
        pc2 = 1 - np.exp(-time_data/tauc2)
        return offset+num_pumps*pc*(np.exp(-2*time_data/tau)-
            np.exp(-3*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-4*time_data/tau2))

    #upper bound for pc: 1*eperdn, but our knowledge of eperdn may have error.

    # Makes sense that you wouldn't find traps at times far away from time
    # constant, so to avoid false good fits, restrict bounds between 10^-6 and
    # 10^-2
    # in order, for one trap:  offset, tauc, tau
    l_bounds_one = [offset_min, tauc_min, tau_min]
    u_bounds_one = [offset_max, tauc_max, tau_max]
    # avoid initial guess of 0
    offset0 = (offset_min+offset_max)/2
    if (offset_min+offset_max)/2 == 0:
        offset0 = min(1 + (offset_min+offset_max)/2, offset_max)
    tauc0 = np.median(times) #tauc0 = 1e-7
    if tauc_min > tauc0 or tauc_max < tauc0:
        tauc0 = (tauc_min+tauc_max)/2
    tau0 = np.median(times)
    if tau_min > tau0 or tau_max < tau0:
        tau0 = (tau_min+tau_max)/2
    p01 = [offset0, tauc0, tau0]
    #l_bounds_one = [-100000, 0, 0]
    #u_bounds_one = [100000, 1, 1]
    bounds_one = (l_bounds_one, u_bounds_one)
    # in order, for two traps:  offset, tauc, tau, tauc2, tau2
    l_bounds_two = [offset_min, tauc_min, tau_min, tauc_min, tau_min]
    u_bounds_two = [offset_max, tauc_max, tau_max, tauc_max, tau_max]
    p02 = [offset0, tauc0, tau0, tauc0, tau0]
    #l_bounds_two = [-100000, 0, 0, 0, 0]
    #u_bounds_two = [100000, 1, 1, 1, 1]
    bounds_two = (l_bounds_two, u_bounds_two)

    if scheme == 1 or scheme == 2:

        # attempt all possible probability functions
        try:
            popt1, pcov1 = curve_fit(P1, times, amps, bounds=bounds_one,
            p0 = p01, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit1 = P1(times, popt1[0], popt1[1], popt1[2])
        ssres1 = np.sum((fit1 - amps)**2)
        sstot1 = np.sum((amps - np.mean(amps))**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value1 = 1 - (ssres1/sstot1)*(len(times) - 1)/(len(times) -
            len(popt1))

        try:
            popt2, pcov2 = curve_fit(P2, times, amps, bounds=bounds_one,
            p0 = p01, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit2 = P2(times, popt2[0], popt2[1], popt2[2])
        ssres2 = np.sum((fit2 - amps)**2)
        sstot2 = np.sum((amps - np.mean(amps))**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value2 = 1 - (ssres2/sstot2)*(len(times) - 1)/(len(times) -
            len(popt2))

        # accept the best fit and require threshold met
        maxR1 = max(R_value1, R_value2)

        if (both_a == None) and maxR1 >= fit_thresh and maxR1 == R_value1:
            tauc = popt1[1]
            tau = popt1[2]
            _, tauc_err, tau_err  = np.sqrt(np.diag(pcov1))
            return {1: [[tauc, tauc_err, tau, tau_err]]}
        if (both_a == None) and maxR1 >= fit_thresh and maxR1 == R_value2:
            tauc = popt2[1]
            tau = popt2[2]
            _, tauc_err, tau_err  = np.sqrt(np.diag(pcov2))
            return {2: [[tauc, tauc_err, tau, tau_err]]}

        # maxR1 must have been below fit_thresh.  Now try 2 traps

        try:
            popt11, pcov11 = curve_fit(P1_P1, times, amps, bounds=bounds_two,
            p0 = p02, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit11 = P1_P1(times, popt11[0], popt11[1], popt11[2], popt11[3],
        popt11[4])
        ssres11 = np.sum((fit11 - amps)**2)
        sstot11 = np.sum((amps - np.mean(amps))**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value11 = 1 - (ssres11/sstot11)*(len(times) - 1)/(len(times) -
            len(popt11))

        try:
            popt12, pcov12 = curve_fit(P1_P2, times, amps, bounds=bounds_two,
            p0 = p02, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit12 = P1_P2(times, popt12[0], popt12[1], popt12[2], popt12[3],
            popt12[4])
        ssres12 = np.sum((fit12 - amps)**2)
        sstot12 = np.sum((amps - np.mean(amps))**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value12 = 1 - (ssres12/sstot12)*(len(times) - 1)/(len(times) -
            len(popt12))

        try:
            popt22, pcov22 = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit22 = P2_P2(times, popt22[0], popt22[1], popt22[2], popt22[3],
            popt22[4])
        ssres22 = np.sum((fit22 - amps)**2)
        sstot22 = np.sum((amps - np.mean(amps))**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value22 = 1 - (ssres22/sstot22)*(len(times) - 1)/(len(times) -
            len(popt22))

        maxR2 = max(R_value11, R_value12, R_value22)
        if maxR2 < fit_thresh:
            warnings.warn('No curve fit gave adjusted R^2 value above '
            'fit_thresh')
            return None

        if maxR2 == R_value11:
            off = popt11[0]
            tauc = popt11[1]
            tau = popt11[2]
            tauc2 = popt11[3]
            tau2 = popt11[4]
            _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov11))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                # TODO v2, for trap_fit and trap_fit_const:
                # if 'amp' list for 'above' is identical to 'amp'
                #list for 'below', then earmark for assignment whenever
                # finding optimal matchings of schemes for sub-el loc; if
                #no particular assignment is better at that point, then the
                #assignment doesn't matter
                pc = 1 - np.exp(-t_a/tauc)
                pc2 = 1 - np.exp(-t_a/tauc2)
                comp_1 = np.sum((amp_a - (off+num_pumps*pc*(np.exp(-t_a/tau)-
                    np.exp(-2*t_a/tau))))**2)
                comp_2 = np.sum((amp_a - (off+num_pumps*pc2*(np.exp(-t_a/tau2)-
                    np.exp(-2*t_a/tau2))))**2)
                # the sum of squared residuals that is less: better fit
                if comp_1 <= comp_2:
                    return {'a':{1: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{1: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{1: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{1: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {1: [[tauc, tauc_err, tau, tau_err],
                [tauc2, tauc2_err, tau2, tau2_err]]}

        if maxR2 == R_value12:
            off = popt12[0]
            tauc = popt12[1]
            tau = popt12[2]
            tauc2 = popt12[3]
            tau2 = popt12[4]
            _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov12))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                pc = 1 - np.exp(-t_a/tauc)
                pc2 = 1 - np.exp(-t_a/tauc2)
                comp_1 = np.sum((amp_a - (off+num_pumps*pc*(np.exp(-t_a/tau)-
                    np.exp(-2*t_a/tau))))**2)
                comp_2 = np.sum((amp_a-(off+num_pumps*pc2*(np.exp(-2*t_a/tau2)-
                    np.exp(-3*t_a/tau2))))**2)
                if comp_1 <= comp_2:
                    return {'a':{1: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{1: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {1: [[tauc, tauc_err, tau, tau_err]],
                2: [[tauc2, tauc2_err, tau2, tau2_err]]}

        if maxR2 == R_value22:
            off = popt22[0]
            tauc = popt22[1]
            tau = popt22[2]
            tauc2 = popt22[3]
            tau2 = popt22[4]
            _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov22))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                pc = 1 - np.exp(-t_a/tauc)
                pc2 = 1 - np.exp(-t_a/tauc2)
                comp_1 = np.sum((amp_a - (off+num_pumps*pc*(np.exp(-2*t_a/tau)-
                    np.exp(-3*t_a/tau))))**2)
                comp_2 = np.sum((amp_a-(off+num_pumps*pc2*(np.exp(-2*t_a/tau2)-
                    np.exp(-3*t_a/tau2))))**2)
                if comp_1 <= comp_2:
                    return {'a':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {2: [[tauc, tauc_err, tau, tau_err],
                [tauc2, tauc2_err, tau2, tau2_err]]}

    if scheme == 3 or scheme == 4:
        #attempt both probability functions
        try:
            popt3, pcov3 = curve_fit(P3, times, amps, bounds=bounds_one,
            p0 = p01, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit3 = P3(times, popt3[0], popt3[1], popt3[2])
        ssres3 = np.sum((fit3 - amps)**2)
        sstot3 = np.sum((amps - np.mean(amps))**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value3 = 1 - (ssres3/sstot3)*(len(times) - 1)/(len(times) -
            len(popt3))

        try:
            popt2, pcov2 = curve_fit(P2, times, amps, bounds=bounds_one,
            p0 = p01, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit2 = P2(times, popt2[0], popt2[1], popt2[2])
        ssres2 = np.sum((fit2 - amps)**2)
        sstot2 = np.sum((amps - np.mean(amps))**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value2 = 1 - (ssres2/sstot2)*(len(times) - 1)/(len(times) -
            len(popt2))

        # accept the best fit and require threshold met
        maxR1 = max(R_value3, R_value2)

        if (both_a == None) and maxR1 >= fit_thresh and maxR1 == R_value3:
            tauc = popt3[1]
            tau = popt3[2]
            _, tauc_err, tau_err  = np.sqrt(np.diag(pcov3))
            return {3: [[tauc, tauc_err, tau, tau_err]]}
        if (both_a == None) and maxR1 >= fit_thresh and maxR1 == R_value2:
            tauc = popt2[1]
            tau = popt2[2]
            _, tauc_err, tau_err  = np.sqrt(np.diag(pcov2))
            return {2: [[tauc, tauc_err, tau, tau_err]]}

        # maxR1 must have been below fit_thresh.  Now try 2 traps

        try:
            popt33, pcov33 = curve_fit(P3_P3, times, amps, bounds=bounds_two,
            p0 = p02, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit33 = P3_P3(times, popt33[0], popt33[1], popt33[2], popt33[3],
            popt33[4])
        ssres33 = np.sum((fit33 - amps)**2)
        sstot33 = np.sum((amps - np.mean(amps))**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value33 = 1 - (ssres33/sstot33)*(len(times) - 1)/(len(times) -
            len(popt33))

        try:
            popt23, pcov23 = curve_fit(P2_P3, times, amps, bounds=bounds_two,
            p0 = p02, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit23 = P2_P3(times, popt23[0], popt23[1], popt23[2], popt23[3],
            popt23[4])
        ssres23 = np.sum((fit23 - amps)**2)
        sstot23 = np.sum((amps - np.mean(amps))**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value23 = 1 - (ssres23/sstot23)*(len(times) - 1)/(len(times) -
            len(popt23))

        try:
            popt22, pcov22 = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit22 = P2_P2(times, popt22[0], popt22[1], popt22[2], popt22[3],
            popt22[4])
        ssres22 = np.sum((fit22 - amps)**2)
        sstot22 = np.sum((amps - np.mean(amps))**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value22 = 1 - (ssres22/sstot22)*(len(times) - 1)/(len(times) -
            len(popt22))

        maxR2 = max(R_value33, R_value23, R_value22)

        if maxR2 < fit_thresh:
            warnings.warn('No curve fit gave adjusted R^2 value above '
            'fit_thresh')
            return None

        if maxR2 == R_value33:
            off = popt33[0]
            tauc = popt33[1]
            tau = popt33[2]
            tauc2 = popt33[3]
            tau2 = popt33[4]
            _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov33))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                pc = 1 - np.exp(-t_a/tauc)
                pc2 = 1 - np.exp(-t_a/tauc2)
                comp_1 = np.sum((amp_a - (off+num_pumps*pc*(np.exp(-t_a/tau)-
                    np.exp(-4*t_a/tau))))**2)
                comp_2 = np.sum((amp_a - (off+num_pumps*pc2*(np.exp(-t_a/tau2)-
                    np.exp(-4*t_a/tau2))))**2)
                if comp_1 <= comp_2:
                    return {'a':{3: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{3: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{3: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{3: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {3: [[tauc, tauc_err, tau, tau_err], [tauc2, tauc2_err,
                tau2, tau2_err]]}

        if maxR2 == R_value23:
            off = popt23[0]
            tauc = popt23[1]
            tau = popt23[2]
            tauc2 = popt23[3]
            tau2 = popt23[4]
            _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov23))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                pc = 1 - np.exp(-t_a/tauc)
                pc2 = 1 - np.exp(-t_a/tauc2)
                comp_1 = np.sum((amp_a - (off+num_pumps*pc*(np.exp(-2*t_a/tau)-
                    np.exp(-3*t_a/tau))))**2)
                comp_2 = np.sum((amp_a - (off+num_pumps*pc2*(np.exp(-t_a/tau2)-
                    np.exp(-4*t_a/tau2))))**2)
                if comp_1 <= comp_2:
                    return {'a':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{3: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{3: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {2: [[tauc, tauc_err, tau, tau_err]],
                3: [[tauc2, tauc2_err, tau2, tau2_err]]}
        if maxR2 == R_value22:
            off = popt22[0]
            tauc = popt22[1]
            tau = popt22[2]
            tauc2 = popt22[3]
            tau2 = popt22[4]
            _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov22))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                pc = 1 - np.exp(-t_a/tauc)
                pc2 = 1 - np.exp(-t_a/tauc2)
                comp_1 = np.sum((amp_a - (off+num_pumps*pc*(np.exp(-2*t_a/tau)-
                    np.exp(-3*t_a/tau))))**2)
                comp_2 = np.sum((amp_a-(off+num_pumps*pc2*(np.exp(-2*t_a/tau2)-
                    np.exp(-3*t_a/tau2))))**2)
                if comp_1 <= comp_2:
                    return {'a':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {2: [[tauc, tauc_err, tau, tau_err],
                [tauc2, tauc2_err, tau2, tau2_err]]}

def trap_fit(scheme, amps, times, num_pumps, fit_thresh, tau_min, tau_max,
    tauc_min, tauc_max, offset_min, offset_max, both_a = None):
    """For a given temperature, scheme, and pixel, this function examines data
    for amplitude vs phase time and fits for release time constant (tau) and
    the probability for capture (pc).  It tries fitting for a single trap in
    the pixel, and if the goodness of fit is not high enough (if less than
    fit_thresh), then the function attempts to fit for two traps in the pixel.
    The exception is the case where a 'both' type pixel is input; then only a
    two-trap fit is attempted.  The function assumes the full time-dependent
    form for capture probability.
    Parameters
    ----------
    scheme : int, {1, 2, 3, 4}
        The scheme under consideration.  Only certain probability functions are
        valid for different schemes.
    amps : array
        Amplitudes of bright pixel of the dipole Units: e-.
    times : array
        Phase times in same order as amps.  Units:  seconds.
    num_pumps : int, > 0
        Number of cycles for trap pumping.
    fit_thresh : (0, 1)
        The minimum value required for adjusted coefficient of determination
        (adjusted R^2) for curve fitting for the release time constant
        (tau) using data for dipole amplitude vs phase time.  The closer to 1,
        the better the fit. Must be between 0 and 1.
    tau_min : float, >= 0
        Lower bound value for tau (release time constant) for curve fitting,
        in seconds.
    tau_max : float, > tau_min
        Upper bound value for tau (release time constant) for curve fitting,
        in seconds.
    tauc_min : float, >= 0
        Lower bound value for tauc (capture time constant) for curve fitting,
        in e-.
    tauc_max : float, > tau_min
        Upper bound value for tauc (capture time constant) for curve fitting,
        in e-.
    offset_min : float
        Lower bound value for the offset in the fitting of data for amplitude
        vs phase time.  Acts as a nuisance parameter.  Units of e-.
    offset_max : float, > offset_min
        Upper bound value for the offset in the fitting of data for amplitude
        vs phase time.  Acts as a nuisance parameter.  Units of e-.
    both_a : None or dict, optional
        Use None if you are fitting for a dipole that is of the 'above' or
        'below' kind.  Use the dictionary corresponding to
        rc_both[(row,col)]['above'] if you are fitting for a dipole that is of
        the 'both' kind.  See doc string of trap_id() for more details on this
        dictionary.  Defaults to None.
    Returns
    -------
    Dictionary of fit data.  Uses the following:
        prob: {1, 2, 3}; denotes probability function that gave the best fit
        pc: constant capture probability, best-fit value
        pc_err: standard deviation of pc
        tau: release time constant, best-fit value
        tau_err: standard deviation of tau
        If both_a = None and one trap is the best fit,
        the return is of the following form:
        {prob: [[pc, pc_err, tau, tau_err]]}
        If both_a = None and two traps is the best fit,
        the return is of the following form:
        {prob1: [[pc, pc_err, tau, tau_err]],
        prob2: [[pc2, pc2_err, tau2, tau2_err]]}
        If both traps are of prob1 type, then the return is like this:
        {prob1: [[pc, pc_err, tau, tau_err], [pc2, pc2_err, tau2, tau2_err]]}
        If both_a is not None, then the return is of the following form:
        {type1:{1: [[pc, pc_err, tau, tau_err]]},
        type2:{2: [[pc2, pc2_err, tau2, tau2_err]]}}
        where type1 is either 'a' for above or 'b' for below, and type2 is
        whichever of those type1 is not.
    """
    # TODO time-dep pc and pc2:  assume constant charge packet throughout the
    # different phase times across all num_pumps, when in reality, the charge
    # packet changes a little with each pump (a la 'express=num_pumps' in
    # arcticpy).  The fit function would technically be a sum of num_pumps
    # terms in which tauc is a slightly different value in each.  So the value
    # of tauc may not be that useful; could compare its uncertainty to that of
    # the values in the literature.  But it's generally good for large charge
    # packet values (perhaps it nominally corresponds to the average between
    # the starting charge packet value for trap pumping and the ending value
    # read off the highest-amp phase time frame.

    # check inputs
    if scheme != 1 and scheme != 2 and scheme != 3 and scheme != 4:
        raise TypeError('scheme must be 1, 2, 3, or 4.')
    try:
        amps = np.array(amps).astype(float)
    except:
        raise TypeError("amps elements should be real numbers")
    check.oneD_array(amps, 'amps', TypeError)
    try:
        times = np.array(times).astype(float)
    except:
        raise TypeError("times elements should be real numbers")
    check.oneD_array(times, 'times', TypeError)
    if len(np.unique(times)) < 6:
        raise IndexError('times must have a number of unique phase times '
        'longer than the number of fitted parameters.')
    if len(amps) != len(times):
        raise ValueError("times and amps should have same number of elements")
    check.positive_scalar_integer(num_pumps, 'num_pumps', TypeError)
    check.real_nonnegative_scalar(fit_thresh, 'fit_thresh', TypeError)
    if fit_thresh > 1:
        raise ValueError('fit_thresh should fall in (0,1).')
    check.real_nonnegative_scalar(tau_min, 'tau_min', TypeError)
    check.real_nonnegative_scalar(tau_max, 'tau_max', TypeError)
    if tau_max <= tau_min:
        raise ValueError('tau_max must be > tau_min')
    check.real_nonnegative_scalar(tauc_min, 'tauc_min', TypeError)
    check.real_nonnegative_scalar(tauc_max, 'tauc_max', TypeError)
    if tauc_max <= tauc_min:
        raise ValueError('tauc_max must be > tauc_min')
    check.real_scalar(offset_min, 'offset_min', TypeError)
    check.real_scalar(offset_max, 'offset_max', TypeError)
    if offset_max <= offset_min:
        raise ValueError('offset_max must be > offset_min')
    if (type(both_a) is not dict) and (both_a is not None):
        raise TypeError('both_a must be a rc_both[(row,col)][\'above\'] '
        'dictionary or None.  See trap_id() doc string for more details.')
    if both_a is not None:
        try:
            x = len(both_a['amp'])
            y = len(both_a['t'])
        except:
            raise KeyError('both_a must contain \'amp\' and \'t\' keys.')
        if x != y:
            raise ValueError('both_a[\'amp\'] and both_a[\'t\'] must have the '
            'same number of elements')
    # if input for both_a doesn't conform to expected dictionary structure,
    # exceptions will be raised

    # define all these inside this function because they depend also on
    # num_pumps, and I can't specify an unfitted parameter in the function
    # definition if I want to use curve_fit
    def P1(time_data, offset, tauc, tau):
        """Probability function 1, one trap.
        """
        pc = 1 - np.exp(-time_data/tauc)
        return offset+(num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-2*time_data/tau)))

    def P1_P1(time_data, offset, tauc, tau, tauc2, tau2):
        """Probability function 1, two traps.
        """
        pc = 1 - np.exp(-time_data/tauc)
        pc2 = 1 - np.exp(-time_data/tauc2)
        return offset+num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-2*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-2*time_data/tau2))

    def P2(time_data, offset, tauc, tau):
        """Probability function 2, one trap.
        """
        pc = 1 - np.exp(-time_data/tauc)
        return offset+(num_pumps*pc*(np.exp(-2*time_data/tau)-
            np.exp(-3*time_data/tau)))

    def P1_P2(time_data, offset, tauc, tau, tauc2, tau2):
        """One trap for probability function 1, one for probability function 2.
        """
        pc = 1 - np.exp(-time_data/tauc)
        pc2 = 1 - np.exp(-time_data/tauc2)
        return offset+num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-2*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-2*time_data/tau2)-np.exp(-3*time_data/tau2))

    def P2_P2(time_data, offset, tauc, tau, tauc2, tau2):
        """Probability function 2, two traps.
        """
        pc = 1 - np.exp(-time_data/tauc)
        pc2 = 1 - np.exp(-time_data/tauc2)
        return offset+num_pumps*pc*(np.exp(-2*time_data/tau)-
            np.exp(-3*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-2*time_data/tau2)-np.exp(-3*time_data/tau2))

    def P3(time_data, offset, tauc, tau):
        """Probability function 3, one trap.
        """
        pc = 1 - np.exp(-time_data/tauc)
        return offset+(num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-4*time_data/tau)))

    def P3_P3(time_data, offset, tauc, tau, tauc2, tau2):
        """Probability function 3, two traps.
        """
        pc = 1 - np.exp(-time_data/tauc)
        pc2 = 1 - np.exp(-time_data/tauc2)
        return offset+num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-4*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-4*time_data/tau2))

    def P2_P3(time_data, offset, tauc, tau, tauc2, tau2):
        """One trap for probability function 2, one for probability function 3.
        """
        pc = 1 - np.exp(-time_data/tauc)
        pc2 = 1 - np.exp(-time_data/tauc2)
        return offset+num_pumps*pc*(np.exp(-2*time_data/tau)-
            np.exp(-3*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-4*time_data/tau2))

    #upper bound for tauc: 1*eperdn, but our knowledge of eperdn may have
    # error.

    # Makes sense that you wouldn't find traps at times far away from time
    # constant, so to avoid false good fits, restrict bounds between 10^-6 and
    # 10^-2
    # in order, for one trap:  offset, tauc, tau
    l_bounds_one = [offset_min, tauc_min, tau_min]
    u_bounds_one = [offset_max, tauc_max, tau_max]
    # avoid initial guess of 0
    offset0 = (offset_min+offset_max)/2
    if offset0 == 0:
        offset0 = min(1 + (offset_min+offset_max)/2, offset_max)
    offset0l = offset_min
    if offset0l == 0:
        offset0l = min(offset0l+1, offset_max)
    offset0u = offset_max
    if offset0u == 0:
        offset0u = max(offset0u-1, offset_min)
    # tauc0 = 1e-9
    # if tauc0 < tauc_min or tauc0 > tauc_max:
    tauc0 = (tauc_min+tauc_max)/2
    tauc0l = tauc_min
    if tauc0l == 0:
        tauc0l = min(tauc0l+1e-12, tauc_max)
    tauc0u = tauc_max
    tau0 = (tau_min + tau_max)/2
    tau0l = tau_min
    if tau0l == 0:
        tau0l = min(tau0l+1e-7, tau_max)
    tau0u = tau_max
    # tau0 = np.median(times)
    # if tau_min > tau0 or tau_max < tau0:
    #     tau0 = (tau_min+tau_max)/2
    # start search from biggest time since these data points more
    # spread out if phase times are taken evenly spaced in log space; don't
    # want to give too much weight to the bunched-up early times and lock in
    # an answer too early in the curve search, so start at other end.
    # Try search from earliest time and biggest time, and see which gives a
    # bigger adj R^2 value
    p01l = [offset0, tauc0, tau0l]
    p01u = [offset0, tauc0, tau0u]
    #l_bounds_one = [-100000, 0, 0]
    #u_bounds_one = [100000, 1, 1]
    bounds_one = (l_bounds_one, u_bounds_one)
    # in order, for two traps:  offset, tauc, tau, tauc2, tau2
    l_bounds_two = [offset_min, tauc_min, tau_min, tauc_min, tau_min]
    u_bounds_two = [offset_max, tauc_max, tau_max, tauc_max, tau_max]
    # p02l = [offset0l, k_min, tauc_min, tau0l, tauc_min, tau0l]
    # p02u = [offset0u, k_max, tauc_max, tau0u, tauc_max, tau0u]
    p02l = [offset0, tauc0u, tau0l, tauc0l, tau0u]
    p02u = [offset0, tauc0l, tau0u, tauc0u, tau0l]
    # p02l = [offset0, tauc0, tau0l, tauc0, tau0u]
    # p02u = [offset0, tauc0, tau0u, tauc0, tau0l]
    #l_bounds_two = [-100000, 0, 0, 0, 0]
    #u_bounds_two = [100000, 1, 1, 1, 1]
    bounds_two = (l_bounds_two, u_bounds_two)

    # same for every curve fit (sum of the squared difference total)
    sstot = np.sum((amps - np.mean(amps))**2)

    if scheme == 1 or scheme == 2:

        if both_a == None:
            # attempt all possible probability functions
            try:
                popt1l, pcov1l = curve_fit(P1, times, amps, bounds=bounds_one,
                p0 = p01l, maxfev = np.inf)#, sigma = 0.1*amps)
                popt1u, pcov1u = curve_fit(P1, times, amps, bounds=bounds_one,
                p0 = p01u, maxfev = np.inf)#, sigma = 0.1*amps)
            except:
                warnings.warn('curve_fit failed')
                return None
            fit1l = P1(times, popt1l[0], popt1l[1], popt1l[2])
            fit1u = P1(times, popt1u[0], popt1u[1], popt1u[2])
            ssres1l = np.sum((fit1l - amps)**2)
            ssres1u = np.sum((fit1u - amps)**2)
            # coefficient of determination, adjusted R^2:
            # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
            R_value1l = 1 - (ssres1l/sstot)*(len(times) - 1)/(len(times) -
                len(popt1l))
            R_value1u = 1 - (ssres1u/sstot)*(len(times) - 1)/(len(times) -
                len(popt1u))
            R_value1 = max(R_value1l, R_value1u)
            if R_value1 == R_value1l:
                popt1 = popt1l; pcov1 = pcov1l
            if R_value1 == R_value1u:
                popt1 = popt1u; pcov1 = pcov1u

            try:
                popt2l, pcov2l = curve_fit(P2, times, amps, bounds=bounds_one,
                p0 = p01l, maxfev = np.inf)#, sigma = 0.1*amps)
                popt2u, pcov2u = curve_fit(P2, times, amps, bounds=bounds_one,
                p0 = p01u, maxfev = np.inf)#, sigma = 0.1*amps)
            except:
                warnings.warn('curve_fit failed')
                return None
            fit2l = P2(times, popt2l[0], popt2l[1], popt2l[2])
            fit2u = P2(times, popt2u[0], popt2u[1], popt2u[2])
            ssres2l = np.sum((fit2l - amps)**2)
            ssres2u = np.sum((fit2u - amps)**2)
            # coefficient of determination, adjusted R^2:
            # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
            R_value2l = 1 - (ssres2l/sstot)*(len(times) - 1)/(len(times) -
                len(popt2l))
            R_value2u = 1 - (ssres2u/sstot)*(len(times) - 1)/(len(times) -
                len(popt2u))
            R_value2 = max(R_value2l, R_value2u)
            if R_value2 == R_value2l:
                popt2 = popt2l; pcov2 = pcov2l
            if R_value2 == R_value2u:
                popt2 = popt2u; pcov2 = pcov2u

            # accept the best fit and require threshold met
            maxR1 = max(R_value1, R_value2)

            if maxR1 >= fit_thresh and maxR1 == R_value1:
                tauc = popt1[1]
                tau = popt1[2]
                _, tauc_err, tau_err  = np.sqrt(np.diag(pcov1))
                return {1: [[tauc, tauc_err, tau, tau_err]]}
            if maxR1 >= fit_thresh and maxR1 == R_value2:
                tauc = popt2[1]
                tau = popt2[2]
                _, tauc_err, tau_err  = np.sqrt(np.diag(pcov2))
                return {2: [[tauc, tauc_err, tau, tau_err]]}

        # maxR1 must have been below fit_thresh.  Now try 2 traps

        try:
            popt11l, pcov11l = curve_fit(P1_P1, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt11u, pcov11u = curve_fit(P1_P1, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit11l = P1_P1(times, popt11l[0], popt11l[1], popt11l[2], popt11l[3],
        popt11l[4])
        fit11u = P1_P1(times, popt11u[0], popt11u[1], popt11u[2], popt11u[3],
        popt11u[4])
        ssres11l = np.sum((fit11l - amps)**2)
        ssres11u = np.sum((fit11u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value11l = 1 - (ssres11l/sstot)*(len(times) - 1)/(len(times) -
            len(popt11l))
        R_value11u = 1 - (ssres11u/sstot)*(len(times) - 1)/(len(times) -
            len(popt11u))
        R_value11 = max(R_value11l, R_value11u)
        if R_value11 == R_value11l:
            popt11 = popt11l; pcov11 = pcov11l
        if R_value11 == R_value11u:
            popt11 = popt11u; pcov11 = pcov11u

        try:
            popt12l, pcov12l = curve_fit(P1_P2, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt12u, pcov12u = curve_fit(P1_P2, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit12l = P1_P2(times, popt12l[0], popt12l[1], popt12l[2], popt12l[3],
        popt12l[4])
        fit12u = P1_P2(times, popt12u[0], popt12u[1], popt12u[2], popt12u[3],
        popt12u[4])
        ssres12l = np.sum((fit12l - amps)**2)
        ssres12u = np.sum((fit12u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value12l = 1 - (ssres12l/sstot)*(len(times) - 1)/(len(times) -
            len(popt12l))
        R_value12u = 1 - (ssres12u/sstot)*(len(times) - 1)/(len(times) -
            len(popt12u))
        R_value12 = max(R_value12l, R_value12u)
        if R_value12 == R_value12l:
            popt12 = popt12l; pcov12 = pcov12l
        if R_value12 == R_value12u:
            popt12 = popt12u; pcov12 = pcov12u

        try:
            popt22l, pcov22l = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt22u, pcov22u = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit22l = P2_P2(times, popt22l[0], popt22l[1], popt22l[2], popt22l[3],
        popt22l[4])
        fit22u = P2_P2(times, popt22u[0], popt22u[1], popt22u[2], popt22u[3],
        popt22u[4])
        ssres22l = np.sum((fit22l - amps)**2)
        ssres22u = np.sum((fit22u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value22l = 1 - (ssres22l/sstot)*(len(times) - 1)/(len(times) -
            len(popt22l))
        R_value22u = 1 - (ssres22u/sstot)*(len(times) - 1)/(len(times) -
            len(popt22u))
        R_value22 = max(R_value22l, R_value22u)
        if R_value22 == R_value22l:
            popt22 = popt22l; pcov22 = pcov22l
        if R_value22 == R_value22u:
            popt22 = popt22u; pcov22 = pcov22u

        maxR2 = max(R_value11, R_value12, R_value22)
        if maxR2 < fit_thresh:
            warnings.warn('No curve fit gave adjusted R^2 value above '
            'fit_thresh')
            return None

        if maxR2 == R_value11:
            off = popt11[0]
            tauc = popt11[1]
            tau = popt11[2]
            tauc2 = popt11[3]
            tau2 = popt11[4]
            _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov11))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                # TODO v2, for trap_fit and trap_fit_const:
                # if 'amp' list for 'above' is identical to 'amp'
                #list for 'below', then earmark for assignment whenever
                # finding optimal matchings of schemes for sub-el loc; if
                #no particular assignment is better at that point, then the
                #assignment doesn't matter
                max_a_ind = np.where(amp_a == np.max(amp_a))
                #P1 for tau
                a_tau = t_a[max_a_ind[0]]/np.log(2)
                #P1 for tau2
                a_tau2 = t_a[max_a_ind[0]]/np.log(2)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{1: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{1: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{1: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{1: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {1: [[tauc, tauc_err, tau, tau_err],
                [tauc2, tauc2_err, tau2, tau2_err]]}

        if maxR2 == R_value12:
            off = popt12[0]
            tauc = popt12[1]
            tau = popt12[2]
            tauc2 = popt12[3]
            tau2 = popt12[4]
            _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov12))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                max_a_ind = np.where(amp_a == np.max(amp_a))
                #P1 for tau
                a_tau = t_a[max_a_ind[0]]/np.log(2)
                #P2 for tau2
                a_tau2 = t_a[max_a_ind[0]]/np.log(3/2)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{1: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{1: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {1: [[tauc, tauc_err, tau, tau_err]],
                2: [[tauc2, tauc2_err, tau2, tau2_err]]}

        if maxR2 == R_value22:
            off = popt22[0]
            tauc = popt22[1]
            tau = popt22[2]
            tauc2 = popt22[3]
            tau2 = popt22[4]
            _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov22))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                max_a_ind = np.where(amp_a == np.max(amp_a))
                #P2 for tau
                a_tau = t_a[max_a_ind[0]]/np.log(3/2)
                #P2 for tau2
                a_tau2 = t_a[max_a_ind[0]]/np.log(3/2)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {2: [[tauc, tauc_err, tau, tau_err],
                [tauc2, tauc2_err, tau2, tau2_err]]}

    if scheme == 3 or scheme == 4:

        if both_a == None:
            #attempt both probability functions
            try:
                popt3l, pcov3l = curve_fit(P3, times, amps, bounds=bounds_one,
                p0 = p01l, maxfev = np.inf)#, sigma = 0.1*amps)
                popt3u, pcov3u = curve_fit(P3, times, amps, bounds=bounds_one,
                p0 = p01u, maxfev = np.inf)#, sigma = 0.1*amps)
            except:
                warnings.warn('curve_fit failed')
                return None
            fit3l = P3(times, popt3l[0], popt3l[1], popt3l[2])
            fit3u = P3(times, popt3u[0], popt3u[1], popt3u[2])
            ssres3l = np.sum((fit3l - amps)**2)
            ssres3u = np.sum((fit3u - amps)**2)
            # coefficient of determination, adjusted R^2:
            # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
            R_value3l = 1 - (ssres3l/sstot)*(len(times) - 1)/(len(times) -
                len(popt3l))
            R_value3u = 1 - (ssres3u/sstot)*(len(times) - 1)/(len(times) -
                len(popt3u))
            R_value3 = max(R_value3l, R_value3u)
            if R_value3 == R_value3l:
                popt3 = popt3l; pcov3 = pcov3l
            if R_value3 == R_value3u:
                popt3 = popt3u; pcov3 = pcov3u

            try:
                popt2l, pcov2l = curve_fit(P2, times, amps, bounds=bounds_one,
                p0 = p01l, maxfev = np.inf)#, sigma = 0.1*amps)
                popt2u, pcov2u = curve_fit(P2, times, amps, bounds=bounds_one,
                p0 = p01u, maxfev = np.inf)#, sigma = 0.1*amps)
            except:
                warnings.warn('curve_fit failed')
                return None
            fit2l = P2(times, popt2l[0], popt2l[1], popt2l[2])
            fit2u = P2(times, popt2u[0], popt2u[1], popt2u[2])
            ssres2l = np.sum((fit2l - amps)**2)
            ssres2u = np.sum((fit2u - amps)**2)
            # coefficient of determination, adjusted R^2:
            # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
            R_value2l = 1 - (ssres2l/sstot)*(len(times) - 1)/(len(times) -
                len(popt2l))
            R_value2u = 1 - (ssres2u/sstot)*(len(times) - 1)/(len(times) -
                len(popt2u))
            R_value2 = max(R_value2l, R_value2u)
            if R_value2 == R_value2l:
                popt2 = popt2l; pcov2 = pcov2l
            if R_value2 == R_value2u:
                popt2 = popt2u; pcov2 = pcov2u

            # accept the best fit and require threshold met
            maxR1 = max(R_value3, R_value2)

            if maxR1 >= fit_thresh and maxR1 == R_value3:
                tauc = popt3[1]
                tau = popt3[2]
                _, tauc_err, tau_err  = np.sqrt(np.diag(pcov3))
                return {3: [[tauc, tauc_err, tau, tau_err]]}
            if maxR1 >= fit_thresh and maxR1 == R_value2:
                tauc = popt2[1]
                tau = popt2[2]
                _, tauc_err, tau_err  = np.sqrt(np.diag(pcov2))
                return {2: [[tauc, tauc_err, tau, tau_err]]}

        # maxR1 must have been below fit_thresh.  Now try 2 traps

        try:
            popt33l, pcov33l = curve_fit(P3_P3, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt33u, pcov33u = curve_fit(P3_P3, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit33l = P3_P3(times, popt33l[0], popt33l[1], popt33l[2], popt33l[3],
        popt33l[4])
        fit33u = P3_P3(times, popt33u[0], popt33u[1], popt33u[2], popt33u[3],
        popt33u[4])
        ssres33l = np.sum((fit33l - amps)**2)
        ssres33u = np.sum((fit33u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value33l = 1 - (ssres33l/sstot)*(len(times) - 1)/(len(times) -
            len(popt33l))
        R_value33u = 1 - (ssres33u/sstot)*(len(times) - 1)/(len(times) -
            len(popt33u))
        R_value33 = max(R_value33l, R_value33u)
        if R_value33 == R_value33l:
            popt33 = popt33l; pcov33 = pcov33l
        if R_value33 == R_value33u:
            popt33 = popt33u; pcov33 = pcov33u

        try:
            popt23l, pcov23l = curve_fit(P2_P3, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt23u, pcov23u = curve_fit(P2_P3, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit23l = P2_P3(times, popt23l[0], popt23l[1], popt23l[2], popt23l[3],
        popt23l[4])
        fit23u = P2_P3(times, popt23u[0], popt23u[1], popt23u[2], popt23u[3],
        popt23u[4])
        ssres23l = np.sum((fit23l - amps)**2)
        ssres23u = np.sum((fit23u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value23l = 1 - (ssres23l/sstot)*(len(times) - 1)/(len(times) -
            len(popt23l))
        R_value23u = 1 - (ssres23u/sstot)*(len(times) - 1)/(len(times) -
            len(popt23u))
        R_value23 = max(R_value23l, R_value23u)
        if R_value23 == R_value23l:
            popt23 = popt23l; pcov23 = pcov23l
        if R_value23 == R_value23u:
            popt23 = popt23u; pcov23 = pcov23u

        try:
            popt22l, pcov22l = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt22u, pcov22u = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit22l = P2_P2(times, popt22l[0], popt22l[1], popt22l[2], popt22l[3],
        popt22l[4])
        fit22u = P2_P2(times, popt22u[0], popt22u[1], popt22u[2], popt22u[3],
        popt22u[4])
        ssres22l = np.sum((fit22l - amps)**2)
        ssres22u = np.sum((fit22u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value22l = 1 - (ssres22l/sstot)*(len(times) - 1)/(len(times) -
            len(popt22l))
        R_value22u = 1 - (ssres22u/sstot)*(len(times) - 1)/(len(times) -
            len(popt22u))
        R_value22 = max(R_value22l, R_value22u)
        if R_value22 == R_value22l:
            popt22 = popt22l; pcov22 = pcov22l
        if R_value22 == R_value22u:
            popt22 = popt22u; pcov22 = pcov22u

        maxR2 = max(R_value33, R_value23, R_value22)

        if maxR2 < fit_thresh:
            warnings.warn('No curve fit gave adjusted R^2 value above '
            'fit_thresh')
            return None

        if maxR2 == R_value33:
            off = popt33[0]
            tauc = popt33[1]
            tau = popt33[2]
            tauc2 = popt33[3]
            tau2 = popt33[4]
            _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov33))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                max_a_ind = np.where(amp_a == np.max(amp_a))
                #P3 for tau
                a_tau = t_a[max_a_ind[0]]/(2*np.log(2)/3)
                #P3 for tau2
                a_tau2 = t_a[max_a_ind[0]]/(2*np.log(2)/3)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{3: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{3: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{3: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{3: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {3: [[tauc, tauc_err, tau, tau_err], [tauc2, tauc2_err,
                tau2, tau2_err]]}

        if maxR2 == R_value23:
            off = popt23[0]
            tauc = popt23[1]
            tau = popt23[2]
            tauc2 = popt23[3]
            tau2 = popt23[4]
            _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov23))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                max_a_ind = np.where(amp_a == np.max(amp_a))
                #P2 for tau
                a_tau = t_a[max_a_ind[0]]/np.log(3/2)
                #P3 for tau2
                a_tau2 = t_a[max_a_ind[0]]/(2*np.log(2)/3)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{3: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{3: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {2: [[tauc, tauc_err, tau, tau_err]],
                3: [[tauc2, tauc2_err, tau2, tau2_err]]}

        if maxR2 == R_value22:
            off = popt22[0]
            tauc = popt22[1]
            tau = popt22[2]
            tauc2 = popt22[3]
            tau2 = popt22[4]
            _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov22))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                max_a_ind = np.where(amp_a == np.max(amp_a))
                #P2 for tau
                a_tau = t_a[max_a_ind[0]]/np.log(3/2)
                #P2 for tau2
                a_tau2 = t_a[max_a_ind[0]]/np.log(3/2)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {2: [[tauc, tauc_err, tau, tau_err],
                [tauc2, tauc2_err, tau2, tau2_err]]}

def trap_fit_const(scheme, amps, times, num_pumps, fit_thresh, tau_min,
    tau_max, pc_min, pc_max, offset_min, offset_max, both_a = None):
    """For a given temperature, scheme, and pixel, this function examines data
    for amplitude vs phase time and fits for release time constant (tau) and
    the probability for capture (pc).  It tries fitting for a single trap in
    the pixel, and if the goodness of fit is not high enough (if less than
    fit_thresh), then the function attempts to fit for two traps in the pixel.
    The exception is the case where a 'both' type pixel is input; then only a
    two-trap fit is attempted.  The function assumes a constant for pc rather
    than its actual time-dependent form.
    Parameters
    ----------
    scheme : int, {1, 2, 3, 4}
        The scheme under consideration.  Only certain probability functions are
        valid for different schemes.
    amps : array
        Amplitudes of bright pixel of the dipole Units: e-.
    times : array
        Phase times in same order as amps.  Units:  seconds.
    num_pumps : int, > 0
        Number of cycles for trap pumping.
    fit_thresh : (0, 1)
        The minimum value required for adjusted coefficient of determination
        (adjusted R^2) for curve fitting for the release time constant
        (tau) using data for dipole amplitude vs phase time.  The closer to 1,
        the better the fit. Must be between 0 and 1.
    tau_min : float, >= 0
        Lower bound value for tau (release time constant) for curve fitting,
        in seconds.
    tau_max : float, > tau_min
        Upper bound value for tau (release time constant) for curve fitting,
        in seconds.
    pc_min : float, >= 0
        Lower bound value for pc (capture probability) for curve fitting,
        in e-.
    pc_max : float, > pc_min
        Upper bound value for pc (capture probability) for curve fitting,
        in e-.
    offset_min : float
        Lower bound value for the offset in the fitting of data for amplitude
        vs phase time.  Acts as a nuisance parameter.  Units of e-.
    offset_max : float, > offset_min
        Upper bound value for the offset in the fitting of data for amplitude
        vs phase time.  Acts as a nuisance parameter.  Units of e-.
    both_a : None or dict, optional
        Use None if you are fitting for a dipole that is of the 'above' or
        'below' kind.  Use the dictionary corresponding to
        rc_both[(row,col)]['above'] if you are fitting for a dipole that is of
        the 'both' kind.  See doc string of trap_id() for more details on this
        dictionary.  Defaults to None.
    Returns
    -------
    Dictionary of fit data.  Uses the following:
        prob: {1, 2, 3}; denotes probability function that gave the best fit
        pc: constant capture probability, best-fit value
        pc_err: standard deviation of pc
        tau: release time constant, best-fit value
        tau_err: standard deviation of tau
        If both_a = None and one trap is the best fit,
        the return is of the following form:
        {prob: [[pc, pc_err, tau, tau_err]]}
        If both_a = None and two traps is the best fit,
        the return is of the following form:
        {prob1: [[pc, pc_err, tau, tau_err]],
        prob2: [[pc2, pc2_err, tau2, tau2_err]]}
        If both traps are of prob1 type, then the return is like this:
        {prob1: [[pc, pc_err, tau, tau_err], [pc2, pc2_err, tau2, tau2_err]]}
        If both_a is not None, then the return is of the following form:
        {type1:{1: [[pc, pc_err, tau, tau_err]]},
        type2:{2: [[pc2, pc2_err, tau2, tau2_err]]}}
        where type1 is either 'a' for above or 'b' for below, and type2 is
        whichever of those type1 is not.
    """
    # TODO time-dep pc and pc2:  assume constant charge packet throughout the
    # different phase times across all num_pumps, when in reality, the charge
    # packet changes a little with each pump (a la 'express=num_pumps' in
    # arcticpy).  The fit function would technically be a sum of num_pumps
    # terms in which tauc is a slightly different value in each.  So the value
    # of tauc may not be that useful; could compare its uncertainty to that of
    # the values in the literature.  But it's generally good for large charge
    # packet values (perhaps it nominally corresponds to the average between
    # the starting charge packet value for trap pumping and the ending value
    # read off the highest-amp phase time frame.

    # check inputs
    if scheme != 1 and scheme != 2 and scheme != 3 and scheme != 4:
        raise TypeError('scheme must be 1, 2, 3, or 4.')
    try:
        amps = np.array(amps).astype(float)
    except:
        raise TypeError("amps elements should be real numbers")
    check.oneD_array(amps, 'amps', TypeError)
    try:
        times = np.array(times).astype(float)
    except:
        raise TypeError("times elements should be real numbers")
    check.oneD_array(times, 'times', TypeError)
    if len(np.unique(times)) < 6:
        raise IndexError('times must have a number of unique phase times '
        'longer than the number of fitted parameters.')
    if len(amps) != len(times):
        raise ValueError("times and amps should have same number of elements")
    check.positive_scalar_integer(num_pumps, 'num_pumps', TypeError)
    check.real_nonnegative_scalar(fit_thresh, 'fit_thresh', TypeError)
    if fit_thresh > 1:
        raise ValueError('fit_thresh should fall in (0,1).')
    check.real_nonnegative_scalar(tau_min, 'tau_min', TypeError)
    check.real_nonnegative_scalar(tau_max, 'tau_max', TypeError)
    if tau_max <= tau_min:
        raise ValueError('tau_max must be > tau_min')
    check.real_nonnegative_scalar(pc_min, 'pc_min', TypeError)
    check.real_nonnegative_scalar(pc_max, 'pc_max', TypeError)
    if pc_max <= pc_min:
        raise ValueError('pc_max must be > pc_min')
    check.real_scalar(offset_min, 'offset_min', TypeError)
    check.real_scalar(offset_max, 'offset_max', TypeError)
    if offset_max <= offset_min:
        raise ValueError('offset_max must be > offset_min')
    if (type(both_a) is not dict) and (both_a is not None):
        raise TypeError('both_a must be a rc_both[(row,col)][\'above\'] '
        'dictionary.  See trap_id() doc string for more details.')
    if both_a is not None:
        try:
            x = len(both_a['amp'])
            y = len(both_a['t'])
        except:
            raise KeyError('both_a must contain \'amp\' and \'t\' keys.')
        if x != y:
            raise ValueError('both_a[\'amp\'] and both_a[\'t\'] must have the '
            'same number of elements')
    # if both_a doesn't have expected keys, exceptions will be raised

    # define all these inside this function because they depend also on
    # num_pumps, and I can't specify an unfitted parameter in the function
    # definition if I want to use curve_fit
    def P1(time_data, offset, pc, tau):
        """Probability function 1, one trap.
        """
        return offset+(num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-2*time_data/tau)))

    def P1_P1(time_data, offset, pc, tau, pc2, tau2):
        """Probability function 1, two traps.
        """
        return offset+num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-2*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-2*time_data/tau2))

    def P2(time_data, offset, pc, tau):
        """Probability function 2, one trap.
        """
        return offset+(num_pumps*pc*(np.exp(-2*time_data/tau)-
            np.exp(-3*time_data/tau)))

    def P1_P2(time_data, offset, pc, tau, pc2, tau2):
        """One trap for probability function 1, one for probability function 2.
        """
        return offset+num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-2*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-2*time_data/tau2)-np.exp(-3*time_data/tau2))

    def P2_P2(time_data, offset, pc, tau, pc2, tau2):
        """Probability function 2, two traps.
        """
        return offset+num_pumps*pc*(np.exp(-2*time_data/tau)-
            np.exp(-3*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-2*time_data/tau2)-np.exp(-3*time_data/tau2))

    def P3(time_data, offset, pc, tau):
        """Probability function 3, one trap.
        """
        return offset+(num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-4*time_data/tau)))

    def P3_P3(time_data, offset, pc, tau, pc2, tau2):
        """Probability function 3, two traps.
        """
        return offset+num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-4*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-4*time_data/tau2))

    def P2_P3(time_data, offset, pc, tau, pc2, tau2):
        """One trap for probability function 2, one for probability function 3.
        """
        return offset+num_pumps*pc*(np.exp(-2*time_data/tau)-
            np.exp(-3*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-4*time_data/tau2))

    #upper bound for pc: 1*eperdn, but our knowledge of eperdn may have error.

    # Makes sense that you wouldn't find traps at times far away from time
    # constant, so to avoid false good fits, restrict bounds between 10^-6 and
    # 10^-2
    # in order, for one trap:  offset, pc, tau
    l_bounds_one = [offset_min, pc_min, tau_min]
    u_bounds_one = [offset_max, pc_max, tau_max]
    # avoid initial guess of 0
    offset0 = (offset_min+offset_max)/2
    if (offset_min+offset_max)/2 == 0:
        offset0 = min(1 + (offset_min+offset_max)/2, offset_max)
    offset0l = offset_min
    if offset0l == 0:
        offset0l = min(offset0l+1, offset_max)
    offset0u = offset_max
    if offset0u == 0:
        offset0u = max(offset0u-1, offset_min)
    pc0 = 1
    if 1 < pc_min or 1 > pc_max:
        pc0 = pc_min
    tau0 = (tau_min + tau_max)/2
    tau0l = tau_min
    if tau0l == 0:
        tau0l = min(tau0l+1e-7, tau_max)
    tau0u = tau_max
    # tau0 = np.median(times)
    # if tau_min > tau0 or tau_max < tau0:
    #     tau0 = (tau_min+tau_max)/2
    # start search from biggest time since these data points more
    # spread out if phase times are taken evenly spaced in log space; don't
    # want to give too much weight to the bunched-up early times and lock in
    # an answer too early in the curve search, so start at other end.
    # Try search from earliest time and biggest time, and see which gives a
    # bigger adj R^2 value
    p01l = [offset0, pc0, tau0l]
    p01u = [offset0, pc0, tau0u]
    #l_bounds_one = [-100000, 0, 0]
    #u_bounds_one = [100000, 1, 1]
    bounds_one = (l_bounds_one, u_bounds_one)
    # in order, for two traps:  offset, tauc, tau, tauc2, tau2
    l_bounds_two = [offset_min, pc_min, tau_min, pc_min, tau_min]
    u_bounds_two = [offset_max, pc_max, tau_max, pc_max, tau_max]
    # p02l = [offset0l, k_min, tauc_min, tau0l, tauc_min, tau0l]
    # p02u = [offset0u, k_max, tauc_max, tau0u, tauc_max, tau0u]
    p02l = [offset0, pc0, tau0l, pc0, tau0u]
    p02u = [offset0, pc0, tau0u, pc0, tau0l]
    #l_bounds_two = [-100000, 0, 0, 0, 0]
    #u_bounds_two = [100000, 1, 1, 1, 1]
    bounds_two = (l_bounds_two, u_bounds_two)

    # same for every curve fit (sum of the squared difference total)
    sstot = np.sum((amps - np.mean(amps))**2)

    if scheme == 1 or scheme == 2:

        if both_a == None:
            # attempt all possible probability functions
            try:
                popt1l, pcov1l = curve_fit(P1, times, amps, bounds=bounds_one,
                p0 = p01l, maxfev = np.inf)#, sigma = 0.1*amps)
                popt1u, pcov1u = curve_fit(P1, times, amps, bounds=bounds_one,
                p0 = p01u, maxfev = np.inf)#, sigma = 0.1*amps)
            except:
                warnings.warn('curve_fit failed')
                return None
            fit1l = P1(times, popt1l[0], popt1l[1], popt1l[2])
            fit1u = P1(times, popt1u[0], popt1u[1], popt1u[2])
            ssres1l = np.sum((fit1l - amps)**2)
            ssres1u = np.sum((fit1u - amps)**2)
            # coefficient of determination, adjusted R^2:
            # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
            R_value1l = 1 - (ssres1l/sstot)*(len(times) - 1)/(len(times) -
                len(popt1l))
            R_value1u = 1 - (ssres1u/sstot)*(len(times) - 1)/(len(times) -
                len(popt1u))
            R_value1 = max(R_value1l, R_value1u)
            if R_value1 == R_value1l:
                popt1 = popt1l; pcov1 = pcov1l
            if R_value1 == R_value1u:
                popt1 = popt1u; pcov1 = pcov1u

            try:
                popt2l, pcov2l = curve_fit(P2, times, amps, bounds=bounds_one,
                p0 = p01l, maxfev = np.inf)#, sigma = 0.1*amps)
                popt2u, pcov2u = curve_fit(P2, times, amps, bounds=bounds_one,
                p0 = p01u, maxfev = np.inf)#, sigma = 0.1*amps)
            except:
                warnings.warn('curve_fit failed')
                return None
            fit2l = P2(times, popt2l[0], popt2l[1], popt2l[2])
            fit2u = P2(times, popt2u[0], popt2u[1], popt2u[2])
            ssres2l = np.sum((fit2l - amps)**2)
            ssres2u = np.sum((fit2u - amps)**2)
            # coefficient of determination, adjusted R^2:
            # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
            R_value2l = 1 - (ssres2l/sstot)*(len(times) - 1)/(len(times) -
                len(popt2l))
            R_value2u = 1 - (ssres2u/sstot)*(len(times) - 1)/(len(times) -
                len(popt2u))
            R_value2 = max(R_value2l, R_value2u)
            if R_value2 == R_value2l:
                popt2 = popt2l; pcov2 = pcov2l
            if R_value2 == R_value2u:
                popt2 = popt2u; pcov2 = pcov2u

            # accept the best fit and require threshold met
            maxR1 = max(R_value1, R_value2)

            if maxR1 >= fit_thresh and maxR1 == R_value1:
                pc = popt1[1]
                tau = popt1[2]
                _, pc_err, tau_err  = np.sqrt(np.diag(pcov1))
                return {1: [[pc, pc_err, tau, tau_err]]}
            if maxR1 >= fit_thresh and maxR1 == R_value2:
                pc = popt2[1]
                tau = popt2[2]
                _, pc_err, tau_err  = np.sqrt(np.diag(pcov2))
                return {2: [[pc, pc_err, tau, tau_err]]}

        # maxR1 must have been below fit_thresh.  Now try 2 traps

        try:
            popt11l, pcov11l = curve_fit(P1_P1, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt11u, pcov11u = curve_fit(P1_P1, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit11l = P1_P1(times, popt11l[0], popt11l[1], popt11l[2], popt11l[3],
        popt11l[4])
        fit11u = P1_P1(times, popt11u[0], popt11u[1], popt11u[2], popt11u[3],
        popt11u[4])
        ssres11l = np.sum((fit11l - amps)**2)
        ssres11u = np.sum((fit11u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value11l = 1 - (ssres11l/sstot)*(len(times) - 1)/(len(times) -
            len(popt11l))
        R_value11u = 1 - (ssres11u/sstot)*(len(times) - 1)/(len(times) -
            len(popt11u))
        R_value11 = max(R_value11l, R_value11u)
        if R_value11 == R_value11l:
            popt11 = popt11l; pcov11 = pcov11l
        if R_value11 == R_value11u:
            popt11 = popt11u; pcov11 = pcov11u

        try:
            popt12l, pcov12l = curve_fit(P1_P2, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt12u, pcov12u = curve_fit(P1_P2, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit12l = P1_P2(times, popt12l[0], popt12l[1], popt12l[2], popt12l[3],
        popt12l[4])
        fit12u = P1_P2(times, popt12u[0], popt12u[1], popt12u[2], popt12u[3],
        popt12u[4])
        ssres12l = np.sum((fit12l - amps)**2)
        ssres12u = np.sum((fit12u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value12l = 1 - (ssres12l/sstot)*(len(times) - 1)/(len(times) -
            len(popt12l))
        R_value12u = 1 - (ssres12u/sstot)*(len(times) - 1)/(len(times) -
            len(popt12u))
        R_value12 = max(R_value12l, R_value12u)
        if R_value12 == R_value12l:
            popt12 = popt12l; pcov12 = pcov12l
        if R_value12 == R_value12u:
            popt12 = popt12u; pcov12 = pcov12u

        try:
            popt22l, pcov22l = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt22u, pcov22u = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit22l = P2_P2(times, popt22l[0], popt22l[1], popt22l[2], popt22l[3],
        popt22l[4])
        fit22u = P2_P2(times, popt22u[0], popt22u[1], popt22u[2], popt22u[3],
        popt22u[4])
        ssres22l = np.sum((fit22l - amps)**2)
        ssres22u = np.sum((fit22u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value22l = 1 - (ssres22l/sstot)*(len(times) - 1)/(len(times) -
            len(popt22l))
        R_value22u = 1 - (ssres22u/sstot)*(len(times) - 1)/(len(times) -
            len(popt22u))
        R_value22 = max(R_value22l, R_value22u)
        if R_value22 == R_value22l:
            popt22 = popt22l; pcov22 = pcov22l
        if R_value22 == R_value22u:
            popt22 = popt22u; pcov22 = pcov22u

        maxR2 = max(R_value11, R_value12, R_value22)
        if maxR2 < fit_thresh:
            warnings.warn('No curve fit gave adjusted R^2 value above '
            'fit_thresh')
            return None

        if maxR2 == R_value11:
            off = popt11[0]
            pc = popt11[1]
            tau = popt11[2]
            pc2 = popt11[3]
            tau2 = popt11[4]
            _, pc_err, tau_err, pc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov11))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                # TODO v2, for trap_fit and trap_fit_const:
                # if 'amp' list for 'above' is identical to 'amp'
                #list for 'below', then earmark for assignment whenever
                # finding optimal matchings of schemes for sub-el loc; if
                #no particular assignment is better at that point, then the
                #assignment doesn't matter
                #TODO v2, for trap_fit and trap_fit_const:
                # if picking peak amp phase time to match with the closer tau
                # value isn't good enough (not decipherable if not enough data
                # points to make it out), then I could look at the
                # complementary probability functions in the corresponding
                #deficit pixels and curve fit those
                max_a_ind = np.where(amp_a == np.max(amp_a))
                #P1 for tau
                a_tau = t_a[max_a_ind[0]]/np.log(2)
                #P1 for tau2
                a_tau2 = t_a[max_a_ind[0]]/np.log(2)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{1: [[pc, pc_err, tau, tau_err]]},
                        'b':{1: [[pc2, pc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{1: [[pc, pc_err, tau, tau_err]]},
                        'a':{1: [[pc2, pc2_err, tau2, tau2_err]]}}
            return {1: [[pc, pc_err, tau, tau_err],
                [pc2, pc2_err, tau2, tau2_err]]}

        if maxR2 == R_value12:
            off = popt12[0]
            pc = popt12[1]
            tau = popt12[2]
            pc2 = popt12[3]
            tau2 = popt12[4]
            _, pc_err, tau_err, pc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov12))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                max_a_ind = np.where(amp_a == np.max(amp_a))
                #P1 for tau
                a_tau = t_a[max_a_ind[0]]/np.log(2)
                #P2 for tau2
                a_tau2 = t_a[max_a_ind[0]]/np.log(3/2)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{1: [[pc, pc_err, tau, tau_err]]},
                        'b':{2: [[pc2, pc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{1: [[pc, pc_err, tau, tau_err]]},
                        'a':{2: [[pc2, pc2_err, tau2, tau2_err]]}}
            return {1: [[pc, pc_err, tau, tau_err]],
                2: [[pc2, pc2_err, tau2, tau2_err]]}

        if maxR2 == R_value22:
            off = popt22[0]
            pc = popt22[1]
            tau = popt22[2]
            pc2 = popt22[3]
            tau2 = popt22[4]
            _, pc_err, tau_err, pc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov22))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                max_a_ind = np.where(amp_a == np.max(amp_a))
                #P2 for tau
                a_tau = t_a[max_a_ind[0]]/np.log(3/2)
                #P2 for tau2
                a_tau2 = t_a[max_a_ind[0]]/np.log(3/2)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{2: [[pc, pc_err, tau, tau_err]]},
                        'b':{2: [[pc2, pc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{2: [[pc, pc_err, tau, tau_err]]},
                        'a':{2: [[pc2, pc2_err, tau2, tau2_err]]}}
            return {2: [[pc, pc_err, tau, tau_err],
                [pc2, pc2_err, tau2, tau2_err]]}

    if scheme == 3 or scheme == 4:

        if both_a == None:
            #attempt both probability functions
            try:
                popt3l, pcov3l = curve_fit(P3, times, amps, bounds=bounds_one,
                p0 = p01l, maxfev = np.inf)#, sigma = 0.1*amps)
                popt3u, pcov3u = curve_fit(P3, times, amps, bounds=bounds_one,
                p0 = p01u, maxfev = np.inf)#, sigma = 0.1*amps)
            except:
                warnings.warn('curve_fit failed')
                return None
            fit3l = P3(times, popt3l[0], popt3l[1], popt3l[2])
            fit3u = P3(times, popt3u[0], popt3u[1], popt3u[2])
            ssres3l = np.sum((fit3l - amps)**2)
            ssres3u = np.sum((fit3u - amps)**2)
            # coefficient of determination, adjusted R^2:
            # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
            R_value3l = 1 - (ssres3l/sstot)*(len(times) - 1)/(len(times) -
                len(popt3l))
            R_value3u = 1 - (ssres3u/sstot)*(len(times) - 1)/(len(times) -
                len(popt3u))
            R_value3 = max(R_value3l, R_value3u)
            if R_value3 == R_value3l:
                popt3 = popt3l; pcov3 = pcov3l
            if R_value3 == R_value3u:
                popt3 = popt3u; pcov3 = pcov3u

            try:
                popt2l, pcov2l = curve_fit(P2, times, amps, bounds=bounds_one,
                p0 = p01l, maxfev = np.inf)#, sigma = 0.1*amps)
                popt2u, pcov2u = curve_fit(P2, times, amps, bounds=bounds_one,
                p0 = p01u, maxfev = np.inf)#, sigma = 0.1*amps)
            except:
                warnings.warn('curve_fit failed')
                return None
            fit2l = P2(times, popt2l[0], popt2l[1], popt2l[2])
            fit2u = P2(times, popt2u[0], popt2u[1], popt2u[2])
            ssres2l = np.sum((fit2l - amps)**2)
            ssres2u = np.sum((fit2u - amps)**2)
            # coefficient of determination, adjusted R^2:
            # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
            R_value2l = 1 - (ssres2l/sstot)*(len(times) - 1)/(len(times) -
                len(popt2l))
            R_value2u = 1 - (ssres2u/sstot)*(len(times) - 1)/(len(times) -
                len(popt2u))
            R_value2 = max(R_value2l, R_value2u)
            if R_value2 == R_value2l:
                popt2 = popt2l; pcov2 = pcov2l
            if R_value2 == R_value2u:
                popt2 = popt2u; pcov2 = pcov2u

            # accept the best fit and require threshold met
            maxR1 = max(R_value3, R_value2)

            if maxR1 >= fit_thresh and maxR1 == R_value3:
                pc = popt3[1]
                tau = popt3[2]
                _, pc_err, tau_err  = np.sqrt(np.diag(pcov3))
                return {3: [[pc, pc_err, tau, tau_err]]}
            if maxR1 >= fit_thresh and maxR1 == R_value2:
                pc = popt2[1]
                tau = popt2[2]
                _, pc_err, tau_err  = np.sqrt(np.diag(pcov2))
                return {2: [[pc, pc_err, tau, tau_err]]}

        # maxR1 must have been below fit_thresh.  Now try 2 traps

        try:
            popt33l, pcov33l = curve_fit(P3_P3, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt33u, pcov33u = curve_fit(P3_P3, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit33l = P3_P3(times, popt33l[0], popt33l[1], popt33l[2], popt33l[3],
        popt33l[4])
        fit33u = P3_P3(times, popt33u[0], popt33u[1], popt33u[2], popt33u[3],
        popt33u[4])
        ssres33l = np.sum((fit33l - amps)**2)
        ssres33u = np.sum((fit33u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value33l = 1 - (ssres33l/sstot)*(len(times) - 1)/(len(times) -
            len(popt33l))
        R_value33u = 1 - (ssres33u/sstot)*(len(times) - 1)/(len(times) -
            len(popt33u))
        R_value33 = max(R_value33l, R_value33u)
        if R_value33 == R_value33l:
            popt33 = popt33l; pcov33 = pcov33l
        if R_value33 == R_value33u:
            popt33 = popt33u; pcov33 = pcov33u

        try:
            popt23l, pcov23l = curve_fit(P2_P3, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt23u, pcov23u = curve_fit(P2_P3, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit23l = P2_P3(times, popt23l[0], popt23l[1], popt23l[2], popt23l[3],
        popt23l[4])
        fit23u = P2_P3(times, popt23u[0], popt23u[1], popt23u[2], popt23u[3],
        popt23u[4])
        ssres23l = np.sum((fit23l - amps)**2)
        ssres23u = np.sum((fit23u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value23l = 1 - (ssres23l/sstot)*(len(times) - 1)/(len(times) -
            len(popt23l))
        R_value23u = 1 - (ssres23u/sstot)*(len(times) - 1)/(len(times) -
            len(popt23u))
        R_value23 = max(R_value23l, R_value23u)
        if R_value23 == R_value23l:
            popt23 = popt23l; pcov23 = pcov23l
        if R_value23 == R_value23u:
            popt23 = popt23u; pcov23 = pcov23u

        try:
            popt22l, pcov22l = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt22u, pcov22u = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit22l = P2_P2(times, popt22l[0], popt22l[1], popt22l[2], popt22l[3],
        popt22l[4])
        fit22u = P2_P2(times, popt22u[0], popt22u[1], popt22u[2], popt22u[3],
        popt22u[4])
        ssres22l = np.sum((fit22l - amps)**2)
        ssres22u = np.sum((fit22u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value22l = 1 - (ssres22l/sstot)*(len(times) - 1)/(len(times) -
            len(popt22l))
        R_value22u = 1 - (ssres22u/sstot)*(len(times) - 1)/(len(times) -
            len(popt22u))
        R_value22 = max(R_value22l, R_value22u)
        if R_value22 == R_value22l:
            popt22 = popt22l; pcov22 = pcov22l
        if R_value22 == R_value22u:
            popt22 = popt22u; pcov22 = pcov22u

        maxR2 = max(R_value33, R_value23, R_value22)

        if maxR2 < fit_thresh:
            warnings.warn('No curve fit gave adjusted R^2 value above '
            'fit_thresh')
            return None

        if maxR2 == R_value33:
            off = popt33[0]
            pc = popt33[1]
            tau = popt33[2]
            pc2 = popt33[3]
            tau2 = popt33[4]
            _, pc_err, tau_err, pc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov33))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                max_a_ind = np.where(amp_a == np.max(amp_a))
                #P3 for tau
                a_tau = t_a[max_a_ind[0]]/(2*np.log(2)/3)
                #P3 for tau2
                a_tau2 = t_a[max_a_ind[0]]/(2*np.log(2)/3)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{3: [[pc, pc_err, tau, tau_err]]},
                        'b':{3: [[pc2, pc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{3: [[pc, pc_err, tau, tau_err]]},
                        'a':{3: [[pc2, pc2_err, tau2, tau2_err]]}}
            return {3: [[pc, pc_err, tau, tau_err], [pc2, pc2_err,
                tau2, tau2_err]]}

        if maxR2 == R_value23:
            off = popt23[0]
            pc = popt23[1]
            tau = popt23[2]
            pc2 = popt23[3]
            tau2 = popt23[4]
            _, pc_err, tau_err, pc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov23))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                max_a_ind = np.where(amp_a == np.max(amp_a))
                #P2 for tau
                a_tau = t_a[max_a_ind[0]]/np.log(3/2)
                #P3 for tau2
                a_tau2 = t_a[max_a_ind[0]]/(2*np.log(2)/3)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{2: [[pc, pc_err, tau, tau_err]]},
                        'b':{3: [[pc2, pc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{2: [[pc, pc_err, tau, tau_err]]},
                        'a':{3: [[pc2, pc2_err, tau2, tau2_err]]}}
            return {2: [[pc, pc_err, tau, tau_err]],
                3: [[pc2, pc2_err, tau2, tau2_err]]}

        if maxR2 == R_value22:
            off = popt22[0]
            pc = popt22[1]
            tau = popt22[2]
            pc2 = popt22[3]
            tau2 = popt22[4]
            _, pc_err, tau_err, pc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov22))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                max_a_ind = np.where(amp_a == np.max(amp_a))
                #P2 for tau
                a_tau = t_a[max_a_ind[0]]/np.log(3/2)
                #P2 for tau2
                a_tau2 = t_a[max_a_ind[0]]/np.log(3/2)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{2: [[pc, pc_err, tau, tau_err]]},
                        'b':{2: [[pc2, pc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{2: [[pc, pc_err, tau, tau_err]]},
                        'a':{2: [[pc2, pc2_err, tau2, tau2_err]]}}
            return {2: [[pc, pc_err, tau, tau_err],
                [pc2, pc2_err, tau2, tau2_err]]}

def trap_fit_k(scheme, amps, times, num_pumps, fit_thresh, tau_min, tau_max,
    tauc_min, tauc_max, offset_min, offset_max, k_min, k_max, both_a = None):
    """For a given temperature, scheme, and pixel, this function examines data
    for amplitude vs phase time and fits for release time constant (tau) and
    the probability for capture (pc).  It tries fitting for a single trap in
    the pixel, and if the goodness of fit is not high enough (if less than
    fit_thresh), then the function attempts to fit for two traps in the pixel.
    The exception is the case where a 'both' type pixel is input; then only a
    two-trap fit is attempted.  The function assumes the full time-dependent
    form for capture probability.
    Parameters
    ----------
    scheme : int, {1, 2, 3, 4}
        The scheme under consideration.  Only certain probability functions are
        valid for different schemes.
    amps : array
        Amplitudes of bright pixel of the dipole Units: e-.
    times : array
        Phase times in same order as amps.  Units:  seconds.
    num_pumps : int, > 0
        Number of cycles for trap pumping.
    fit_thresh : (0, 1)
        The minimum value required for adjusted coefficient of determination
        (adjusted R^2) for curve fitting for the release time constant
        (tau) using data for dipole amplitude vs phase time.  The closer to 1,
        the better the fit. Must be between 0 and 1.
    tau_min : float, >= 0
        Lower bound value for tau (release time constant) for curve fitting,
        in seconds.
    tau_max : float, > tau_min
        Upper bound value for tau (release time constant) for curve fitting,
        in seconds.
    tauc_min : float, >= 0
        Lower bound value for tauc (capture time constant) for curve fitting,
        in e-.
    tauc_max : float, > tau_min
        Upper bound value for tauc (capture time constant) for curve fitting,
        in e-.
    offset_min : float
        Lower bound value for the offset in the fitting of data for amplitude
        vs phase time.  Acts as a nuisance parameter.  Units of e-.
    offset_max : float, > offset_min
        Upper bound value for the offset in the fitting of data for amplitude
        vs phase time.  Acts as a nuisance parameter.  Units of e-.
    k_min : float, >= 0
        Lower bound value for k in the fitting of data for amplitude vs phase
        time.  The intent of this parameter is to account for and absorb
        any inaccuracy in k gain determination.  Unitless.
    k_max : float, > offset_min
        Upper bound value for k in the fitting of data for amplitude vs phase
        time.  The intent of this parameter is to account for and absorb
        any inaccuracy in k gain determination.  Unitless.
    both_a : None or dict, optional
        Use None if you are fitting for a dipole that is of the 'above' or
        'below' kind.  Use the dictionary corresponding to
        rc_both[(row,col)]['above'] if you are fitting for a dipole that is of
        the 'both' kind.  See doc string of trap_id() for more details on this
        dictionary.  Defaults to None.
    Returns
    -------
    Dictionary of fit data.  Uses the following:
        prob: {1, 2, 3}; denotes probability function that gave the best fit
        pc: constant capture probability, best-fit value
        pc_err: standard deviation of pc
        tau: release time constant, best-fit value
        tau_err: standard deviation of tau
        If both_a = None and one trap is the best fit,
        the return is of the following form:
        {prob: [[pc, pc_err, tau, tau_err]]}
        If both_a = None and two traps is the best fit,
        the return is of the following form:
        {prob1: [[pc, pc_err, tau, tau_err]],
        prob2: [[pc2, pc2_err, tau2, tau2_err]]}
        If both traps are of prob1 type, then the return is like this:
        {prob1: [[pc, pc_err, tau, tau_err], [pc2, pc2_err, tau2, tau2_err]]}
        If both_a is not None, then the return is of the following form:
        {type1:{1: [[pc, pc_err, tau, tau_err]]},
        type2:{2: [[pc2, pc2_err, tau2, tau2_err]]}}
        where type1 is either 'a' for above or 'b' for below, and type2 is
        whichever of those type1 is not.
    """
    # TODO time-dep pc and pc2:  assume constant charge packet throughout the
    # different phase times across all num_pumps, when in reality, the charge
    # packet changes a little with each pump (a la 'express=num_pumps' in
    # arcticpy).  The fit function would technically be a sum of num_pumps
    # terms in which tauc is a slightly different value in each.  So the value
    # of tauc may not be that useful; could compare its uncertainty to that of
    # the values in the literature.  But it's generally good for large charge
    # packet values (perhaps it nominally corresponds to the average between
    # the starting charge packet value for trap pumping and the ending value
    # read off the highest-amp phase time frame.

    # TODO Output k parameter in the output info about capture probability
    #as well?

    # check inputs
    if scheme != 1 and scheme != 2 and scheme != 3 and scheme != 4:
        raise TypeError('scheme must be 1, 2, 3, or 4.')
    try:
        amps = np.array(amps).astype(float)
    except:
        raise TypeError("amps elements should be real numbers")
    check.oneD_array(amps, 'amps', TypeError)
    try:
        times = np.array(times).astype(float)
    except:
        raise TypeError("times elements should be real numbers")
    check.oneD_array(times, 'times', TypeError)
    if len(np.unique(times)) < 6:
        raise IndexError('times must have a number of unique phase times '
        'longer than the number of fitted parameters.')
    if len(amps) != len(times):
        raise ValueError("times and amps should have same number of elements")
    check.positive_scalar_integer(num_pumps, 'num_pumps', TypeError)
    check.real_nonnegative_scalar(fit_thresh, 'fit_thresh', TypeError)
    if fit_thresh > 1:
        raise ValueError('fit_thresh should fall in (0,1).')
    check.real_nonnegative_scalar(tau_min, 'tau_min', TypeError)
    check.real_nonnegative_scalar(tau_max, 'tau_max', TypeError)
    if tau_max <= tau_min:
        raise ValueError('tau_max must be > tau_min')
    check.real_nonnegative_scalar(tauc_min, 'tauc_min', TypeError)
    check.real_nonnegative_scalar(tauc_max, 'tauc_max', TypeError)
    if tauc_max <= tauc_min:
        raise ValueError('tauc_max must be > tauc_min')
    check.real_scalar(offset_min, 'offset_min', TypeError)
    check.real_scalar(offset_max, 'offset_max', TypeError)
    if offset_max <= offset_min:
        raise ValueError('offset_max must be > offset_min')
    check.real_nonnegative_scalar(k_min, 'k_min', TypeError)
    check.real_nonnegative_scalar(k_max, 'k_max', TypeError)
    if k_max <= k_min:
        raise ValueError('k_max must be > k_min')
    if (type(both_a) is not dict) and (both_a is not None):
        raise TypeError('both_a must be a rc_both[(row,col)][\'above\'] '
        'dictionary or None.  See trap_id() doc string for more details.')
    if both_a is not None:
        try:
            x = len(both_a['amp'])
            y = len(both_a['t'])
        except:
            raise KeyError('both_a must contain \'amp\' and \'t\' keys.')
        if x != y:
            raise ValueError('both_a[\'amp\'] and both_a[\'t\'] must have the '
            'same number of elements')
    # if input for both_a doesn't conform to expected dictionary structure,
    # exceptions will be raised

    # define all these inside this function because they depend also on
    # num_pumps, and I can't specify an unfitted parameter in the function
    # definition if I want to use curve_fit
    def P1(time_data, offset, k, tauc, tau):
        """Probability function 1, one trap.
        """
        pc = 1 - np.exp(-time_data/tauc)
        return offset+(num_pumps*k*pc*(np.exp(-time_data/tau)-
            np.exp(-2*time_data/tau)))

    def P1_P1(time_data, offset, k, tauc, tau, tauc2, tau2):
        """Probability function 1, two traps.
        """
        pc = 1 - np.exp(-time_data/tauc)
        pc2 = 1 - np.exp(-time_data/tauc2)
        return offset+num_pumps*k*pc*(np.exp(-time_data/tau)-
            np.exp(-2*time_data/tau))+ \
            num_pumps*k*pc2*(np.exp(-time_data/tau2)-np.exp(-2*time_data/tau2))

    def P2(time_data, offset, k, tauc, tau):
        """Probability function 2, one trap.
        """
        pc = 1 - np.exp(-time_data/tauc)
        return offset+(num_pumps*k*pc*(np.exp(-2*time_data/tau)-
            np.exp(-3*time_data/tau)))

    def P1_P2(time_data, offset, k, tauc, tau, tauc2, tau2):
        """One trap for probability function 1, one for probability function 2.
        """
        pc = 1 - np.exp(-time_data/tauc)
        pc2 = 1 - np.exp(-time_data/tauc2)
        return offset+num_pumps*k*pc*(np.exp(-time_data/tau)-
            np.exp(-2*time_data/tau))+ \
            num_pumps*k*pc2*(np.exp(-2*time_data/tau2)-np.exp(-3*time_data/tau2))

    def P2_P2(time_data, offset, k, tauc, tau, tauc2, tau2):
        """Probability function 2, two traps.
        """
        pc = 1 - np.exp(-time_data/tauc)
        pc2 = 1 - np.exp(-time_data/tauc2)
        return offset+num_pumps*k*pc*(np.exp(-2*time_data/tau)-
            np.exp(-3*time_data/tau))+ \
            num_pumps*k*pc2*(np.exp(-2*time_data/tau2)-np.exp(-3*time_data/tau2))

    def P3(time_data, offset, k, tauc, tau):
        """Probability function 3, one trap.
        """
        pc = 1 - np.exp(-time_data/tauc)
        return offset+(num_pumps*k*pc*(np.exp(-time_data/tau)-
            np.exp(-4*time_data/tau)))

    def P3_P3(time_data, offset, k, tauc, tau, tauc2, tau2):
        """Probability function 3, two traps.
        """
        pc = 1 - np.exp(-time_data/tauc)
        pc2 = 1 - np.exp(-time_data/tauc2)
        return offset+num_pumps*k*pc*(np.exp(-time_data/tau)-
            np.exp(-4*time_data/tau))+ \
            num_pumps*k*pc2*(np.exp(-time_data/tau2)-np.exp(-4*time_data/tau2))

    def P2_P3(time_data, offset, k, tauc, tau, tauc2, tau2):
        """One trap for probability function 2, one for probability function 3.
        """
        pc = 1 - np.exp(-time_data/tauc)
        pc2 = 1 - np.exp(-time_data/tauc2)
        return offset+num_pumps*k*pc*(np.exp(-2*time_data/tau)-
            np.exp(-3*time_data/tau))+ \
            num_pumps*k*pc2*(np.exp(-time_data/tau2)-np.exp(-4*time_data/tau2))

    #upper bound for pc: 1*eperdn, but our knowledge of eperdn may have error.

    # Makes sense that you wouldn't find traps at times far away from time
    # constant, so to avoid false good fits, restrict bounds between 10^-6 and
    # 10^-2
    # in order, for one trap:  offset, k, tauc, tau
    l_bounds_one = [offset_min, k_min, tauc_min, tau_min]
    u_bounds_one = [offset_max, k_max, tauc_max, tau_max]
    # avoid initial guess of 0
    offset0 = (offset_min+offset_max)/2
    if (offset_min+offset_max)/2 == 0:
        offset0 = min(1 + (offset_min+offset_max)/2, offset_max)
    offset0l = offset_min
    if offset0l == 0:
        offset0l = min(offset0l+1, offset_max)
    offset0u = offset_max
    if offset0u == 0:
        offset0u = max(offset0u-1, offset_min)
    k0 = 1
    if 1 < k_min or 1 > k_max:
        k0 = (k_min+k_max)/2
    # tauc0 = np.median(times) #tauc0 = 1e-7
    tauc0 = 1e-9
    if tauc_min > tauc0 or tauc_max < tauc0:
        tauc0 = (tauc_min+tauc_max)/2
    tauc0l = tauc_min
    if tauc0l == 0:
        tauc0l = min(tauc0l+1e-12, tauc_max)
    tauc0u = tauc_max
    tau0 = (tau_min + tau_max)/2
    tau0l = tau_min
    if tau0l == 0:
        tau0l = min(tau0l+1e-7, tau_max)
    tau0u = tau_max
    tau0m = np.median(times)
    if tau_min > tau0m or tau_max < tau0m:
        tau0m = (tau_min+tau_max)/2
    # start search from biggest time since these data points more
    # spread out if phase times are taken evenly spaced in log space; don't
    # want to give too much weight to the bunched-up early times and lock in
    # an answer too early in the curve search, so start at other end.
    # Try search from earliest time and biggest time, and see which gives a
    # bigger adj R^2 value
    p01l = [offset0, k0, tauc0, tau0l]
    p01u = [offset0, k0, tauc0, tau0u]
    #l_bounds_one = [-100000, 0, 0]
    #u_bounds_one = [100000, 1, 1]
    bounds_one = (l_bounds_one, u_bounds_one)
    # in order, for two traps:  offset, tauc, tau, tauc2, tau2
    l_bounds_two = [offset_min, k_min, tauc_min, tau_min, tauc_min, tau_min]
    u_bounds_two = [offset_max, k_max, tauc_max, tau_max, tauc_max, tau_max]
    # p02l = [offset0l, k_min, tauc_min, tau0l, tauc_min, tau0l]
    # p02u = [offset0u, k_max, tauc_max, tau0u, tauc_max, tau0u]
    p02l = [offset0, k0, tauc0, tau0l, tauc0, tau0u]
    p02m = [offset0, k0, tauc0, tau0m, tauc0, tau0m]
    p02u = [offset0, k0, tauc0, tau0u, tauc0, tau0l]
    #l_bounds_two = [-100000, 0, 0, 0, 0]
    #u_bounds_two = [100000, 1, 1, 1, 1]
    bounds_two = (l_bounds_two, u_bounds_two)

    # same for every curve fit (sum of the squared difference total)
    sstot = np.sum((amps - np.mean(amps))**2)

    if scheme == 1 or scheme == 2:

        if both_a == None:
            # attempt all possible probability functions
            try:
                popt1l, pcov1l = curve_fit(P1, times, amps, bounds=bounds_one,
                p0 = p01l, maxfev = np.inf)#, sigma = 0.1*amps)
                popt1u, pcov1u = curve_fit(P1, times, amps, bounds=bounds_one,
                p0 = p01u, maxfev = np.inf)#, sigma = 0.1*amps)
            except:
                warnings.warn('curve_fit failed')
                return None
            fit1l = P1(times, popt1l[0], popt1l[1], popt1l[2], popt1l[3])
            fit1u = P1(times, popt1u[0], popt1u[1], popt1u[2], popt1u[3])
            ssres1l = np.sum((fit1l - amps)**2)
            ssres1u = np.sum((fit1u - amps)**2)
            # coefficient of determination, adjusted R^2:
            # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
            R_value1l = 1 - (ssres1l/sstot)*(len(times) - 1)/(len(times) -
                len(popt1l))
            R_value1u = 1 - (ssres1u/sstot)*(len(times) - 1)/(len(times) -
                len(popt1u))
            R_value1 = max(R_value1l, R_value1u)
            if R_value1 == R_value1l:
                popt1 = popt1l; pcov1 = pcov1l
            if R_value1 == R_value1u:
                popt1 = popt1u; pcov1 = pcov1u

            try:
                popt2l, pcov2l = curve_fit(P2, times, amps, bounds=bounds_one,
                p0 = p01l, maxfev = np.inf)#, sigma = 0.1*amps)
                popt2u, pcov2u = curve_fit(P2, times, amps, bounds=bounds_one,
                p0 = p01u, maxfev = np.inf)#, sigma = 0.1*amps)
            except:
                warnings.warn('curve_fit failed')
                return None
            fit2l = P2(times, popt2l[0], popt2l[1], popt2l[2], popt2l[3])
            fit2u = P2(times, popt2u[0], popt2u[1], popt2u[2], popt2u[3])
            ssres2l = np.sum((fit2l - amps)**2)
            ssres2u = np.sum((fit2u - amps)**2)
            # coefficient of determination, adjusted R^2:
            # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
            R_value2l = 1 - (ssres2l/sstot)*(len(times) - 1)/(len(times) -
                len(popt2l))
            R_value2u = 1 - (ssres2u/sstot)*(len(times) - 1)/(len(times) -
                len(popt2u))
            R_value2 = max(R_value2l, R_value2u)
            if R_value2 == R_value2l:
                popt2 = popt2l; pcov2 = pcov2l
            if R_value2 == R_value2u:
                popt2 = popt2u; pcov2 = pcov2u

            # accept the best fit and require threshold met
            maxR1 = max(R_value1, R_value2)

            if maxR1 >= fit_thresh and maxR1 == R_value1:
                tauc = popt1[2]
                tau = popt1[3]
                _, _, tauc_err, tau_err  = np.sqrt(np.diag(pcov1))
                return {1: [[tauc, tauc_err, tau, tau_err]]}
            if maxR1 >= fit_thresh and maxR1 == R_value2:
                tauc = popt2[2]
                tau = popt2[3]
                _, _, tauc_err, tau_err  = np.sqrt(np.diag(pcov2))
                return {2: [[tauc, tauc_err, tau, tau_err]]}

        # maxR1 must have been below fit_thresh.  Now try 2 traps

        try:
            popt11l, pcov11l = curve_fit(P1_P1, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt11m, pcov11m = curve_fit(P1_P1, times, amps, bounds=bounds_two,
            p0 = p02m, maxfev = np.inf)#, sigma = 0.1*amps)
            popt11u, pcov11u = curve_fit(P1_P1, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit11l = P1_P1(times, popt11l[0], popt11l[1], popt11l[2], popt11l[3],
        popt11l[4], popt11l[5])
        fit11m = P1_P1(times, popt11m[0], popt11m[1], popt11m[2], popt11m[3],
        popt11m[4], popt11m[5])
        fit11u = P1_P1(times, popt11u[0], popt11u[1], popt11u[2], popt11u[3],
        popt11u[4], popt11u[5])
        ssres11l = np.sum((fit11l - amps)**2)
        ssres11m = np.sum((fit11m - amps)**2)
        ssres11u = np.sum((fit11u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value11l = 1 - (ssres11l/sstot)*(len(times) - 1)/(len(times) -
            len(popt11l))
        R_value11m = 1 - (ssres11m/sstot)*(len(times) - 1)/(len(times) -
            len(popt11m))
        R_value11u = 1 - (ssres11u/sstot)*(len(times) - 1)/(len(times) -
            len(popt11u))
        R_value11 = max(R_value11l, R_value11m, R_value11u)
        if R_value11 == R_value11l:
            popt11 = popt11l; pcov11 = pcov11l
        if R_value11 == R_value11m:
            popt11 = popt11m; pcov11 = pcov11m
        if R_value11 == R_value11u:
            popt11 = popt11u; pcov11 = pcov11u

        try:
            popt12l, pcov12l = curve_fit(P1_P2, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt12m, pcov12m = curve_fit(P1_P2, times, amps, bounds=bounds_two,
            p0 = p02m, maxfev = np.inf)#, sigma = 0.1*amps)
            popt12u, pcov12u = curve_fit(P1_P2, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit12l = P1_P2(times, popt12l[0], popt12l[1], popt12l[2], popt12l[3],
        popt12l[4], popt12l[5])
        fit12m = P1_P2(times, popt12m[0], popt12m[1], popt12m[2], popt12m[3],
        popt12m[4], popt12m[5])
        fit12u = P1_P2(times, popt12u[0], popt12u[1], popt12u[2], popt12u[3],
        popt12u[4], popt12u[5])
        ssres12l = np.sum((fit12l - amps)**2)
        ssres12m = np.sum((fit12m - amps)**2)
        ssres12u = np.sum((fit12u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value12l = 1 - (ssres12l/sstot)*(len(times) - 1)/(len(times) -
            len(popt12l))
        R_value12m = 1 - (ssres12m/sstot)*(len(times) - 1)/(len(times) -
            len(popt12m))
        R_value12u = 1 - (ssres12u/sstot)*(len(times) - 1)/(len(times) -
            len(popt12u))
        R_value12 = max(R_value12l, R_value12m, R_value12u)
        if R_value12 == R_value12l:
            popt12 = popt12l; pcov12 = pcov12l
        if R_value12 == R_value12m:
            popt12 = popt12m; pcov12 = pcov12m
        if R_value12 == R_value12u:
            popt12 = popt12u; pcov12 = pcov12u

        try:
            popt22l, pcov22l = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt22m, pcov22m = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02m, maxfev = np.inf)#, sigma = 0.1*amps)
            popt22u, pcov22u = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit22l = P2_P2(times, popt22l[0], popt22l[1], popt22l[2], popt22l[3],
        popt22l[4], popt22l[5])
        fit22m = P2_P2(times, popt22m[0], popt22m[1], popt22m[2], popt22m[3],
        popt22m[4], popt22m[5])
        fit22u = P2_P2(times, popt22u[0], popt22u[1], popt22u[2], popt22u[3],
        popt22u[4], popt22u[5])
        ssres22l = np.sum((fit22l - amps)**2)
        ssres22m = np.sum((fit22m - amps)**2)
        ssres22u = np.sum((fit22u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value22l = 1 - (ssres22l/sstot)*(len(times) - 1)/(len(times) -
            len(popt22l))
        R_value22m = 1 - (ssres22m/sstot)*(len(times) - 1)/(len(times) -
            len(popt22m))
        R_value22u = 1 - (ssres22u/sstot)*(len(times) - 1)/(len(times) -
            len(popt22u))
        R_value22 = max(R_value22l, R_value22m, R_value22u)
        if R_value22 == R_value22l:
            popt22 = popt22l; pcov22 = pcov22l
        if R_value22 == R_value22m:
            popt22 = popt22m; pcov22 = pcov22m
        if R_value22 == R_value22u:
            popt22 = popt22u; pcov22 = pcov22u

        maxR2 = max(R_value11, R_value12, R_value22)
        if maxR2 < fit_thresh:
            warnings.warn('No curve fit gave adjusted R^2 value above '
            'fit_thresh')
            return None

        if maxR2 == R_value11:
            off = popt11[0]
            k = popt11[1]
            tauc = popt11[2]
            tau = popt11[3]
            tauc2 = popt11[4]
            tau2 = popt11[5]
            _, _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov11))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                # TODO v2, for trap_fit and trap_fit_const:
                # if 'amp' list for 'above' is identical to 'amp'
                #list for 'below', then earmark for assignment whenever
                # finding optimal matchings of schemes for sub-el loc; if
                #no particular assignment is better at that point, then the
                #assignment doesn't matter
                #TODO v2, for trap_fit and trap_fit_const:
                # if picking peak amp phase time to match with the closer tau
                # value isn't good enough (not decipherable if not enough data
                # points to make it out), then I could look at the
                # complementary probability functions in the corresponding
                #deficit pixels and curve fit those
                comp1 = P1(t_a, off, k, tauc, tau)
                comp2 = P1(t_a, off, k, tauc2, tau2)
                ssres1 = np.sum((comp1 - amp_a)**2)
                ssres2 = np.sum((comp2 - amp_a)**2)
                sstota = np.sum((amp_a - np.mean(amp_a))**2)
                # coefficient of determination, adjusted R^2:
                # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
                R1 = 1 - (ssres1/sstota)*(len(t_a) - 1)/(len(t_a) - 4)
                R2 = 1 - (ssres2/sstota)*(len(t_a) - 1)/(len(t_a) - 4)
                if R1 >= R2:
                    return {'a':{1: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{1: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{1: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{1: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {1: [[tauc, tauc_err, tau, tau_err],
                [tauc2, tauc2_err, tau2, tau2_err]]}

        if maxR2 == R_value12:
            off = popt12[0]
            k = popt12[1]
            tauc = popt12[2]
            tau = popt12[3]
            tauc2 = popt12[4]
            tau2 = popt12[5]
            _, _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov12))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                comp1 = P1(t_a, off, k, tauc, tau)
                comp2 = P2(t_a, off, k, tauc2, tau2)
                ssres1 = np.sum((comp1 - amp_a)**2)
                ssres2 = np.sum((comp2 - amp_a)**2)
                sstota = np.sum((amp_a - np.mean(amp_a))**2)
                # coefficient of determination, adjusted R^2:
                # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
                R1 = 1 - (ssres1/sstota)*(len(t_a) - 1)/(len(t_a) - 4)
                R2 = 1 - (ssres2/sstota)*(len(t_a) - 1)/(len(t_a) - 4)
                if R1 >= R2:
                    return {'a':{1: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{1: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {1: [[tauc, tauc_err, tau, tau_err]],
                2: [[tauc2, tauc2_err, tau2, tau2_err]]}

        if maxR2 == R_value22:
            off = popt22[0]
            k = popt22[1]
            tauc = popt22[2]
            tau = popt22[3]
            tauc2 = popt22[4]
            tau2 = popt22[5]
            _, _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov22))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                comp1 = P2(t_a, off, k, tauc, tau)
                comp2 = P2(t_a, off, k, tauc2, tau2)
                ssres1 = np.sum((comp1 - amp_a)**2)
                ssres2 = np.sum((comp2 - amp_a)**2)
                sstota = np.sum((amp_a - np.mean(amp_a))**2)
                # coefficient of determination, adjusted R^2:
                # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
                R1 = 1 - (ssres1/sstota)*(len(t_a) - 1)/(len(t_a) - 4)
                R2 = 1 - (ssres2/sstota)*(len(t_a) - 1)/(len(t_a) - 4)
                if R1 >= R2:
                    return {'a':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {2: [[tauc, tauc_err, tau, tau_err],
                [tauc2, tauc2_err, tau2, tau2_err]]}

    if scheme == 3 or scheme == 4:

        if both_a == None:
            #attempt both probability functions
            try:
                popt3l, pcov3l = curve_fit(P3, times, amps, bounds=bounds_one,
                p0 = p01l, maxfev = np.inf)#, sigma = 0.1*amps)
                popt3u, pcov3u = curve_fit(P3, times, amps, bounds=bounds_one,
                p0 = p01u, maxfev = np.inf)#, sigma = 0.1*amps)
            except:
                warnings.warn('curve_fit failed')
                return None
            fit3l = P3(times, popt3l[0], popt3l[1], popt3l[2], popt3l[3])
            fit3u = P3(times, popt3u[0], popt3u[1], popt3u[2], popt3u[3])
            ssres3l = np.sum((fit3l - amps)**2)
            ssres3u = np.sum((fit3u - amps)**2)
            # coefficient of determination, adjusted R^2:
            # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
            R_value3l = 1 - (ssres3l/sstot)*(len(times) - 1)/(len(times) -
                len(popt3l))
            R_value3u = 1 - (ssres3u/sstot)*(len(times) - 1)/(len(times) -
                len(popt3u))
            R_value3 = max(R_value3l, R_value3u)
            if R_value3 == R_value3l:
                popt3 = popt3l; pcov3 = pcov3l
            if R_value3 == R_value3u:
                popt3 = popt3u; pcov3 = pcov3u

            try:
                popt2l, pcov2l = curve_fit(P2, times, amps, bounds=bounds_one,
                p0 = p01l, maxfev = np.inf)#, sigma = 0.1*amps)
                popt2u, pcov2u = curve_fit(P2, times, amps, bounds=bounds_one,
                p0 = p01u, maxfev = np.inf)#, sigma = 0.1*amps)
            except:
                warnings.warn('curve_fit failed')
                return None
            fit2l = P2(times, popt2l[0], popt2l[1], popt2l[2], popt2l[3])
            fit2u = P2(times, popt2u[0], popt2u[1], popt2u[2], popt2u[3])
            ssres2l = np.sum((fit2l - amps)**2)
            ssres2u = np.sum((fit2u - amps)**2)
            # coefficient of determination, adjusted R^2:
            # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
            R_value2l = 1 - (ssres2l/sstot)*(len(times) - 1)/(len(times) -
                len(popt2l))
            R_value2u = 1 - (ssres2u/sstot)*(len(times) - 1)/(len(times) -
                len(popt2u))
            R_value2 = max(R_value2l, R_value2u)
            if R_value2 == R_value2l:
                popt2 = popt2l; pcov2 = pcov2l
            if R_value2 == R_value2u:
                popt2 = popt2u; pcov2 = pcov2u

            # accept the best fit and require threshold met
            maxR1 = max(R_value3, R_value2)

            if maxR1 >= fit_thresh and maxR1 == R_value3:
                tauc = popt3[2]
                tau = popt3[3]
                _, _, tauc_err, tau_err  = np.sqrt(np.diag(pcov3))
                return {3: [[tauc, tauc_err, tau, tau_err]]}
            if maxR1 >= fit_thresh and maxR1 == R_value2:
                tauc = popt2[2]
                tau = popt2[3]
                _, _, tauc_err, tau_err  = np.sqrt(np.diag(pcov2))
                return {2: [[tauc, tauc_err, tau, tau_err]]}

        # maxR1 must have been below fit_thresh.  Now try 2 traps

        try:
            popt33l, pcov33l = curve_fit(P3_P3, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt33m, pcov33m = curve_fit(P3_P3, times, amps, bounds=bounds_two,
            p0 = p02m, maxfev = np.inf)#, sigma = 0.1*amps)
            popt33u, pcov33u = curve_fit(P3_P3, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit33l = P3_P3(times, popt33l[0], popt33l[1], popt33l[2], popt33l[3],
        popt33l[4], popt33l[5])
        fit33m = P3_P3(times, popt33m[0], popt33m[1], popt33m[2], popt33m[3],
        popt33m[4], popt33m[5])
        fit33u = P3_P3(times, popt33u[0], popt33u[1], popt33u[2], popt33u[3],
        popt33u[4], popt33u[5])
        ssres33l = np.sum((fit33l - amps)**2)
        ssres33m = np.sum((fit33m - amps)**2)
        ssres33u = np.sum((fit33u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value33l = 1 - (ssres33l/sstot)*(len(times) - 1)/(len(times) -
            len(popt33l))
        R_value33m = 1 - (ssres33m/sstot)*(len(times) - 1)/(len(times) -
            len(popt33m))
        R_value33u = 1 - (ssres33u/sstot)*(len(times) - 1)/(len(times) -
            len(popt33u))
        R_value33 = max(R_value33l, R_value33m, R_value33u)
        if R_value33 == R_value33l:
            popt33 = popt33l; pcov33 = pcov33l
        if R_value33 == R_value33m:
            popt33 = popt33m; pcov33 = pcov33m
        if R_value33 == R_value33u:
            popt33 = popt33u; pcov33 = pcov33u

        try:
            popt23l, pcov23l = curve_fit(P2_P3, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt23m, pcov23m = curve_fit(P2_P3, times, amps, bounds=bounds_two,
            p0 = p02m, maxfev = np.inf)#, sigma = 0.1*amps)
            popt23u, pcov23u = curve_fit(P2_P3, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit23l = P2_P3(times, popt23l[0], popt23l[1], popt23l[2], popt23l[3],
        popt23l[4], popt23l[5])
        fit23m = P2_P3(times, popt23m[0], popt23m[1], popt23m[2], popt23m[3],
        popt23m[4], popt23m[5])
        fit23u = P2_P3(times, popt23u[0], popt23u[1], popt23u[2], popt23u[3],
        popt23u[4], popt23u[5])
        ssres23l = np.sum((fit23l - amps)**2)
        ssres23m = np.sum((fit23m - amps)**2)
        ssres23u = np.sum((fit23u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value23l = 1 - (ssres23l/sstot)*(len(times) - 1)/(len(times) -
            len(popt23l))
        R_value23m = 1 - (ssres23m/sstot)*(len(times) - 1)/(len(times) -
            len(popt23m))
        R_value23u = 1 - (ssres23u/sstot)*(len(times) - 1)/(len(times) -
            len(popt23u))
        R_value23 = max(R_value23l, R_value23m, R_value23u)
        if R_value23 == R_value23l:
            popt23 = popt23l; pcov23 = pcov23l
        if R_value23 == R_value23m:
            popt23 = popt23m; pcov23 = pcov23m
        if R_value23 == R_value23u:
            popt23 = popt23u; pcov23 = pcov23u

        try:
            popt22l, pcov22l = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt22m, pcov22m = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02m, maxfev = np.inf)#, sigma = 0.1*amps)
            popt22u, pcov22u = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit22l = P2_P2(times, popt22l[0], popt22l[1], popt22l[2], popt22l[3],
        popt22l[4], popt22l[5])
        fit22m = P2_P2(times, popt22m[0], popt22m[1], popt22m[2], popt22m[3],
        popt22m[4], popt22m[5])
        fit22u = P2_P2(times, popt22u[0], popt22u[1], popt22u[2], popt22u[3],
        popt22u[4], popt22u[5])
        ssres22l = np.sum((fit22l - amps)**2)
        ssres22m = np.sum((fit22m - amps)**2)
        ssres22u = np.sum((fit22u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value22l = 1 - (ssres22l/sstot)*(len(times) - 1)/(len(times) -
            len(popt22l))
        R_value22m = 1 - (ssres22m/sstot)*(len(times) - 1)/(len(times) -
            len(popt22m))
        R_value22u = 1 - (ssres22u/sstot)*(len(times) - 1)/(len(times) -
            len(popt22u))
        R_value22 = max(R_value22l, R_value22m, R_value22u)
        if R_value22 == R_value22l:
            popt22 = popt22l; pcov22 = pcov22l
        if R_value22 == R_value22m:
            popt22 = popt22m; pcov22 = pcov22m
        if R_value22 == R_value22u:
            popt22 = popt22u; pcov22 = pcov22u

        maxR2 = max(R_value33, R_value23, R_value22)

        if maxR2 < fit_thresh:
            warnings.warn('No curve fit gave adjusted R^2 value above '
            'fit_thresh')
            return None

        if maxR2 == R_value33:
            off = popt33[0]
            k = popt33[1]
            tauc = popt33[2]
            tau = popt33[3]
            tauc2 = popt33[4]
            tau2 = popt33[5]
            _, _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov33))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                comp1 = P3(t_a, off, k, tauc, tau)
                comp2 = P3(t_a, off, k, tauc2, tau2)
                ssres1 = np.sum((comp1 - amp_a)**2)
                ssres2 = np.sum((comp2 - amp_a)**2)
                sstota = np.sum((amp_a - np.mean(amp_a))**2)
                # coefficient of determination, adjusted R^2:
                # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
                R1 = 1 - (ssres1/sstota)*(len(t_a) - 1)/(len(t_a) - 4)
                R2 = 1 - (ssres2/sstota)*(len(t_a) - 1)/(len(t_a) - 4)
                if R1 >= R2:
                    return {'a':{3: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{3: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{3: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{3: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {3: [[tauc, tauc_err, tau, tau_err], [tauc2, tauc2_err,
                tau2, tau2_err]]}

        if maxR2 == R_value23:
            off = popt23[0]
            k = popt23[1]
            tauc = popt23[2]
            tau = popt23[3]
            tauc2 = popt23[4]
            tau2 = popt23[5]
            _, _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov23))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                comp1 = P2(t_a, off, k, tauc, tau)
                comp2 = P3(t_a, off, k, tauc2, tau2)
                ssres1 = np.sum((comp1 - amp_a)**2)
                ssres2 = np.sum((comp2 - amp_a)**2)
                sstota = np.sum((amp_a - np.mean(amp_a))**2)
                # coefficient of determination, adjusted R^2:
                # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
                R1 = 1 - (ssres1/sstota)*(len(t_a) - 1)/(len(t_a) - 4)
                R2 = 1 - (ssres2/sstota)*(len(t_a) - 1)/(len(t_a) - 4)
                if R1 >= R2:
                    return {'a':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{3: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{3: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {2: [[tauc, tauc_err, tau, tau_err]],
                3: [[tauc2, tauc2_err, tau2, tau2_err]]}

        if maxR2 == R_value22:
            off = popt22[0]
            k = popt22[1]
            tauc = popt22[2]
            tau = popt22[3]
            tauc2 = popt22[4]
            tau2 = popt22[5]
            _, _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov22))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                comp1 = P2(t_a, off, k, tauc, tau)
                comp2 = P2(t_a, off, k, tauc2, tau2)
                ssres1 = np.sum((comp1 - amp_a)**2)
                ssres2 = np.sum((comp2 - amp_a)**2)
                sstota = np.sum((amp_a - np.mean(amp_a))**2)
                # coefficient of determination, adjusted R^2:
                # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
                R1 = 1 - (ssres1/sstota)*(len(t_a) - 1)/(len(t_a) - 4)
                R2 = 1 - (ssres2/sstota)*(len(t_a) - 1)/(len(t_a) - 4)
                if R1 >= R2:
                    return {'a':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {2: [[tauc, tauc_err, tau, tau_err],
                [tauc2, tauc2_err, tau2, tau2_err]]}


def tau_temp(temp_data, E, cs):
    '''Calculates the release time constant (tau, s), based on the input
    temperature (temp_data, K), the energy level (E, eV), and the capture cross
    section for holes (cs, 1e-19 m^2, or 1e-15 cm^2).  See Appendix in
    2020 Bush et al.pdf from docs folder for details.'''
    k = 8.6173e-5 # eV/K
    kb = 1.381e-23 # mks units
    hconst = 6.626e-34 # mks units
    Eg = 1.1692 - (4.9e-4)*temp_data**2/(temp_data+655) #eV
    me = 9.109e-31 # kg, rest mass of electron
    mlstar = 0.1963 * me #kg
    mtstar = 0.1905 * 1.1692 * me / Eg #kg
    mstardc = 6**(2/3) * (mtstar**2*mlstar)**(1/3) #kg
    vth = np.sqrt(3*kb*temp_data/mstardc) # m/s
    Nc = 2*(2*np.pi*mstardc*kb*temp_data/(hconst**2))**1.5 # 1/m^3
    # added a factor of 1e-19 so that curve_fit step size reasonable
    return np.e**(E/(k*temp_data))/(cs*Nc*vth*1e-19)

def sig_tau_temp(temp_data, E, cs, sig_E, sig_cs):
    '''Calculates the standard deviation of the release time constant
    (sig_tau, s) via error propagation
    based on the input temperature (temp_data, K), the energy
    level (E, eV), the capture cross section for holes (cs, 1e-19 m^2, or
    1e-15 cm^2),
    the standard deviations for the energy level (sig_E), and the standard
    deviation for the capture cross section for holes (sig_cs).
    See Appendix in 2020 Bush et al.pdf from docs folder for details.'''
    k = 8.6173e-5 # eV/K
    kb = 1.381e-23 # mks units
    hconst = 6.626e-34 # mks units
    Eg = 1.1692 - (4.9e-4)*temp_data**2/(temp_data+655)
    me = 9.109e-31 # kg, rest mass of electron
    mlstar = 0.1963 * me #kg
    mtstar = 0.1905 * 1.1692 * me / Eg #kg
    mstardc = 6**(2/3) * (mtstar**2*mlstar)**(1/3) #kg
    vth = np.sqrt(3*kb*temp_data/mstardc) # m/s
    Nc = 2*(2*np.pi*mstardc*kb*temp_data/(hconst**2))**1.5 # 1/m^3
    # added a factor of 1e-19 so that curve_fit step size reasonable
    tau = np.e**(E/(k*temp_data))/(cs*Nc*vth*1e-19)
    dtau_dcs = - tau/(cs*1e-19)
    dtau_dE = tau/(k*temp_data)
    sig_tau = np.sqrt((dtau_dcs*sig_cs*1e-19)**2 + (dtau_dE*sig_E)**2)
    return sig_tau


def fit_cs(taus, tau_errs, temps, cs_fit_thresh, E_min, E_max, cs_min, cs_max,
    input_T):
    """This function fits the cross section for holes (cs) for a given trap
    by curve-fitting release time constant (tau) vs temperature.  Returns fit
    parameters and the release time constant at the desired input temperature.
    Parameters
    ----------
    taus : array
        Array of tau values (in seconds).
    tau_errs : array
        Array of tau uncertainty values (in seconds), with elements in
        the same order as that of taus.
    temps : array
        Array of temperatures (in Kelvin), with elements in the same
        order as that of taus.
    cs_fit_thresh : (0, 1)
        The minimum value required for adjusted coefficient of determination
        (adjusted R^2) for curve fitting for the capture cross section
        for holes (cs) using data for tau vs temperature.  The closer to 1,
        the better the fit. Must be between 0 and 1.
    E_min : float, >= 0
        Lower bound for E (energy level in release time constant) for curve
        fitting, in eV.
    E_max : float, > E_min
        Upper bound for E (energy level in release time constant) for curve
        fitting, in eV.
    cs_min : float, >= 0
        Lower bound for cs (capture cross section for holes in release time
        constant) for curve fitting, in 1e-19 m^2.
    cs_max : float, > cs_min
        Upper bound for cs (capture cross section for holes in release time
        constant) for curve fitting, in 1e-19 m^2.
    input_T : float, > 0
        Temperature of Roman EMCCD at which to calculate the
        release time constant (in units of Kelvin).
    Returns
    -------
    E : float
        Energy level (in eV).
    sig_E: float
        Standard deviation error of energy level.  In eV.
    cs : float
        Cross section for holes.  In cm^2.
    sig_cs: float
        Standard deviation error of cross section for holes. In cm^2.
    Rsq : float
        Adjusted R^2 for the tau vs temperature fit that was done to obtain cs.
    tau_input_T : float
        Tau evaluated at desired temperature of Roman EMCCD,
        input_T.  In seconds.
    sig_tau_input_T: float
        Standard deviation error of tau at desired
        temperature of Roman EMCCD, input_T.  Found by propagating error by
        utilizing sig_cs and sig_E.  In seconds.

    """
    # input checks
    try:
        taus = np.array(taus).astype(float)
    except:
        raise TypeError("taus elements should be real numbers")
    check.oneD_array(taus, 'taus', TypeError)
    try:
        temps = np.array(temps).astype(float)
    except:
        raise TypeError("temps elements should be real numbers")
    check.oneD_array(temps, 'temps', TypeError)
    if len(temps) != len(taus):
        raise ValueError('temps and taus must have the same length')
    try:
        tau_errs = np.array(tau_errs).astype(float)
    except:
        raise TypeError("tau_errs elements should be real numbers")
    check.oneD_array(tau_errs, 'tau_errs', TypeError)
    if len(tau_errs) != len(taus):
        raise ValueError('tau_errs and taus must have the same length')
    check.real_nonnegative_scalar(cs_fit_thresh, 'cs_fit_thresh', TypeError)
    if cs_fit_thresh > 1:
        raise ValueError('cs_fit_thresh should fall in (0,1).')
    check.real_nonnegative_scalar(E_min, 'E_min', TypeError)
    check.real_nonnegative_scalar(E_max, 'E_max', TypeError)
    if E_max <= E_min:
        raise ValueError('E_max must be > E_min')
    check.real_nonnegative_scalar(cs_min, 'cs_min', TypeError)
    check.real_nonnegative_scalar(cs_max, 'cs_max', TypeError)
    if cs_max <= cs_min:
        raise ValueError('cs_max must be > cs_min')
    check.real_positive_scalar(input_T, 'input_T', TypeError)
    if len(np.unique(temps)) < 3:
        warnings.warn('temps did not have a unique number of temperatures '
        'longer than the number of fitted parameters.')
        return None, None, None, None, None, None, None

    l_bounds = [E_min, cs_min]
    # typically, E is no more than 0.5 eV, and eV = 1.6e-19 J
    # cs usually 1e-15 to 1e-14 cm^2, or 1e-19 to 1e-18 m^2
    u_bounds = [E_max, cs_max]  # E in eV, cs in 1e-19 m^2
    bounds = (l_bounds, u_bounds)
    p0 = [(E_min + E_max)/2, (cs_min + cs_max)/2]
    # in case you have a perfect tau_err of 0 (highly unlikely), set it equal
    # to machine precision eps to prevent the curve_fit from crashing
    tau_errs[tau_errs==0] = np.finfo(float).eps

    try:
        # We consider absolute_sigma=True since we likely won't have many data
        # points, so sample variance of residuals may be low.  When it's False,
        # the tau_errs are scaled to match sample variance of residuals after
        # the fit, which would  most likely be a scaling down.  So when it's
        # True, the resulting std dev from pcov would likely be bigger and
        # maybe more accurate.  If we do have enough data points, the opposite
        # would be true.  So best thing to do is to take the case where the
        # error is bigger:
        poptt, pcovt = curve_fit(tau_temp, temps, taus, bounds = bounds, p0=p0,
                sigma = tau_errs, absolute_sigma=True, maxfev = np.inf)
        poptf, pcovf = curve_fit(tau_temp, temps, taus, bounds = bounds, p0=p0,
                sigma = tau_errs, absolute_sigma=False, maxfev = np.inf)
        if np.sum(np.sqrt(np.diag(pcovt))) >= np.sum(np.sqrt(np.diag(pcovf))):
            popt = poptt
            pcov = pcovt
        else:
            popt = poptf
            pcov = pcovf
    except:
            warnings.warn('curve_fit failed')
            return None, None, None, None, None, None, None
    E = popt[0] # in eV
    sig_E = np.sqrt(np.diag(pcov))[0]
    cs = popt[1]*1e-15 # in cm^2
    sig_cs = np.sqrt(np.diag(pcov))[1]*1e-15 # in cm^2
    tau_input_T = tau_temp(input_T, E, cs*1e15) # in s
    sig_tau_input_T = sig_tau_temp(input_T, E, cs*1e15, sig_E, sig_cs*1e15) #s

    fit = tau_temp(temps, popt[0], popt[1])
    ssres = np.sum((fit - taus)**2)
    sstot = np.sum((taus - np.mean(taus))**2)
    # coefficient of determination, adjusted R^2:
    # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
    Rsq = 1 - (ssres/sstot)*(len(temps) - 1)/(len(temps) - len(popt))

    if Rsq < cs_fit_thresh:
        warnings.warn('Fitting of tau vs temperature has an adjusted R^2 '
        'value < cs_fit_thresh')

    return (E, sig_E, cs, sig_cs, Rsq, tau_input_T, sig_tau_input_T)

if __name__ == '__main__':
    taus = np.array([1e-6, 1.44e-6, 1.5e-6])
    temps = np.array([160, 162, 164])
    tau_errs = np.array([.1,.1, -.2])
    E, sig_E, cs, sig_cs, Rsq, tau_input_T, sig_tau_input_T = \
        fit_cs(taus, tau_errs, temps,0.8, 0.01,
      1, 0.01, 1, 100)
    # h = trap_fit(1, np.array([1,2]), np.array([1,2]), 10000, .8, 0, 1,
    #  0, 1, -10000,10000, both_a = None)
    # print(h)

    num_pumps = 10000
    def P1(time_data, offset, tauc, tau):
        """Probability function 1, one trap.
        """
        pc = 1 - np.exp(-time_data/tauc)
        return offset+(num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-2*time_data/tau)))

    def P1_P1(time_data, offset, tauc, tau, tauc2, tau2):
        """Probability function 1, two traps.
        """
        pc = 1 - np.exp(-time_data/tauc)
        pc2 = 1 - np.exp(-time_data/tauc2)
        return offset+num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-2*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-2*time_data/tau2))

    def P2(time_data, offset, tauc, tau):
        """Probability function 2, one trap.
        """
        pc = 1 - np.exp(-time_data/tauc)
        return offset+(num_pumps*pc*(np.exp(-2*time_data/tau)-
            np.exp(-3*time_data/tau)))

    def P1_P2(time_data, offset, tauc, tau, tauc2, tau2):
        """One trap for probability function 1, one for probability function 2.
        """
        pc = 1 - np.exp(-time_data/tauc)
        pc2 = 1 - np.exp(-time_data/tauc2)
        return offset+num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-2*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-2*time_data/tau2)-np.exp(-3*time_data/tau2))

    def P2_P2(time_data, offset, tauc, tau, tauc2, tau2):
        """Probability function 2, two traps.
        """
        pc = 1 - np.exp(-time_data/tauc)
        pc2 = 1 - np.exp(-time_data/tauc2)
        return offset+num_pumps*pc*(np.exp(-2*time_data/tau)-
            np.exp(-3*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-2*time_data/tau2)-np.exp(-3*time_data/tau2))

    def P3(time_data, offset, tauc, tau):
        """Probability function 3, one trap.
        """
        pc = 1 - np.exp(-time_data/tauc)
        return offset+(num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-4*time_data/tau)))

    def P3_P3(time_data, offset, tauc, tau, tauc2, tau2):
        """Probability function 3, two traps.
        """
        pc = 1 - np.exp(-time_data/tauc)
        pc2 = 1 - np.exp(-time_data/tauc2)
        return offset+num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-4*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-4*time_data/tau2))

    def P2_P3(time_data, offset, tauc, tau, tauc2, tau2):
        """One trap for probability function 2, one for probability function 3.
        """
        pc = 1 - np.exp(-time_data/tauc)
        pc2 = 1 - np.exp(-time_data/tauc2)
        return offset+num_pumps*pc*(np.exp(-2*time_data/tau)-
            np.exp(-3*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-4*time_data/tau2))

    def tau_temp(temp_data, E, cs):
        k = 8.6173e-5 # eV/K
        kb = 1.381e-23 # mks units
        hconst = 6.626e-34 # mks units
        Eg = 1.1692 - (4.9e-4)*temp_data**2/(temp_data+655)
        me = 9.109e-31 # kg
        mlstar = 0.1963 * me
        mtstar = 0.1905 * 1.1692 * me / Eg
        mstardc = 6**(2/3) * (mtstar**2*mlstar)**(1/3)
        vth = np.sqrt(3*kb*temp_data/mstardc) # m/s
        Nc = 2*(2*np.pi*mstardc*kb*temp_data/(hconst**2))**1.5 # 1/m^3
        # added a factor of 1e-19 so that curve_fit step size reasonable
        return np.e**(E/(k*temp_data))/(cs*Nc*vth*1e-19)
    tauc = 1e-8
    tauc2 = 1.2e-8
    tauc3 = 1e-8
    #In order of amplitudes overall (given comparable tau and tau2):
    # P1 biggest, then P3, then P2
    # E,E3 and cs,cs3 params below chosen to ensure a P1 trap found at its
    # peak amp for good eperdn determination
    # E3,cs3: will give tau outside of 1e-6,1e-2
    # for all temps except 220K; we'll just make sure it's present in all
    # scheme 1 stacks for all temps to ensure good eperdn for all temps;
    # E, cs: will give tau outside of 1e-6, 1e-2
    # for just 170K, which I took out of temp_data
    # E2, cs2: fine for all temps
    E = 0.32 #eV
    E2 = 0.28 # eV
    E3 = 0.4 #eV
    cs = 2 #in 1e-19 m^2
    cs2 = 8 #12 # in 1e-19 m^2
    cs3 = 2 # in 1e-19 m^2
    #temp_data = np.array([170, 180, 190, 200, 210, 220])
    temp_data = np.array([180, 190, 200, 210, 220])
    taus = {}
    taus2 = {}
    taus3 = {}
    for i in temp_data:
        taus[i] = tau_temp(i, E, cs)
        taus2[i] = tau_temp(i, E2, cs2)
        taus3[i] = tau_temp(i, E3, cs3)
    time_data = (np.logspace(-6, -2, 100))*10**6 # in us
    #time_data = (np.linspace(1e-6, 1e-2, 100))*10**6 # in us
    time_data = time_data.astype(float)
    # make one phase time a repitition
    time_data[-1] = time_data[-2]
    time_data_s = time_data/10**6 # in s
    amps1 = {}; amps2 = {}; amps3 = {}; amps1_k = {}
    amps11 = {}; amps12 = {}; amps22 = {}; amps23 = {}; amps33 = {}
    for i in temp_data:
        amps1[i] = P1(time_data_s, 0, tauc, taus[i])
        amps11[i] = P1_P1(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i])
        amps2[i] = P2(time_data_s, 0, tauc, taus[i])
        amps12[i] = P1_P2(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i])
        amps22[i] = P2_P2(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i])
        amps3[i] = P3(time_data_s, 0, tauc, taus[i])
        amps33[i] = P3_P3(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i])
        amps23[i] = P2_P3(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i])

    # (41,15), sch3: should be P23
    times = np.array([1.00000000e-06, 1.20679264e-06, 1.45634848e-06,
        1.75751062e-06,
       1.04811313e-03, 1.09854114e-04, 1.15139540e-05, 1.26485522e-03,
       1.38949549e-05, 1.32571137e-04, 1.52641797e-03, 1.59985872e-04,
       1.67683294e-05, 1.84206997e-03, 1.93069773e-04, 2.12095089e-06,
       2.55954792e-06, 2.02358965e-05, 2.22299648e-03, 2.32995181e-04,
       2.44205309e-05, 2.68269580e-03, 2.81176870e-04, 2.94705170e-05,
       3.08884360e-06, 3.72759372e-06, 3.23745754e-03, 3.39322177e-04,
       3.55648031e-05, 3.90693994e-03, 4.49843267e-06, 4.09491506e-04,
       4.29193426e-05, 4.71486636e-03, 4.94171336e-04, 5.42867544e-06,
       5.17947468e-05, 5.68986603e-03, 5.96362332e-04, 6.55128557e-06,
       6.25055193e-05, 6.86648845e-03, 7.90604321e-06, 7.19685673e-04,
       7.54312006e-05, 8.28642773e-03, 8.28642773e-03, 8.68511374e-04,
       9.54095476e-06, 9.10298178e-05])

    amps = np.array([ 232.3263813 ,  279.24483302,  335.36048524, 402.35081306,
       1059.93913456, 4062.45181149, 2186.17905714, 1178.07914572,
       2522.28535159, 3565.95940436, 1290.26251711, 3008.49088901,
       2884.18933611, 1385.84726225, 2442.57043815,  482.14228811,
        576.92052212, 3263.20579889, 1453.50926938, 1918.53096961,
       3645.9285654 , 1481.41297232, 1476.72266823, 4013.64107116,
        689.12604491,  821.42693177, 1458.76387861, 1142.75823995,
       4342.35496872, 1378.3316254 ,  976.65618063,  925.75702091,
       4603.89793424, 1239.41852348,  819.13877237, 1157.69779369,
       4768.44792094, 1050.3644731 ,  804.04253841, 1367.30125725,
       4808.6881146 ,  829.26893898, 1607.80030552,  855.1642488 ,
       4705.26045396,  601.65154929,  601.65154929,  947.30763119,
       1880.71001702, 4452.46533616])
    both_a = {'amp': amps23[190][35:50], 't': time_data_s[35:50]}
    # fd4 = trap_fit(3, amps,
    #     times, num_pumps, fit_thresh=0.8,
    #     tau_min=0.7e-6, tau_max=1.3e-2, tauc_min=0, tauc_max=1e-5,offset_min=-10,
    #                         offset_max=10,
    #                         both_a = both_a)
    # print(fd4)
    # fd5 = trap_fit_const(3, amps,
    #     times, num_pumps, fit_thresh=0.8,
    #     tau_min=0.7e-6, tau_max=1.3e-2, pc_min=0, pc_max=2, offset_min=-10,
    #                         offset_max=10)
    # print(fd5)
    # fd3 = trap_fit(3, amps23[180],
    #     time_data_s, num_pumps, fit_thresh=0.8,
    #     tau_min=0.7e-6, tau_max=1.3e-2, tauc_min=0, tauc_max=1e-5, offset_min=-10,
    #                         offset_max=10)
    # print(fd3)
    # fd1 = trap_fit(2, amps12[180],
    #     time_data_s, num_pumps, fit_thresh=0.8,
    #     tau_min=0.7e-6, tau_max=1.3e-2, tauc_min=0, tauc_max=1e-5, offset_min=-10,
    #                         offset_max=10)
    # print(fd1)
    # fd2 = trap_fit_const(3, amps23[180],
    #     time_data_s, num_pumps, fit_thresh=0.8,
    #     tau_min=0.7e-6, tau_max=1.3e-2, pc_min = 0, pc_max = 2, offset_min=-10,
    #                         offset_max=10)
    # print(fd2)

    # fd6 = trap_fit_const(3, amps,
    #     times, num_pumps, fit_thresh=0.8,
    #     tau_min=0.7e-6, tau_max=1.3e-2, pc_min = 0, pc_max = 2, offset_min=-10,
    #                         offset_max=10)
    # print(fd6)


    # fd7 = trap_fit(1, amps12[180],
    #     time_data_s, num_pumps, fit_thresh=0.8,
    #     tau_min=0.7e-6, tau_max=1.3e-2, tauc_min=0, tauc_max=1e-5, offset_min=-10,
    #                         offset_max=10)
    # print('fd: ',fd7)

    # fd = trap_fit_k(1, amps12[180],
    #     time_data_s, num_pumps, fit_thresh=0.8,
    #     tau_min=0.7e-6, tau_max=1.3e-2, tauc_min=0, tauc_max=1e-5, offset_min=-10,
    #                         offset_max=10, k_min = 0, k_max = 2)
    # print('fd: ',fd)
    # # trap_fit_const: (41,15) for 200, 210, 220
    # fd8 = trap_fit_const(3, amps23[200],
    #     time_data_s, num_pumps, fit_thresh=0.9,
    #     tau_min=0.7e-6, tau_max=1.3e-2, pc_min = 0, pc_max = 2, offset_min=-10,
    #                         offset_max=10)
    # print("trap_fit_const, 23[200]: ", fd8)
    # fd9 = trap_fit_const(3, amps23[210],
    #     time_data_s, num_pumps, fit_thresh=0.9,
    #     tau_min=0.7e-6, tau_max=1.3e-2, pc_min = 0, pc_max = 2, offset_min=-10,
    #                         offset_max=10)
    # print("trap_fit_const, 23[210]: ", fd9)
    # fd10 = trap_fit_const(3, amps23[220],
    #     time_data_s, num_pumps, fit_thresh=0.9,
    #     tau_min=0.7e-6, tau_max=1.3e-2, pc_min = 0, pc_max = 2, offset_min=-10,
    #                         offset_max=10)
    # print("trap_fit_const, 23[220]: ", fd10)

    # trap_fit_const:  (77,90) for 210, 220 needs sch1 12, sch2 11 for 210, 220
    # could also the 'a' separation isn't working for those temps
    both_a_7790 = {'amp': amps12[180][30:100], 't': time_data_s[30:100]}
    fd11 = trap_fit_const(1, amps12[180],
        time_data_s, num_pumps, fit_thresh=0.9,
        tau_min=0.7e-6, tau_max=1.3e-2, pc_min = 0, pc_max = 2, offset_min=-10,
                            offset_max=10, both_a = both_a_7790)
    print('trap_fit_const, 12[180]: ', fd11)
    both_a_7790_220 = {'amp': amps12[220][30:100], 't': time_data_s[30:100]}
    fd12 = trap_fit_const(1, amps12[220],
        time_data_s, num_pumps, fit_thresh=0.9,
        tau_min=0.7e-6, tau_max=1.3e-2, pc_min = 0, pc_max = 2, offset_min=-10,
                            offset_max=10, both_a = both_a_7790_220)
    print('trap_fit_const, 12[220]: ', fd12)
    both_a_7790_180_2 = {'amp': amps11[180][0:53], 't': time_data_s[0:53]}
    fd13 = trap_fit_const(2, amps11[180],
        time_data_s, num_pumps, fit_thresh=0.9,
        tau_min=0.7e-6, tau_max=1.3e-2, pc_min = 0, pc_max = 2, offset_min=-10,
                            offset_max=10, both_a = both_a_7790_180_2)
    print('trap_fit_const, 11[180]: ', fd13)
    both_a_7790_220_2 = {'amp': amps11[220][0:53], 't': time_data_s[0:53]}
    fd14 = trap_fit_const(2, amps11[220],
        time_data_s, num_pumps, fit_thresh=0.9,
        tau_min=0.7e-6, tau_max=1.3e-2, pc_min = 0, pc_max = 2, offset_min=-10,
                            offset_max=10, both_a=both_a_7790_220_2)
    print('trap_fit_const, 11[220]: ', fd14)


    # # trap_fit: (41,15) for 200, 210, 220
    # fd8 = trap_fit(3, amps23[200],
    #     time_data_s, num_pumps, fit_thresh=0.7,
    #     tau_min=0.7e-6, tau_max=1.3e-2, tauc_min = 1e-10, tauc_max = 1e-7, offset_min=-2,
    #                         offset_max=2)
    # print("trap_fit, 23[200]: ", fd8)
    # fd9 = trap_fit(3, amps23[210],
    #     time_data_s, num_pumps, fit_thresh=0.6,
    #     tau_min=0.7e-6, tau_max=1.3e-2, tauc_min = 0, tauc_max = 1e-5, offset_min=-10,
    #                         offset_max=10)
    # print("trap_fit, 23[210]: ", fd9)
    # fd10 = trap_fit(3, amps23[220],
    #     time_data_s, num_pumps, fit_thresh=0.6,
    #     tau_min=0.7e-6, tau_max=1.3e-2, tauc_min = 0, tauc_max = 1e-5, offset_min=-10,
    #                         offset_max=10)
    # print("trap_fit, 23[220]: ", fd10)

    # # trap_fit:  (77,90) for 210, 220 needs sch1 12, sch2 11 for 210, 220
    # # could also the 'a' separation isn't working for those temps
    # both_a_7790 = {'amp': amps12[210][80:100], 't': time_data_s[80:100]}
    # fd11 = trap_fit(1, amps12[210],
    #     time_data_s, num_pumps, fit_thresh=0.6,
    #     tau_min=0.7e-6, tau_max=1.3e-2, tauc_min = 0, tauc_max = 1e-5, offset_min=-10,
    #                         offset_max=10, both_a = both_a_7790)
    # print('trap_fit, 12[210]: ', fd11)
    # both_a_7790_220 = {'amp': amps12[220][80:100], 't': time_data_s[80:100]}
    # fd12 = trap_fit(1, amps12[220],
    #     time_data_s, num_pumps, fit_thresh=0.6,
    #     tau_min=0.7e-6, tau_max=1.3e-2, tauc_min = 0, tauc_max = 1e-5, offset_min=-10,
    #                         offset_max=10, both_a = both_a_7790_220)
    # print('trap_fit, 12[220]: ', fd12)
    # both_a_7790_210_2 = {'amp': amps11[210][80:100], 't': time_data_s[80:100]}
    # fd13 = trap_fit(2, amps11[210],
    #     time_data_s, num_pumps, fit_thresh=0.6,
    #     tau_min=0.7e-6, tau_max=1.3e-2, tauc_min = 0, tauc_max = 1e-5, offset_min=-10,
    #                         offset_max=10, both_a = both_a_7790_210_2)
    # print('trap_fit, 11[210]: ', fd13)
    # both_a_7790_220_2 = {'amp': amps11[220][80:100], 't': time_data_s[80:100]}
    # fd14 = trap_fit(2, amps11[220],
    #     time_data_s, num_pumps, fit_thresh=0.6,
    #     tau_min=0.7e-6, tau_max=1.3e-2, tauc_min = 1e-10, tauc_max = 1e-7, offset_min=-10,
    #                         offset_max=10, both_a=both_a_7790_220_2)
    # print('trap_fit, 11[220]: ', fd14)