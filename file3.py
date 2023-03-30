import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile
import os
import time
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t




def import_Fcurves(folder, excluded=[]):
    files = os.listdir(folder)
    all_data = {}
    for file in files:
        if file.endswith('.csv'):
            if file.split('_')[1] in excluded:
                continue
            data = pd.read_csv(folder + file, index_col=0).T.to_numpy()
            if np.any(np.isnan(data[0])):
                continue
            all_data[file.split('.csv')[0]] = data
    return all_data


def average_molecules(fc_curves, -, curve='B'):
    f_total = []
    d_total = []
    for key in fc_curves.keys():
        vars = key.split('_')
        if vars[0] == curve and vars[1] in ex_concs:
            f_total.append(np.array(fc_curves[key][0]))
            d_total.append(np.array(fc_curves[key][1]))
    f_average = np.average(np.array(f_total, dtype='object'), axis=0)
    f_std = np.std(np.array(f_total), axis=0)
    d_average = np.average(np.array(d_total, dtype='object'), axis=0)
    d_std = np.std(np.array(d_total), axis=0)
    return np.array([f_average, d_average]), np.array([f_std, d_std])


def convert_to_extensions(fc_curves):

    extensions = {}
    for key in fc_curves.keys():
        vars = key.split('_')
        if vars[0] == 'B':
            continue
        else:
            d_new = fc_curves[key][1] - fc_curves[f'B_{vars[1]}_{vars[2]}'][1]
            extensions[key] = [fc_curves[key][0], d_new]
    return extensions


def mean_fd(FD_curves, type):

    step = 1.5
    x_filtered = np.arange(1, 52+step, step)
    y_filtered = [[] for i in x_filtered]
    for i in FD_curves.keys():
        if type in i:
            x = FD_curves[i][0]
            y = FD_curves[i][1]
            x = x[y < 18]
            y = y[y < 18]

            for i in range(len(x_filtered)):
                for j in y[(x > x_filtered[i]-step/2) & (x < x_filtered[i]+step/2)]:
                    y_filtered[i].append(j)

    y_filtered = [np.mean(i) for i in y_filtered]

    return [x_filtered, y_filtered]


def make_fit_plot(params, fit_f, plotname, extended=True):

    # create figure
    # plt.rc('font', family='serif')
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    # all concentrations
    ex_conc_list = []
    for key in FC_curves.keys():
        if key.split('_')[1] not in ex_conc_list:
            ex_conc_list.append(key.split('_')[1])

    # average curve scatters
    for key in FC_curves.keys():
        vars = key.split('_')
        average_curve, std_curve = average_molecules(
            FC_curves, [vars[1]], vars[0])
        if vars[0] == plotname:
            ax.errorbar(average_curve[1], average_curve[0],
                        yerr=std_curve[0],
                        xerr=std_curve[1],
                        fmt='.',
                        c=colors[vars[1]], label=vars[1])
    bare_FC, bare_std = average_molecules(FC_curves, ex_conc_list, 'B')
    ax.errorbar(bare_FC[1], bare_FC[0],
                yerr=bare_std[0],
                xerr=bare_std[1],
                fmt='.',
                c=colors['B'], label='bareDNA')
    if 'YO' in plotname:
        yo_curve, std_yo = average_molecules(FC_curves, ex_conc_list, 'YO')
        ax.errorbar(yo_curve[1], yo_curve[0],
                    yerr=std_yo[0],
                    xerr=std_yo[1],
                    fmt='.',
                    c=colors['0'], label='0.0')

    # fit plots
    if extended:
        # Bare FD
        fdata = mean_fd(FD_curves, 'B')
        ax.plot(fdata[1], fdata[0], c=colors['B'], label='bareDNA')
    else:
        fdata = [np.arange(1, 53.5, 1.5), np.arange(1, 53.5, 1.5) * 0]
        baredna = fit_f(fdata[0], *params, 0, 0)
        ax.plot(baredna, fdata[0], c=colors['B'], label='bareDNA')
    for i in ex_conc_list:
        # (YO) ISO fits
        if plotname == f'YO{ISOMER}':
            I1 = YO_CONC
        else:
            I1 = 0
        theta, exst = fit_f(fdata[0], *params, I1, float(i), True)
        ax.plot(exst + fdata[1], fdata[0], c=colors[i], label=i)

        # save thetas
        np.save(file_path + f'thetafiles/{MODEL}_{plotname}_{i}', theta)
    if 'YO' in plotname:
        # YO fit
        t_yo, exst_yo = fit_f(
            fdata[0], *params, YO_CONC, 0, True)
        ax.plot(exst_yo + fdata[1], fdata[0], c=colors['0'])
        np.save(file_path + f'thetafiles/{MODEL}_YO{ISOMER}_0', t_yo)

    # edit figure
    [x.set_linewidth(1.5) for x in ax.spines.values()]
    ax.tick_params(length=6, width=1.5, direction='in')
    ax.set_xlabel('Distance [\u03BCm]')
    ax.set_ylabel('Force [pN]')
    ax.set_xlim(12, 23)
    ax.set_ylim(0, 50)

    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # save figure
    fig.savefig(
        f'{file_path}figures/fit_{MODEL}_{plotname}.pdf',
        bbox_inches="tight", transparent=True,  dpi=300)
    plt.close()

    return


def make_theta_plot(name, plotname):

    for n in range(len(name)):

        # create figure
        # plt.rc('font', family='serif')
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)

        # x data
        x = np.arange(1, 53.5, 1.5)

        # read theta data
        files = os.listdir(f'{file_path}thetafiles/')
        for file in files:

            vars = file.split('.npy')[0].split('_')
            if vars[0] == MODEL and vars[1] == plotname:
                y = np.load(f'{file_path}thetafiles/{file}')
                ax.plot(x, y[n], c=colors[vars[2]], label=vars[2])


        # edit figure
        [x.set_linewidth(1.5) for x in ax.spines.values()]
        ax.tick_params(length=6, width=1.5, direction='in')
        ax.set_ylabel('Coverage')
        ax.set_xlabel('Force [pN]')
        ax.set_xlim(5, 50)
        ax.set_ylim(0, 1)
        ax.legend()

        # save figure
        fig.savefig(
            f'{file_path}figures/thetas_{MODEL}_{plotname}_{name[n]}.pdf',
            bbox_inches="tight", transparent=True, dpi=300)
        plt.close()

    return


def eq_cmlb2(f, K01, n1, xeq1, K02, n2, xeq2, I1, I2, t_r=False):
    -

    return dL


def eq_cmlb3(f, K01, n1, xeq1, K02, xeq2, n2, K03, xeq3, n3, I1, I2, t_r=False):

    -

    return dL


def eq_cmlb3iso(f, K01, n1, xeq1, K02, xeq2, n2, K03, xeq3, n3, I1, I2, t_r=False):

    # fixed params
    # xeq1 = 0.34
    xeq3 = 0
    # xeq2 = 0.34
    # n2 = 11
    # K02 = 4457.105
    # xeq2 = 0.259
    # n3 = 11
    K02 = 6841.768
    n2 = 1
    xeq2 = 0.222

    # n2 = 3.22
    # n3 = 3.22

    # convert concentrations
    I_iso = I2
    I1 = I1 * 10**-9
    I2 = I_iso * 0.78 * 10**-6
    I3 = I_iso * 0.22 * 10**-6

    # Theta calculations
    -
    # extension
    dL = T1*xeq1*Nbp/n1 + T2*xeq2*Nbp/n2 + T3*xeq3*Nbp/n3

    if t_r:
        return [[T1, T2, T3], dL]

    return dL


def eq_atgc(f, K01, n1, xeq1, K02, n2, xeq2, K03, n3, K04, I1, I2, t_r=False):

    # fixed params
    # xeq1 = 0.34
    xeq3 = 0
    n2 = 1
    n3 = 11


    # part of the DNA that is GC rich and AT rich
    fGC = 1/2
    fAT = 1/2

    # convert concentrations
    I1 = I1 * 10**-9
    I2 = I2 * 10**-6
    # K04 = 6075615

    # calculate thetas
    -

    Ty = fAT * TyAT + fGC * TyGC
    Ti = fAT * TiAT + fGC * TiGC
    Tg = fAT * TgAT + fGC * TgGC

    # extension
    dL = Ty*xeq1*Nbp/n1 + Ti*xeq2*Nbp/n2 + Tg*xeq3*Nbp/n3

    """Get coverages for AT/GC rich regions, in reality max 60% of GC or AT content"""
    real_TgGC = TgAT * 0.4 + TgGC * 0.6
    real_TgAT = TgAT * 0.6 + TgGC * 0.4
    real_TiGC = TiAT * 0.4 + TiGC * 0.6
    real_TiAT = TiAT * 0.6 + TiGC * 0.4
    total_AT = real_TiAT + real_TgAT
    total_GC = real_TiGC + real_TgGC

    if t_r:
        return [[Ty, Ti, Tg, real_TgAT, real_TgGC, real_TiAT, real_TiGC, total_AT, total_GC], dL]

    return dL


def eq_atgc_dim(f, K02, n2, xeq2, K03, n3, xeq3, K04, K_dim, K05, n5, xeq5, I1, I2, t_r=False):
    """Model that corrects for sequence differential groove-binding, intercalation
    mode and dimerisation of ligands (and groove-binding of the dimers)"""
    # Na = 6.02 * 10**23
    # fixed params
    K01 = 915028.000
    n1 = 3.218
    xeq1 = 0.347
    xeq3 = 0.01
    n2 = 2.5
    xeq2 = 0.34
    n3 = 4
    # K_dim = 2.8 * 10**5
    xeq5  = 0.02
    n5 = 8


    # part of the DNA that is GC rich and AT rich
    fGC = 1/2
    fAT = 1/2

    # convert concentrations
    I1 = I1 * 10**-9
    I2 = I2 * 10**-6
    I2 = (np.sqrt(8 * K_dim * I2 + 1) - 1) / (4 * K_dim)
    print("this is monomer conc:", I2)

    Id = I2**2 * K_dim #concentration of the dimers
    print("this is dimer conc:", Id)
    # Id = I2 * Kd /(2 * Na)
    # Im = I2 - 2 * Id
    # print(Im)
    # K04 = 6075615

    # calculate thetas
    -


    Ty = fAT * TyAT + fGC * TyGC
    Ti = fAT * TiAT + fGC * TiGC
    Tg = fAT * TgAT + fGC * TgGC

    # extension
    dL = Ty*xeq1*Nbp/n1 + Ti*xeq2*Nbp/n2 + Tg*xeq3*Nbp/n3 + Tgd*xeq5*Nbp/n5

    """Get coverages for AT/GC rich regions, in reality max 60% of GC or AT content"""
    real_TgGC = TgAT * 0.4 + TgGC * 0.6
    real_TgAT = TgAT * 0.6 + TgGC * 0.4
    real_TiGC = TiAT * 0.4 + TiGC * 0.6
    real_TiAT = TiAT * 0.6 + TiGC * 0.4
    total_AT = real_TiAT + real_TgAT
    total_GC = real_TiGC + real_TgGC

    if t_r:
        return [[Ty, Ti, Tg, Tgd, real_TgAT, real_TgGC, real_TiAT, real_TiGC, total_AT, total_GC], dL]

    return dL


def eq_atgcX(f, K01, n1, xeq1, K02, n2, xeq2, K03, n3, xeq3, K04, I1, I2, t_r=False):
    """AT-GC dependent grrove binding that introduces extension but is force independent"""

    # fixed params
    # K01 = 1115182.979
    # n1 = 2.593
    # xeq1 = 0.298
    xeq2 = 0.34
    # xeq3 = 0.01
    n2 = 0.1
    # n3 = 0.1

    # part of the DNA that is GC rich and AT rich
    fGC = 1/2
    fAT = 1/2

    # convert concentrations
    I1 = I1 * 10**-9
    I2 = I2 * 10**-6

    # calculate thetas
    -

    Ty = fAT * TyAT + fGC * TyGC
    Ti = fAT * TiAT + fGC * TiGC
    Tg = fAT * TgAT + fGC * TgGC

    # extension
    dL = Ty*xeq1*Nbp/n1 + Ti*xeq2*Nbp/n2 + Tg*xeq3*Nbp/n3



    if t_r:
        return [[Ty, Ti, TgAT, TgGC], dL]

    return dL


def eq_atgc_dna(f, l0, Lp0, St0, K01, xeq1, n1, K02, xeq2, Lp2, St2, K03, xeq3, n3,  Lp3, St3, K04, I1, I2, t_r=False):

    l0 = 0.342
    Lp0 = 39.565
    St0 = 2952.159
    Lp1 = Lp0
    St1 = St0
    Lp2 = Lp0
    St2 = St0
    # Lp3 = Lp0
    # St3 = St0
    # K01 = 1115182.979
    # n1 = 2.593
    xeq1 = 0.298
    n2 = 1
    n3 = 11
    xeq3 = 0
    # xeq2 = 0.34


    -

    fGC = 1/2
    fAT = 1/2
    I_1 = I1 * 10**(-9)
    I_2 = I2 * 10**(-6)

    -

    Ty = fAT * TyAT + fGC * TyGC
    Ti = fAT * TiAT + fGC * TiGC
    Tg = fAT * TgAT + fGC * TgGC
    Tb = 1 - Ty - Ti - Tg

    # Hyperstretching paper S6
    -

    if t_r:
        return [[Ty, Ti, TgAT, TgGC], L]

    return L


def global_fitting(all_d, fit_f, p_0, r_max, names, folder, bnds=[0, np.inf]):

    # combine data
    x = [xi for key in all_d.keys() for xi in all_d[key][0][1:r_max]]
    y = [yi for key in all_d.keys() for yi in all_d[key][1][1:r_max]]

    def combined_function(data, *params):

        xtot = 0
        results = []
        for key in all_d.keys():
            xnew = all_d[key][0][1:r_max]
            data2 = data[xtot:len(xnew)+xtot]
            vars = key.split('_')

            # get concentrations
            I1 = YO_CONC
            I2 = float(vars[1])
            if vars[0] == 'YO':
                I2 = 0
            elif vars[0] == ISOMER:
                I1 = 0
            elif vars[0] == 'B':
                I1 = 0
                I2 = 0

            result = fit_f(data2, *params, I1, I2)
            results = np.append(np.copy(results), result)

            xtot += len(xnew)

        return results

    # run global fit
    popt, pcov = curve_fit(combined_function, x, y, p0=p_0, bounds=bnds,
                           method='trf', maxfev=100000000)
    errors = np.sqrt(np.diagonal(pcov))

    # save parameters
    with open(f'{folder}parameters/{MODEL}_params.txt', 'w') as txtf:
        for i in range(len(popt)):
            txtf.write(f'{names[i]} = %.3f +/-' % popt[i]
                       + ' %.2f' % errors[i] + '\n')
            print(f'{names[i]} = %.3f +/-' % popt[i]
                  + ' %.2f' % errors[i])

    return popt, errors


if __name__ == '__main__':
    plt.rcParams['pdf.fonttype'] = 42   #get editable figure text
    # plt.rcParams['ps.fonttype'] = 42    #get editable figure text

    file_path = '/Users/agatamalinowska/Desktop/MDA/H_33342/'
    MODELS = ['atgc_dim']

    # file_path = './datadump/'
    # MODELS = ['cmlb3', 'atgc', 'atgcdna']

    # colors to be changed
    colors = {'B': '#737AB1', '0.001':'red', '0.01': 'blue', '0.1': 'orange',
              '1.0': 'green', '10.0': 'pink',
              '100.0': 'purple', '0': 'black'}

    # (initial) parameters
    ISOMER = 'H'
    YO_CONC = 20
    Nbp = 48.502
    kT = 4.11

    # load FC/FD curves
    FC_curves = import_Fcurves(file_path + 'FC_curves/')
    FD_curves = import_Fcurves(file_path + 'FD_curves/')

    # extension data
    extensions = convert_to_extensions(FC_curves)

    l0 = 0.342
    Lp0 = 39.565
    St0 = 2952.159
    Lp3 = 39.565
    Lp2 = 39.565
    St3 = 2952.159
    St2 = 2952.159
    K01 = 915028
    xeq1 = 0.317
    n1 = 3.22
    K02 = 2000
    xeq2 = 0.2
    n2 = 1
    K03 = 60756150000
    xeq3 = 0
    n3 = 4
    n5 = 8
    K04 = K03/10
    K02 = 2000
    K_dim = 2.8 * 10**18
    K05 = 2000
    xeq5 = 0

    for MODEL in MODELS:

        # select depending on model
        if MODEL == 'cmlb3':
            name = ['YO', f'{ISOMER}_int', f'{ISOMER}_gr']
            params = [K01, n1, xeq1, K02, xeq2, n2, K03, xeq3, n3]
            names = ['K01', 'n1', 'xeq1', 'K02', 'xeq2', 'n2', 'K03', 'xeq3', 'n3']
            fit_func = eq_cmlb3
            fit_data = extensions
            max_r = -1
            extension = True
        elif MODEL == 'cmlb3iso':
            name = ['YO', f'{ISOMER}_E', f'{ISOMER}_Z']
            params = [K01, n1, xeq1, K02, xeq2, n2, K03, xeq3, n3]
            names = ['K01', 'n1', 'xeq1', 'K02', 'xeq2', 'n2', 'K03', 'xeq3', 'n3']
            fit_func = eq_cmlb3iso
            fit_data = extensions
            max_r = -1
            extension = True
        elif MODEL == 'atgc':
            name = ['YO', f'{ISOMER}_int', f'{ISOMER}_gr',
                    f'{ISOMER}_gr_AT', f'{ISOMER}_gr_GC', f'{ISOMER}_int_AT',
                    f'{ISOMER}_int_GC', f'{ISOMER}_total_AT', f'{ISOMER}_total_GC']
            params = [K01, n1, xeq1, K02, n2, xeq2, K03, n3, K04]
            names = ['K01', 'n1', 'xeq1', 'K02', 'n2', 'xeq2', 'K03', 'n3', 'K04']
            fit_func = eq_atgc
            fit_data = extensions
            max_r = -1
            extension = True
        elif MODEL == 'atgcX':
            name = ['YO', f'{ISOMER}_int',
                    f'{ISOMER}_gr_AT', f'{ISOMER}_gr_GC']
            params = [K01, n1, xeq1, K02, n2, xeq2, K03, n3, xeq3, K04]
            names = ['K01', 'n1', 'xeq1', 'K02', 'n2', 'xeq2', 'K03', 'n3', 'xeq3', 'K04']
            fit_func = eq_atgcX
            fit_data = extensions
            max_r = -1
            extension = True
        elif MODEL == 'atgcdna':
            name = ['YO', f'{ISOMER}_int',
                    f'{ISOMER}_gr_AT', f'{ISOMER}_gr_GC']
            params = [l0, Lp0, St0, K01, n1, xeq1, K02, xeq2, Lp2, St2, K03, xeq3, n3, Lp3, St3, K04]
            names = ['l0', 'Lp0', 'St0', 'K01', 'n1', 'xeq1',
                     'K02', 'xeq2', 'Lp2', 'St2', 'K03', 'xeq3', 'n3', 'Lp3', 'St3', 'K04']
            fit_func = eq_atgc_dna
            fit_data = FC_curves
            max_r = 6
            extension = False
        elif MODEL == 'cmlb2':
            name = ['YO', ISOMER]
            params = [K01, n1, xeq1, K02, n2, xeq2]
            names = ['K01', 'n1', 'xeq1', 'K02', 'n2', 'xeq2']
            fit_func = eq_cmlb2
            fit_data = extensions
            max_r = -1
            extension = True
            # bounds = [[0, 6, 0], [np.inf, np.inf, np.inf]]
        elif MODEL =='atgc_dim':
            name = ['YO', f'{ISOMER}_int', f'{ISOMER}_gr', f'{ISOMER}_gr_dim',
                    f'{ISOMER}_gr_AT', f'{ISOMER}_gr_GC', f'{ISOMER}_int_AT',
                    f'{ISOMER}_int_GC', f'{ISOMER}_total_AT', f'{ISOMER}_total_GC']
            params = [K02, n2, xeq2, K03, n3, xeq3, K04, K_dim, K05, n5, xeq5]
            names = ['K02', 'n2', 'xeq2', 'K03', 'n3', 'xeq3', 'K04', 'K_dim', 'K05','n5', 'xeq5']
            fit_func = eq_atgc_dim
            fit_data = extensions
            max_r = -1
            extension = True


        print(MODEL)

        # do the fit
        params, errors = global_fitting(fit_data, fit_func, params, max_r,
                                        names, file_path)

        # create figures
        make_fit_plot(params, fit_func, ISOMER, extension)
        make_fit_plot(params, fit_func, f'YO{ISOMER}', extension)
        make_theta_plot(name, ISOMER)
        make_theta_plot(name, f'YO{ISOMER}')
