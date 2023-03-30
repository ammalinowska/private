import math
import pandas as pd
import numpy as np
import cv2
import time
import os


import matplotlib.pyplot as plt
from scipy import interpolate
import copy
from scipy.optimize import curve_fit



def delta_x(p1, p2):
    return p2[0] - p1[0]


def delta_y(p1, p2):
    return p2[1] - p1[1]


def EUCL_dist(p1, p2):
    """Calculate the absolute distance between two points in 2D"""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def EUCL_dist3d(p1, p2):
    """Calculate the absolute distance between two points in 3D"""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)


def theta(p1, p2):
    """Calculates the angle of the vector between the 2 points"""
    # sin_theta = delta_y(p1, p2)/EUCL_dist(p1, p2)
    # cos_theta = delta_x(p1, p2)/EUCL_dist(p1, p2)
    # theta_y = math.asin(sin_theta) * 180/math.pi
    # theta_x = math.acos(cos_theta) * 180/math.pi
    theta_tan = math.atan2(delta_y(p1, p2), delta_x(p1, p2)) * 180/math.pi

    return theta_tan

# def vector_dist(attributes, p1, p2):
#     """This function calculates the distance between points and adds sign that depends whether the point"""
#     try:
#         COM = np.array(attributes[:1],attributes[:2])
#         if EUCL_dist(p1, COM) > EUCL_dist(p2, COM):
#             sign = -1
#         else: sign = 1
#     except:
#         pass
#     return sign * math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def list_avg(list_in):
    return sum(list_in) / len(list_in)


def point_in_contour(contour, p):
    sign = cv2.pointPolygonTest(contour, p, measureDist=False)
    # print(sign)
    return sign


def reorder(attributes, start_index):
    """This function reorders a list by changing the start index and cut+pasting the entire part of the list
    before the start_index after the remaining part of the list.
    """

    try:
        reordered_attributes = np.concatenate(
            (attributes[start_index:], attributes[:start_index]))
        return reordered_attributes
    except:
        return attributes


def score_function(contour_1, contour_2):
    """This function determines the total sum of all distances between equally indexed points within two
    up following timeframes.
    """
    # for i in range(len(contour_1)):
    # print(f'contour1 {contour_1[i][:2]}')
    # print(f'contour2 {contour_2[i][:2]}')
    distances = [EUCL_dist(contour_1[i][:2], contour_2[i][:2])
                 for i in range(len(contour_1))]
    return sum(distances)


def score_function3D(contour_1, contour_2):
    """This function determines the total sum of all distances between equally indexed points within two
    up following timeframes.
    """
    # for i in range(len(contour_1)):
    # print(f'contour1 {contour_1[i][:3]}')
    # print(f'contour2 {contour_2[i][:3]}')
    a= min(len(contour_1), len(contour_2))
    distances = [EUCL_dist3d(contour_1[i][:3], contour_2[i][:3])
                 for i in range(a-1)]
    return sum(distances)


def stretch_func(arr, length):
    """This function stretches the input array to the target length provided as input.
    The stretching happens by creating duplicates of already existing elements in the list.
    """
    repetitions = np.round(np.linspace(0, length, arr.shape[0] + 1))[1:] - np.round(
        np.linspace(0, length, arr.shape[0] + 1))[:-1]
    repeated = np.repeat(arr, repetitions.astype(np.int), axis=0)

    return repeated


def load_file(filename, pixel_scale, sign):
    """This function loads csv file, rescales the curvature and changes the sign of the curvatures (this varies per dataset)"""
    print(filename)
    df = pd.read_csv(filename,
                     usecols=['sx', 'sy', 'curvature', 'mean(curvature)', 'contour_num', 'frame', 'area', 'length', 'comx', 'comy'])
    df['curvature'] = (df['curvature']) * pixel_scale * sign
    df['attributes'] = df[['sx', 'sy', 'curvature',
                           'frame', 'comx', 'comy']].values.tolist()

    frame_numbers = pd.unique(df.frame)

    all_frames = []
    for frame_value in frame_numbers:
        single_frame_df = df.loc[df['frame'] == frame_value]
        all_frames.append(single_frame_df.attributes.values.tolist())

    # cut_off_frames = all_frames[:cut_off_time]  # slice frames to equal length

    # lengths = [len(attribute_list) for attribute_list in cut_off_frames]

    # target_length = max(lengths)
    # stretched_frames = [stretch_func(np.array(attribute_list), target_length) for attribute_list in cut_off_frames]

    return all_frames


def correct_center(fram):
    '''
    Corrects coordinates to fit the center of the first crystal
    so that comx and comy are constant everywhere
    '''
    for i in range(len(fram)):
        for j in range(len(fram[i])):
            fram[i][j][0] = fram[i][j][0] - \
                fram[i][j][4]  # + fram[0][0][4]
            fram[i][j][1] = fram[i][j][1] - \
                fram[i][j][5]  # + fram[0][0][5]
            fram[i][j][4] = 0  # fram[0][0][4]
            fram[i][j][5] = 0  # fram[0][0][5]
    return fram


def crop_and_move(contours, frames, raw_frames, write_dir):

    n_contours = copy.deepcopy(contours)
    xmt = 0
    ymt = 0

    def get_mask(i):
        # draw contour
        mask = frames[i]*0

        for j in contours[i]:
            mask[int(j[1]), int(j[0])] = (255, 255, 255)

        # center of mass
        COM = (int(raw_frames[i][0][4]), int(raw_frames[i][0][5]))

        # fill to create mask
        cv2.floodFill(mask, None, COM, (255, 255, 255))

        blur = cv2.GaussianBlur(frames[i], (5, 5), 0)

        return blur*(mask//255), COM

    frame_y = [int(np.max(contours[:, :, 0])), int(np.min(contours[:, :, 0]))]
    frame_x = [int(np.max(contours[:, :, 1])), int(np.min(contours[:, :, 1]))]

    for i in range(len(contours)-1):
        start = time.time()
        # Get masks with center of mass (COM)
        mask_1, COM_1 = get_mask(i)
        mask_2, COM_2 = get_mask(i+1)


        # save mask
        # if not os.path.exists(f'{write_dir}/rawcrops'):
        #     os.makedirs(f'{write_dir}/rawcrops')
        # plt.figure()
        # plt.imshow(mask_1[frame_x[1]:frame_x[0], frame_y[1]:frame_y[0]]/np.max(frames[i]))
        # plt.savefig(f'{write_dir}/rawcrops/{str(i).zfill(3)}.png',
        #             bbox_inches="tight", dpi=300)
        # plt.close()

        # find biggest boundary of crystals
        max_ybound = int(max([abs(min(contours[i, :, 0]) - COM_1[0]),
                              abs(min(contours[i+1, :, 0]) - COM_2[0]),
                              abs(max(contours[i, :, 0]) - COM_1[0]),
                              abs(max(contours[i+1, :, 0]) - COM_2[0])]))
        max_xbound = int(max([abs(min(contours[i, :, 1]) - COM_1[1]),
                              abs(min(contours[i+1, :, 1]) - COM_2[1]),
                              abs(max(contours[i, :, 1]) - COM_1[1]),
                              abs(max(contours[i+1, :, 1]) - COM_2[1])]))


        # boundaries
        min_y = - max_ybound - 20
        max_y = max_ybound + 20
        min_x = - max_xbound - 20
        max_x = max_xbound + 20



        # first frame

        if np.any(np.array([int(COM_1[1])+min_x < 0, int(COM_1[1]) + max_x > mask_1.shape[1],
                  int(COM_1[0])+min_y < 0, int(COM_1[0])+max_y > mask_1.shape[0]])):
            print('This is a boundary crystal, cannot correct for drift')
            return np.array([])

        image_1 = mask_1[int(COM_1[1])+min_x:int(COM_1[1])
                         + max_x, int(COM_1[0])+min_y:int(COM_1[0])+max_y]/np.max(frames[i])

        # Second frame scan
        # loop over images for different x and y offsets
        differences = []

        # dr
        dCOM = np.round(
            np.sqrt((COM_1[0]-COM_2[0])**2 + (COM_1[1]-COM_2[1])**2))
        dr = max([3, int(dCOM)])

        # could be adjusted to be more efficient
        for x_i in range(-dr, dr):

            for y_i in range(-dr, dr):


                # check radius
                if np.sqrt(x_i**2 + y_i**2) > dr:
                    continue

                if np.any(np.array([int(COM_2[1])+min_x+x_i < 0, int(COM_2[1]) + max_x+x_i > mask_1.shape[1],
                          int(COM_2[0])+min_y+y_i < 0, int(COM_2[0])+max_y+y_i > mask_1.shape[0]])):
                    print('This is a boundary crystal, cannot correct for drift')
                    # continue
                    return np.array([])

                image_2 = mask_2[int(COM_2[1])+min_x+x_i:int(COM_2[1])
                                 + max_x+x_i, int(COM_2[0])+min_y+y_i:int(COM_2[0])+max_y+y_i]/np.max(frames[i+1])

                diff = cv2.sumElems(cv2.absdiff(image_1, image_2))[0]

                # if i % 25 == 0:
                #     if not os.path.exists(f'{write_dir}/scans'):
                #         os.makedirs(f'{write_dir}/scans')
                #     if not os.path.exists(f'{write_dir}/scans/scan_{i}'):
                #         os.makedirs(f'{write_dir}/scans/scan_{i}')
                #     plt.figure()
                #     plt.title(f'{diff}_{x_i}_{y_i}')
                #     plt.imshow(abs(image_1 - image_2))
                #     plt.savefig(f'{write_dir}/scans/scan_{i}/{str(len(differences)).zfill(3)}.png',
                #                 bbox_inches="tight", dpi=300)
                #     plt.close()
                differences.append(np.array([diff, x_i, y_i], dtype=object))

        # change parameters
        _, x_move, y_move = differences[np.argmin(np.array(differences)[:, 0])]
        xmt += x_move
        ymt += y_move

        # mianxs = max((frame_x[0] - frame_x[1]),
        #              (frame_y[0] - frame_y[1])) // 2 + 20
        # image_1 = mask_1[int(COM_1[1])-mianxs:int(COM_1[1])+mianxs,
        #                  int(COM_1[0])-mianxs:int(COM_1[0])+mianxs]/np.max(frames[i])
        #
        # # save relocated crops
        # if i == 0:
        #     if not os.path.exists(f'{write_dir}/relocatedcrops'):
        #         os.makedirs(f'{write_dir}/relocatedcrops')
        #     plt.figure()
        #     plt.imshow(image_1)
        #     plt.savefig(f'{write_dir}/relocatedcrops/{str(i).zfill(3)}.png',
        #                 bbox_inches="tight", dpi=300)
        #     plt.close()
        #
        # image_2 = mask_2[int(COM_2[1])-mianxs+xmt:int(COM_2[1])
        #                  + mianxs+xmt, int(COM_2[0])-mianxs+ymt:int(COM_2[0])+mianxs+ymt]/np.max(frames[i+1])
        # plt.figure()
        # plt.imshow(image_2)
        # plt.savefig(f'{write_dir}/relocatedcrops/{str(i+1).zfill(3)}.png',
        #             bbox_inches="tight", dpi=300)
        # plt.close()

        # fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
        # ax1.imshow(image_1)
        # ax2.imshow(image_2)
        # ax3.imshow(abs(image_1-image_2))
        # plt.show()

        # adjust contour coordinates
        n_contours[(i+1):, :, 0] = n_contours[(i+1):, :, 0] - \
            y_move + COM_1[0] - COM_2[0]
        n_contours[(i+1):, :, 1] = n_contours[(i+1):, :, 1] - \
            x_move + COM_1[1] - COM_2[1]

        print(time.time()-start, x_move, y_move)

    return n_contours


def closest_point(contour, point, i, P_F, P_I):
    # approx_loc = i * int(P_F/P_I)
    # print(approx_loc)
    distances = [EUCL_dist(contour[i], point)
                 for i in range(0, P_F)]

    return np.argmin(distances)


def parameters_eucl(k, dist):
    return sum(dist[0:k])/sum(dist)


def arclength_parameterization(data, N):

    data = data[:-2]
    data.append(data[0])

    dist = [EUCL_dist(data[i], data[i+1])
            for i in range(len(data)-1)]

    parameters = [parameters_eucl(i, dist)
                  for i in range(len(data))]

    # x and y interpolation
    s_i = 0  # 0 if we just want to connect the dots
    pnew = np.linspace(0, 1, num=N, endpoint=False)
    tck_x = interpolate.splrep(parameters, [i[0] for i in data], s=s_i)
    f_x = interpolate.splev(pnew, tck_x, der=0)
    tck_y = interpolate.splrep(parameters, [i[1] for i in data], s=s_i)
    f_y = interpolate.splev(pnew, tck_y, der=0)
    tck_c = interpolate.splrep(parameters, [i[2] for i in data], s=s_i)
    f_c = interpolate.splev(pnew, tck_c, der=0)

    f_x[-1] = f_x[0]
    f_y[-1] = f_y[0]
    f_c[-1] = f_c[0]

    # plt.figure()
    # plt.plot(f_x, f_y)
    # plt.scatter([i[0] for i in data], [i[1] for i in data], s=2)
    # plt.show()

    return np.array(list(zip(f_x, f_y, f_c))), [f_c]


def interpolate_points(f, N):
    dist2 = [EUCL_dist(f[i], f[i+1])
             for i in range(len(f)-1)]

    t_len = sum(dist2)
    data = [f[0]]
    p_len = 0
    for i in range(0, len(f)-1):
        p_len += dist2[i]
        if p_len >= t_len/N:
            data.append(f[i])
            p_len = p_len % (t_len/N)

    data.pop()
    # plt.figure()
    # plt.scatter([i[0] for i in data[:-1]], [i[1] for i in data[:-1]])
    # plt.scatter(data[-1][0], data[-1][1], s=2)
    # plt.show()
    return data


def read_scale(frames, images, cut_off_time, write_dir):
    """This function reads a csv file into a pandas dataframe. The dataframe is split into 3D lists. The 1st
    dimension is the entire dataframe. The 2nd dimension are the timeframes. The 3rd dimension are the attribute
    lists of individual points within a timeframe. These attributes contain coordinates and curvature, but can
    be altered to contain any information required from the dataframe.
    """
    # print(filename)
    # df = pd.read_csv(filename,
    #                  usecols=['sx', 'sy', 'curvature', 'mean(curvature)', 'contour_num', 'frame', 'area', 'length', 'comx', 'comy'])
    # df['curvature'] = (df['curvature']) * pixel_scale
    # df['attributes'] = df[['sx', 'sy', 'curvature', 'comx', 'comy']].values.tolist()
    #
    # frame_numbers = pd.unique(df.frame)
    #
    # all_frames = []
    # for frame_value in frame_numbers:
    #     single_frame_df = df.loc[df['frame'] == frame_value]
    #     all_frames.append(single_frame_df.attributes.values.tolist())

    cut_off_frames = frames[:cut_off_time]  # slice frames to equal length
    cut_off_images = images[:cut_off_time]  # slice images to equal length
    lengths = [len(attribute_list) for attribute_list in cut_off_frames]
    # print(lengths)

    # take biggest crystal as start point
    start_frame = np.argmax(lengths)
    POINTS_INT = 1500
    POINTS_FIT = 2000

    start = time.time()
    # create arclen fits for each frame
    interp_frames = []
    tcks = []
    for i in cut_off_frames:
        arc_output = arclength_parameterization(i, POINTS_FIT)

        interp_frames.append(np.array(arc_output[0]))
        tcks.append(arc_output[1])
    print(f'arclength took: {time.time()-start}s')
    start = time.time()
    # correct for drift by comparing pixel values
    interp_frames = crop_and_move(
        np.array(interp_frames), cut_off_images, cut_off_frames, write_dir)
    if interp_frames.any()==None:
        return np.array([])
    print(f'crop_and_move took: {time.time()-start}s')

    interp_points=[]
    # start_interp = interpolate_points(interp_frames[start_frame], POINTS_INT)
    for p in interp_frames:
        interp_p = interpolate_points(p, POINTS_INT)
        # print(interp_p)
        interp_points.append(np.array(interp_p))
    # print(interp_points)





    # # empty assigned frames list
    # assigned_frames = [[] for _ in range(len(interp_frames))]
    #
    # # if lengths[0] < lengths[-1]:
    # #     interp_frames = interp_frames[::-1]
    #
    # start = time.time()
    #
    # # interpolate points
    # assigned_frames[start_frame] = start_interp
    # for i in reversed(range(0, start_frame)):
    #     i_f = interp_frames[i]
    #     assigned_frames[i] = np.array([i_f[closest_point(i_f, j, k, POINTS_FIT, POINTS_INT)] for
    #                                    k, j in enumerate(assigned_frames[i+1])])
    #
    # if start_frame < len(interp_frames):
    #     for i in range(start_frame+1, len(interp_frames)):
    #         i_f = interp_frames[i]
    #         assigned_frames[i] = np.array([i_f[closest_point(i_f, j, k,
    #                                                          POINTS_FIT,
    #                                                          POINTS_INT)] for
    #                                        k, j in enumerate(assigned_frames[i-1])])

    # for i in range(1, len(assigned_frames)):
    # plt.figure()
    # plt.scatter([j[0] for j in assigned_frames[0]],
    #             [j[1] for j in assigned_frames[0]])
    #     plt.scatter([j[0] for j in assigned_frames[i]],
    #                 [j[1] for j in assigned_frames[i]])
    #     for k in range(len(assigned_frames[0])):
    #         plt.plot([assigned_frames[i][k][0], assigned_frames[i-1][k][0]],
    #                  [assigned_frames[i][k][1], assigned_frames[i-1][k][1]],
    #                  c='red')
    #     plt.ylim(450, 550)
    #     plt.xlim(600, 700)
    # plt.show()

    print(f'closest point assignment took: {time.time()-start}s')

    # if lengths[0] < lengths[-1]:
    #     stretched_frames = stretched_frames[::-1]
    #     interp_frames = interp_frames[::-1]
    # print(len(stretched_frames), [len(j) for j in stretched_frames])
    # return np.array(assigned_frames), interp_frames
    return interp_points, interp_frames


def get_curvatures(frames, time_step, TIME_FACTOR, IBP_CONCENTRATION, MUTANT, CRYSTAL_NUMBER, dir_pickle):
    # frames = [np.array(frames)]
    curv_all = pd.DataFrame()
    s_time = np.array(frames[0])
    start_time = s_time[:, 3][0]

    for i in range(0, len(frames), time_step):

        array = np.array(frames[i])

        # split attributes into lists
        curvature = array[:, 2]

        curvatures = pd.DataFrame(
            curvature, columns=[f'curvature {CRYSTAL_NUMBER}'])
        time = (start_time + i) * TIME_FACTOR * np.ones(len(curvature))

        curvatures[f'times {CRYSTAL_NUMBER}'] = time
        curv_all = curv_all.append(curvatures)

    np.save(f'{dir_pickle}/{IBP_CONCENTRATION}_{MUTANT}/curv_all_{CRYSTAL_NUMBER}.npy',
            curv_all.to_numpy())
    return curv_all


def lsw_m(t, k, m):
    return k ** (-1.0 / m) * t ** (1.0 / m)

def lsw_2(t, k):
    return k ** (-1 / 2) * t ** (1 / 2)


def lsw_3(t, k):
    return k ** (-1 / 3) * t ** (1 / 3)

def fit_fwhm(fwhm, sd_0, time):
    B = fwhm[0]/sd_0
    print(f'this is B: {B}')
    # b = 1/sd_0
    b = fwhm[0]

    def lsw_m(x, k, m):
        # return b ** (1.0 / m) + k * x ** (1.0 / m) - B
        return (x*k + b**m)*B

    xdata = time
    ydata = fwhm
    popt, pcov = curve_fit(lsw_m, xdata, ydata, maxfev=1000000, bounds=(0, [10, 5]))
    kd_m = popt[0]
    m = popt[1]
    print(f'this are the parameters: m = {m}, kd = {kd_m}')

    plt.figure()
    ax1 = plt.scatter(xdata, ydata, color='xkcd:magenta')
    # ax1.set_xlabel('Time [s]', fontsize=20)
    # ax1.set_ylabel(r'B/FWHMC$', fontsize=20)
    plt.plot(xdata, lsw_m(xdata, *popt), 'r-',
             label='fit R^m: k_d=%5.10f, m=%5.3f' % tuple(popt))
    plt.show()
