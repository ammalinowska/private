import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import os
# from scipy import stats
from scipy import spatial
# import trackpy as tp
# import pims
# import skimage.io as io
# from math import sqrt
# from pprint import pprint
# import itertools
# import sys
# from tkinter import filedialog
from tkinter import *
# import platform
# import subprocess
# from glob import glob
# import fullframe.plot_r3
# import fullframe.plot_k
# import fullframe.plot_A
# import fullframe.fit_data
# import fullframe.plot_Q
# import pickle
from itertools import combinations


class FrameImg:
    """This class handles the entire image processing. It takes as input all the captured frames, and
    outputs dataframes with crystal properties. These properties will be used in the crystal tracking
    to determine which crystals belong together."""

    ROI_crop = [50, 800, 400, 1240,]


    [ylow, yup, xleft, xright] = ROI_crop
    crop_boo = False

    def __init__(self, file_name, file_path=os.getcwd(), frame_num=1):
        self.file_name = file_name
        self.file_path = file_path
        self.frame_num = frame_num

        img, img_treshold = self.load_img()
        self.get_img_contours(img_treshold)
        self.process_contours()
        self.create_crystal_attr_list()
        self.check_contours()
        # self.drop_upper_outlier_area_crystal()
        # self.drop_edge_contours()

    def drop_upper_outlier_area_crystal(self):
        self.crystal_areas.sort()
        c_max = self.crystal_areas[len(self.crystal_areas) - 1]  # Get largest crystal area
        c_max_s = self.crystal_areas[len(self.crystal_areas) - 2]  # Get second largest crystal area
        area_ratio = c_max / c_max_s
        if area_ratio > 10:  # If largest area is 10 times that of the second largest areabluet
            max_crys = [c for c in self.crystalobjects if c.area == c_max][0]  # Retrieve crystal object
            print(f'Dropping max crystal, size is {round(area_ratio, 2)} times the second biggest crystal')
            # Remove crystal attributes from list, and the crystal form the crystalobjects list.
            self.crystal_areas.remove(max_crys.area)
            self.crystal_lengths.remove(max_crys.length)
            self.contours_lengths.remove(max_crys.contour_length)
            # self.crystal_centers.remove(max_crys.center_arr)
            # Currently not removing the center coord of max crys. Might cause issues late on
            self.crystalobjects.remove(max_crys)

    def load_img(self):
        """ Loads in the image, and crops if it crop_boo is set to True. """
        img_path = os.path.join(self.file_path, self.file_name)
        self.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if FrameImg.crop_boo:
            img = self.img[FrameImg.ylow:FrameImg.yup, FrameImg.xleft:FrameImg.xright]
        else:
            img = self.img
        self.img_attributes(img)
        img_denoised = self.denoise_img(img)
        img_treshold = self.tresholding_img(img_denoised)
        # self.plot_stages(img, img_denoised, img_treshold)
        # Add in possible plot stages method here before the images go out of scope.
        return img, img_treshold

    def plot_stages(self, img, img_denoised, img_treshold):
        images = [img, img_denoised, img_treshold]
        for i in range(len(images)):
            plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
            plt.xticks([]), plt.yticks([])
        plt.suptitle(self.file_name, fontsize=8)
        plt.show()

    def img_attributes(self, img):
        """ Retrieve image height and width and store them as an attribute. Used for checking if the
        contours are flipped """
        self.img_height, self.img_width = img.shape[:2]

    def denoise_img(self, img):
        """ Denoise image.
            Docs : https://docs.opencv.org/2.4/modules/photo/doc/denoising.html """
        return cv2.fastNlMeansDenoising(src=img, dst=None, h=10,
                                        templateWindowSize=1, searchWindowSize=27)

    def tresholding_img(self, img_denoised):
        ''' Treshold image. 
            Docs: https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3 '''
        return cv2.adaptiveThreshold(src=img_denoised, maxValue=255,
                                     adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                     thresholdType=cv2.THRESH_BINARY, blockSize=71, C=0.5)

    def get_img_contours(self, img_treshold):
        ''' Retrieve image contours. ## NEED TO WORK WITH RETR_CCOMP instead of RETR_EXTERNAL TO GET HIERACHY FOR DEALIG WITH INCEPTIONS
            Docs: https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a'''
        self.contours, self.hierarchy = cv2.findContours(img_treshold,
                                                         cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    def process_contours(self):
        ''' First creates two empty lists to store the crystals and the remaining/other objects. Next,
            loops through all the retrieves contours: If the len(contour), which is the amount of coordinate
            points is large enough, it creates a CrystalObject class instance. Depending on whether this is a contour
            with a child or not, child contour info will be passed to the crystalobject as well.
            If a contour is a child, no seperate object is created for those contour. Then, if the 'contour length'
            is greater than 2, it adds the CrystalObject to the list. If the conditions are not met, the object
            is stored in the other objects list.   '''
        self.crystalobjects = []
        self.otherobjects = []
        num_contours = len(self.contours)
        print(f'Number of contours found: {num_contours}')

        for i, contour in enumerate(self.contours):
            if len(contour) > 10:  # Checks the amount of coordinate points in the contour
                print(f'Processing img contours {i}/{num_contours}', end='\r')

                if self.hierarchy[0][i][3] == -1:  # Only create a crystal when it's not a hole contour
                    # Check whether contour is a parent, if so add the children contour as input as well to create hole
                    if self.hierarchy[0][i][2] != -1:
                        child_contour = self.contours[i + 1]
                        obj = CrystalObject(contour, self.hierarchy[0][i], child_contour,
                                            True, contour_num=i, frame_num=self.frame_num)
                    else:
                        obj = CrystalObject(contour, self.hierarchy[0][i], None,
                                            False, contour_num=i, frame_num=self.frame_num)

                    if obj.x_center == 0 or obj.y_center == 0:
                        self.otherobjects.append(obj)
                    else:
                        self.crystalobjects.append(obj)

        print('Contour processing done  .......')
        print(f'Number of contours stored: {len(self.crystalobjects)}')
        print(f'Number of other objects stored: {len(self.otherobjects)}')

    def create_crystal_attr_list(self):
        self.crystal_areas = [i.area for i in self.crystalobjects]
        self.crystal_lengths = [i.length for i in self.crystalobjects]
        self.contours_lengths = [i.contour_length for i in self.crystalobjects]
        self.crystal_centers = [i.center_arr for i in self.crystalobjects]

    def check_contours(self):
        """ Function that checks if the x and y axes have been swapped, and corrects this if so.
            Done by checking if the highest x found in the contour coordinates if higher than the image
            width, or the higest y contour coordinate is higher than the image height. 
            Correction is done for both the smoothed contours array, and the smoothed contours dataframe.
            Whether the action is performed or not is stored in the flipped contour attribute.  """
        self.y_maximum = max([i.s_contours_df['y'].max() for i in self.crystalobjects])
        self.x_maximum = max([i.s_contours_df['x'].max() for i in self.crystalobjects])
        if self.x_maximum > self.img_width or self.y_maximum > self.img_height:
            self.flipped_contours = True
            print(f'{self.x_maximum} vs {self.img_width}; {self.y_maximum} vs {self.img_height} ')
            for c in self.crystalobjects:
                # c.s_contours[:, 0], c.s_contours[:, 1] = c.s_contours[:, 1], c.s_contours[:, 0].copy()
                c.s_contours_df.rename(columns={'x': 'y', 'y': 'x', 'sx': 'sy', 'sy': 'sx'}, inplace=True)
        else:
            self.flipped_contours = False
        print(f'Contours flipped: {self.flipped_contours}')

    def drop_edge_contours(self):
        ''' Funtion to cutoff the crystals that are within a 'cutoff_pct' * the width and height of the img.
            First, creates a list of cutoff coordinate values for both axis, then loops through each crystal countour 
            coorinates to check if the cutoff coordinate values occur in the smoothed contours coordinates.
            If so, it removes the crystal from the crystalobjects list, and adds it to the edge_objects list. '''
        self.edge_objects = []
        cutoff_pct = 0.01

        cutoff_y_val = cutoff_pct * self.img_height
        y_drop = [self.img_height - i for i in range(0, int(cutoff_y_val))]
        y_drop.extend(list(range(0, int(cutoff_y_val))))

        cutoff_x_val = cutoff_pct * self.img_width
        x_drop = [self.img_width - i for i in range(0, int(cutoff_x_val))]
        x_drop.extend(list(range(0, int(cutoff_x_val))))
        pre_drop_count = len(self.crystalobjects)
        for crystal in reversed(self.crystalobjects):
            for valx, valy in zip(x_drop, y_drop):
                if valx in crystal.s_contours_df.sx.values or \
                        valy in crystal.s_contours_df.sy.values:
                    self.crystalobjects.remove(crystal)
                    self.edge_objects.append(crystal)
                    break  # to stop the loop   
        post_drop_count = len(self.crystalobjects)
        print(f'Edge dropping dropped {pre_drop_count - post_drop_count} crystals')

    def plot_contours(self, mark_center=False, mark_number=False,
                      save_image=False, file_name=f'contourplot1.png'):
        """ Plot the contours on the original image. """
        self.contoured_img = self.img
        num_crystal = len(self.crystalobjects)
        print(f'Total number of contours to plot: {num_crystal}')
        for i, crystal in enumerate(self.crystalobjects):
            print(f'Drawing contour of crystal {i} / {num_crystal}', end='\r')
            # Draw original contours in black
            cv2.drawContours(self.contoured_img, crystal.contour_raw,
                             -1, (0, 0, 0), 1)
            # Draw smoothed contours in white.
            cv2.drawContours(self.contoured_img, [crystal.s_contours.astype(int)],
                             -1, (255, 255, 255), 1)
            if mark_center:
                cv2.circle(self.contoured_img, (crystal.x_center, crystal.y_center), 1,
                           (255, 255, 255), -1)
            if mark_number:
                cv2.putText(self.contoured_img, f'{crystal.contour_num}', (crystal.x_center - 2,
                                                                           crystal.y_center - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            if save_image:
                cv2.imwrite(file_name, self.contoured_img)
        print('Crystal contour drawing done ..................')
        cv2.imshow(f'{self.file_name}', self.contoured_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plot_s_contours(self, show_plot=False,
                        save_image=False, frame_img_name='',
                        file_name=f'contourplot3.png'):
        ''' Plots the smoothed contours for each crystalobject in the frame on the original image.  '''
        fig = plt.figure()
        if FrameImg.crop_boo:
            org_img = mpimg.imread(os.path.join(self.file_path, self.file_name))[FrameImg.ylow:FrameImg.yup,
                      FrameImg.xleft:FrameImg.xright]
        else:
            org_img = mpimg.imread(os.path.join(self.file_path, self.file_name))
        plt.imshow(org_img)
        for crystal in self.crystalobjects:
            plt.scatter(crystal.s_contours[..., 0], crystal.s_contours[..., 1], s=0.05, c='black')
            plt.scatter(crystal.center_arr[0], crystal.center_arr[1], s=0.05, c='black')
            plt.xlabel('points (pixel)')
            plt.ylabel('points (pixel)')
        # fig.suptitle(frame_img_name, fontsize = 8)
        if save_image:
            fig.savefig(f'{file_name}')
        plt.close()


class CrystalObject:
    """The crystal object keeps track of all the attributes of a crystal belonging to a contour in single frame
        NB: there are two very similar dataframe functions for the case of regular crystal or crystal with hole."""

    def __init__(self, contour_raw, hierarchy, child_contour_raw, parent_bool, contour_num, frame_num):
        ###

    def get_center_point(self):
        """ Using the contour moments, retrieves the center x and y coordinates,
            and an array of said coordinates. Initially just calculates the COM of the outer contour. However, if there
            is a hole in the contour the COM will be recomputed with the hole taken into account"""

        # calculate crystal moments and center coords
        if self.moments['m00'] != 0:
            self.x_center = int(self.moments['m10'] / self.moments['m00'])
            self.y_center = int(self.moments['m01'] / self.moments['m00'])
        else:
            self.x_center = 0
            self.y_center = 0

        # in parent case, change the COM according to the hole
        if self.parent_bool == True:
            # calculate child moments and COM
            if self.child_moments['m00'] != 0:
                self.x_center_child = int(self.child_moments['m10'] / self.child_moments['m00'])
                self.y_center_child = int(self.child_moments['m01'] / self.child_moments['m00'])
            else:
                self.x_center_child = 0
                self.y_center_child = 0
            # calculate common COM, assuming uniform thickness and density
            A = cv2.contourArea(self.contour_raw)  # outer area
            a1 = cv2.contourArea(self.child_contour_raw)  # hole area
            a2 = A - a1  # crystal area
            print(f"before any computation the center of parent is ({self.x_center, self.y_center})")
            self.x_center = (A * self.x_center - a1 * self.x_center_child) / a2
            self.y_center = (A * self.y_center - a1 * self.y_center_child) / a2
            print(f"After computation the center of parent now is is ({self.x_center, self.y_center})")

        self.center_arr = np.array([self.x_center, self.y_center])

    def get_area(self):
        """ Sets the area of the contour. Two separate paths for regular crystals of crystals with holes"""
        if self.parent_bool == True:
            self.area = cv2.contourArea(self.contour_raw) - cv2.contourArea(self.child_contour_raw)
        else:
            self.area = cv2.contourArea(self.contour_raw)

    def get_length(self):
        """ Sets the 'length' of the contour. Two separate paths for regular crystals of crystals with holes"""
        if self.parent_bool == True:
            self.length = cv2.arcLength(self.contour_raw, True) + cv2.arcLength(self.child_contour_raw, True)
        else:
            self.length = cv2.arcLength(self.contour_raw, True)

    def calculate_curvature(self, df, smoothing):
        """ Calculates the curvature, ands its mean, of the curvature coordinates, and creates the sx and sy
            columns, which are the rounded values of the coordinates which used in the edge crystal removal function.
            This function starts by creating df columns of the first and second derivative, together with the helper
            columns. Next, creates the curvature and mean curvature columns, and THRESH_BINARY drops the created
            helper columns. Finally, it removes the padding rows (previously done by the 'listacrop' function). """
        for z in ['x', 'y']:
            df[f'd{z}'] = np.gradient(df[f'{z}'])
            df[f'd{z}'] = df[f'd{z}'].rolling(smoothing, center=True).mean()
            df[f'd2{z}'] = np.gradient(df[f'd{z}'])
            df[f'd2{z}'] = df[f'd2{z}'].rolling(smoothing, center=True).mean()
            df[f's{z}'] = df[f'{z}']  # .round(2)
        df['curvature'] = df.eval('(dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5')
        df['curvature'] = df.curvature.rolling(smoothing, center=True).mean()  # .round(2)
        self.mean_curvature = df['mean(curvature)'] = np.mean(abs(df.curvature))
        df = df.drop(['dx', 'd2x', 'dy', 'd2y'], axis=1)
        df = df.dropna()
        return df

    def set_contours_dataframe(self, smoothing=10):
        """
        Function to prepare the crystal dataframe in case of a regular crystal. Loads in the contours array, pads
        them (reason?), and then smooths the coordinates using rolling mean. Next, calls the calculate_curvature
        function in order to retrieve the curvature and its mean. Adds the contour number, frame, area, and length
        of the contour,and saves the resulting df. Finally, it creates an additional df with the rounded coordinates.
        """

        contours_raw = np.reshape(self.contour_raw, (self.contour_raw.shape[0], 2))
        contours = np.pad(contours_raw, ((20, 20), (0, 0)), 'wrap')
        df = pd.DataFrame(contours, columns={'x', 'y'})
        df = df.reset_index(drop=True).rolling(smoothing).mean().dropna()
        df = self.calculate_curvature(df, smoothing)
        df['contour_num'] = self.contour_num
        df['frame'] = self.frame_num
        df['area'] = self.area
        df['length'] = self.length
        df['comx'] = self.center_arr[0]
        df['comy'] = self.center_arr[1]
        self.s_contours_df = df

    def set_contours_child_dataframe(self, smoothing=10):
        """
        Function to prepare the crystal dataframe. Loads in the contours array, pads them (reason?), and then
        smooths the coordinates using rolling mean. Next, calls the calculate_curvature function in order to
        retrieve the curvature and its mean. Adds the contour number, frame, area, and length of the contour,
        and saves the resulting df. Finally, it creates an additional df with the rounded coordinates.
        """

        child_contours_raw = np.reshape(self.child_contour_raw, (self.child_contour_raw.shape[0], 2))
        child_contours = np.pad(child_contours_raw, ((20, 20), (0, 0)), 'wrap')
        df = pd.DataFrame(child_contours, columns={'x', 'y'})
        df = df.reset_index(drop=True).rolling(smoothing).mean().dropna()
        df = self.calculate_curvature(df, smoothing)
        df['contour_num'] = self.contour_num
        df['frame'] = self.frame_num
        df['area'] = self.area
        df['length'] = self.length
        df['comx'] = self.center_arr[0]
        df['comy'] = self.center_arr[1]
        self.s_contours_df_child = df


class CrystalRecog:
    def __init__(self, c_obj):
        """ Start off by creating the empty lists (which will be appended with the attributes
        for the specific crystal for each FrameImg. Then, add the attributes for the First 
        Frame to start off with. """
        self.s_contours_dfs = []
        self.raw_contours = []
        self.s_contours = []
        self.lengths = []
        self.areas = []
        self.center_arrays = []
        self.mean_curvatures = []
        self.c_count = 0
        self.frames_used = []
        self.parent_bool = []
        self.child_contours_raw = []
        self.x_center = c_obj.x_center
        self.y_center = c_obj.y_center
        self.hierarchy = c_obj.hierarchy
        self.count_num = c_obj.contour_num
        self.add_crystalobject(c_obj)

    def add_crystalobject(self, c_obj):
        """ Add crystal attributes to the respective class instances. """
        self.s_contours_dfs.append(c_obj.s_contours_df)
        self.raw_contours.append(c_obj.contour_raw)
        self.s_contours.append(c_obj.s_contours)
        self.lengths.append(c_obj.length)

        self.areas.append(c_obj.area)
        self.center_arrays.append(c_obj.center_arr)
        self.child_contours_raw.append(c_obj.child_contour_raw)
        self.parent_bool.append(c_obj.parent_bool)
        self.mean_curvatures.append(c_obj.s_contours_df['mean(curvature)'].min())
        self.frames_used.append(str(c_obj.frame_num))
        self.c_count += 1

    def retrieve_outer_bounds(self, padding_margin):
        max_y = self.s_contours_dfs[len(self.s_contours_dfs) - 1].y.max()
        min_y = self.s_contours_dfs[len(self.s_contours_dfs) - 1].y.min()
        y_padding = (max_y - min_y) * padding_margin
        max_y += y_padding
        min_y -= min_y - y_padding
        max_x = self.s_contours_dfs[len(self.s_contours_dfs) - 1].x.max()
        min_x = self.s_contours_dfs[len(self.s_contours_dfs) - 1].x.min()
        x_padding = (max_x - min_x) * padding_margin
        max_x += x_padding
        min_x -= x_padding
        return min_y, max_y, min_x, max_x

    def plot_contours_across_frames(self, file_count, output_img_dir):
        fig = plt.figure()
        fig.tight_layout()
        gs1 = fig.add_gridspec(nrows=3, ncols=2)
        fig.suptitle(t=f'#{self.count_num}; FU{self.c_count}/{file_count}', fontsize=12, va='top')
        fig_ax1 = fig.add_subplot(gs1[:-1, :])
        fig_ax1.title.set_text('Contours')
        fig_ax1.title.set_fontsize(12)
        for contour in self.s_contours:
            fig_ax1.scatter(contour[..., 0], contour[..., 1])
        fig_ax1.invert_yaxis()

        fig_ax2 = fig.add_subplot(gs1[-1, :-1])
        fig_ax2.title.set_text('Area')
        fig_ax2.plot(self.areas)
        fig_ax2.title.set_fontsize(10)

        fig_ax3 = fig.add_subplot(gs1[-1, -1])
        fig_ax3.title.set_text('Mean Curvature')
        fig_ax3.plot(self.mean_curvatures)
        fig_ax3.title.set_fontsize(10)

        frames_used = ','.join(self.frames_used)
        fig.text(0.02, 0.02, 'FU: ' + frames_used, color='grey', fontsize=4)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=0.90,
                            wspace=0.3, hspace=0.5)
        fig.savefig(os.path.join(output_img_dir, f'newtest_img{self.count_num}.png'))
        plt.close()


def closest(cur_pos, positions):
    """Get the euclidean distance to the closest point to current coordinate
        and store the closest point and its distance"""
    dist = spatial.distance.cdist([tuple(cur_pos)], positions)
    min_dist = dist.min()
    min_index = dist.tolist()[0].index(min_dist)
    closest_pos = positions[min_index]
    return closest_pos, min_index, min_dist


def EUCL_distance(p1, p2):
    """Calculate the distance between two points"""
    a = np.array(p1)
    b = np.array(p2)
    return np.linalg.norm(a - b)


def get_img_files_ordered(dir_i):
    """ Function returns a list of all input frames, ordered by their frame number, 
    together with a count of the total number of frames.  """
    img_files = []
    for file in os.listdir(dir_i):
        if not file.startswith('.') and file.lower().endswith((".png", ".gif")):
            try:
                file_i = {
                    'filename': file,
                    'file_num': int(file.split('s.')[0])
                }
                print(file_i)
                img_files.append(file_i)
            except IndexError:
                pass
    ordered_img_files = sorted(img_files, key=lambda k: k['file_num'])
    file_count = len(ordered_img_files)
    return ordered_img_files, file_count


def set_and_check_folder(FOLDER_NAME, create_boo=False):
    """ Function to set up a directory and check if it exists."""
    fol_path = os.path.join(os.getcwd(), FOLDER_NAME)
    if os.path.isdir(fol_path):
        return fol_path
    else:
        if create_boo:
            os.mkdir(fol_path)
            print(f'Dir {FOLDER_NAME} has been created.')
            return fol_path
        else:
            print(f'Dir {FOLDER_NAME} does not seem to exist. Terminating program.')
            sys.exit()


def create_frame_list(img_files, file_count, imgs_dir,
                      output_img_dir, IMAGE_FORMAT, plot_boolean=False):
    """ Function that loops through the input img files folder, and creates
    and instance of FrameImg for each image found with the correct image format.
    Optionally, plots all found contours on the inputted image. """
    frame_list = []
    for f_numerator, file in enumerate(img_files):
        file_name = file['filename']
        if file_name.endswith(IMAGE_FORMAT):
            print('------------------------------------------------------')
            print(f'Processing: {file_name}; #{f_numerator + 1}/{file_count}')
            frame_i = FrameImg(file_name, imgs_dir, f_numerator)
            frame_list.append(frame_i)
            if plot_boolean:
                frame_i.plot_s_contours(show_plot=False, save_image=True, frame_img_name=file_name,
                                        file_name=os.path.join(output_img_dir,
                                                               f'contour_plot_frame_{f_numerator + 1}{IMAGE_FORMAT}'))
        else:
            print(f'{file_name} has a different file format than the expected {IMAGE_FORMAT}.')
    return frame_list


def rename(dir_i):
    files = []
    lst = os.listdir(dir_i)
    lst.sort()
    lst.remove('.DS_Store')
    for i, file in enumerate(lst):
        os.rename(f'{dir_i}/{file}', f'{dir_i}/frame{i}.png')


def common_COM(p1, a1, p2, a2):
    """Function that returns that common COM of two objects based on the separate COM's
        and the areas of the objects"""
    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]
    A = a1 + a2

    x_common = (x1 * a1 + x2 * a2) / A
    y_common = (y1 * a1 + y2 * a2) / A

    return x_common, y_common


if __name__ == "__main__":
    start_time = time.time()
    from fullframe import baseplot as bp

    # Constants:
    IMAGE_FORMAT = '.png'
    INPUT_FOLDER_NAME = 
    IMAGE_OUTPUT_FOLDER_NAME = 
    CSV_EXPORT_FOLDER = 
    # INPUT_FOLDER_NAME = 
    # IMAGE_OUTPUT_FOLDER_NAME =
    # CSV_EXPORT_FOLDER =
    MAX_CENTER_DISTANCE = 10
    AREA_PCT = 1
    CENTER_PCT = 0.5
    MIN_PLOT_FRAMES = 1
    PLOT_FRAME_CONTOURS = True

    imgs_dir = set_and_check_folder(INPUT_FOLDER_NAME)
    output_img_dir = set_and_check_folder(IMAGE_OUTPUT_FOLDER_NAME, True)
    csv_export_dir = set_and_check_folder(CSV_EXPORT_FOLDER, True)
    # rename(imgs_dir) # when using the rename function, the program will still crash. Just run it again and it will work
    img_files, file_count = get_img_files_ordered(imgs_dir)
    frame_list = create_frame_list(img_files, file_count, imgs_dir,
                                   output_img_dir, IMAGE_FORMAT, PLOT_FRAME_CONTOURS)

    img_processing_time = time.time() - start_time  # Log time it took to process images.

    # Extract data from the frame list.
    times, Q = bp.calculate_volume_fraction(frame_list,
                                            output_img_dir)  # Timestamps and ice volume fractions per frame.
    N = bp.amount_of_crystals_per_ROIarea(frame_list, output_img_dir)  # Number of crystals per frame.
    A = bp.avg_crystal_area(frame_list, output_img_dir)  # Total area per frame.
    mr_k = bp.mode_mean_radius(frame_list, output_img_dir)  # Mode of the mean radius of curvature.
    r_k = bp.mean_radius_of_curvature(frame_list, output_img_dir)  # Mean radius of curvature.
    r_k3 = bp.mean_radius_of_curvature3(frame_list, output_img_dir)  # Mean radius cubed of curvature.
    l = bp.circumference(frame_list, output_img_dir)  # The circumferences.
    mean_r3_A_div_l = bp.r3_by_A_div_l(frame_list, output_img_dir)  # Mean radius.

    # Calculate and plot the distribution of area and radius.
    # A_distribution(frame_list)
    # r_distribution(frame_list)

    # Calculate area of ROI.
    ROI_area = frame_list[0].img_height * frame_list[0].img_width

    # Export all extracted quantities to a csv file.
    bp.export_quantities(times, Q, N, A, r_k, r_k3, mean_r3_A_div_l, l, ROI_area, mr_k, IMAGE_OUTPUT_FOLDER_NAME)

    # Fit parameters to the extracted data and plot the extracted data with their fits.

    """This is part that sums up all data for multiple experiments..."""
    # try:
    #     df_path = os.path.join(IMAGE_OUTPUT_FOLDER_NAME, os.path.basename(IMAGE_OUTPUT_FOLDER_NAME) + '.csv')
    #     df = pd.read_csv(df_path, index_col='index').dropna()  # Drop rows which have at least one NaN.
    #
    #     # Correct for faster playback speed. Only do this if you know you need this!
    #     # df.times = df.times * 13 / 21.52
    #     # df['time_corrected'] = True  # Mark current csv file as 'corrected for time'
    #
    #     # Perform fitting.
    #     print("Fitting parameters.")
    #     df = fit_data.fitting(df, df_path)
    #     fit_data.plot(df, df_path)
    #
    #     try:
    #         print("Plotting Qs.")
    #         Q_path = os.path.join(IMAGE_OUTPUT_FOLDER_NAME, os.pardir)
    #         df_Q = plot_Q.extract_Q(Q_path)
    #         plot_Q.plot_Q(df_Q, Q_path)
    #     except FileNotFoundError:
    #         print("Cannot plot Q's, because plot_Q.py is missing.")
    #
    #     try:
    #         print("Plotting As.")
    #         A_path = os.path.join(IMAGE_OUTPUT_FOLDER_NAME, os.pardir)
    #         df_A = plot_A.extract_A(A_path)
    #         plot_A.plot_A(df_A, A_path)
    #     except FileNotFoundError:
    #         print("Cannot plot A's, because plot_A.py is missing.")
    #
    #     try:
    #         # import plot_critical.py
    #         # print("Plotting r^3.")
    #         # r3_path = os.path.join(IMAGE_OUTPUT_FOLDER_NAME, os.pardir)
    #         # df_r3 = plot_critical.extract_r3(r3_path)
    #         # plot_critical.plot_r3(df_r3, r3_path)
    #
    #         r3_path = os.path.join(IMAGE_OUTPUT_FOLDER_NAME, os.pardir)
    #         df_r3 = plot_r3.extract_r3(r3_path)
    #         plot_r3.plot_r3(df_r3, r3_path)
    #     except FileNotFoundError:
    #         # print("Cannot plot r^3, because plot_critical.py is missing.")
    #         print("Cannot plot r^3, because plot_r3.py is missing.")
    #
    #     try:
    #         print("Plotting kd.")
    #         k_path = os.path.join(IMAGE_OUTPUT_FOLDER_NAME, os.pardir)
    #         df_k = plot_k.extract_k(k_path)
    #         plot_k.plot_k(df_k, k_path)
    #     except FileNotFoundError:
    #         print("Cannot plot k, because plot_k.py is missing.")
    # except FileNotFoundError:
    #     print("Cannot fit, because fit_data.py is missing.")

    # Create initial crystals
    crystal_tracking_list = []
    for i, obj in enumerate(frame_list[0].crystalobjects):
        crystal_tracking_list.append(CrystalRecog(obj))
    print('Frame # 1:')
    print(f'Used count: {len(crystal_tracking_list)}')

    # Start from 1 here, because frame 0 / the first frame already done above
    for i in range(1, len(frame_list)):
        print(f'Frame # {i + 1}:')
        c_central_list = frame_list[i].crystal_centers
        c_crystal_areas_list = frame_list[i].crystal_areas
    #
        pre_frame_center_coord_count = len(c_central_list)
    #     # loop through a copy because we need to potentially add something to the original list
        for target_index, target_crys in enumerate(list(crystal_tracking_list)):

            found_matching_crystal = False
            # find the coordinates, index of said coordinates, and distance to last center point
            closest_coord, index_closest, distance = \
                closest(target_crys.center_arrays[len(target_crys.center_arrays) - 1], c_central_list)
            if distance < MAX_CENTER_DISTANCE:  # False only if no double object has been added
    #
                # Find Crystal object corresponding to the central coord
                for crys in frame_list[i].crystalobjects:
                    if (crys.center_arr == closest_coord).all():
                        if crys.area * (1 - AREA_PCT) <= target_crys.areas[len(target_crys.areas) - 1] \
                                <= crys.area * (1 + AREA_PCT):
                            target_crys.add_crystalobject(crys)
                            found_matching_crystal = True
    #
    #         # no crystal match over the two frames was found
    #         if found_matching_crystal == False:
    #
    #             # assume a split is happening
    #             target_area = target_crys.areas[-1]
    #             target_center = target_crys.center_arrays[-1]
    #
    #             # check if in the next frame there is a combination of frames that equal the area of target_crys
    #             for area_combination in combinations(c_crystal_areas_list, 2):
    #
    #                 # if target_area * (1 - 0.5*AREA_PCT) <= sum(area_combination) <= target_area * (1 + 0.5*AREA_PCT):
    #                 if target_area * (1 - 0.5 * AREA_PCT) <= sum(area_combination) <= target_area * (
    #                         1 + 0.5 * AREA_PCT):
    #
    #                     # determine which crystals make up the combination
    #                     crystal_indices = [c_crystal_areas_list.index(area) for area in area_combination]
    #                     split_products = []
    #                     for crys in frame_list[i].crystalobjects:
    #                         for ind in crystal_indices:
    #                             if crys.area == c_crystal_areas_list[ind]:
    #                                 split_products.append(crys)
    #
    #                     # find the common COM of these two crystals
    #                     if EUCL_distance(split_products[0].center_arr, split_products[1].center_arr) <= np.sqrt(split_products[0].area + split_products[1].area):
    #                         split_products_COM = common_COM(split_products[0].center_arr, split_products[0].area,
    #                                                     split_products[1].center_arr, split_products[1].area)
    #
    #                         # check if the common COM meets the condition on the COM of target_crys and append both crystals
    #                         if EUCL_distance(target_center, split_products_COM) < CENTER_PCT * MAX_CENTER_DISTANCE:
    #
    #                             # find which of the two objects is most similar to the target according to area
    #                             if abs(target_area - split_products[0].area) <= abs(target_area - split_products[1].area):
    #                                 track_crystal = split_products[0]
    #                                 new_crystal = split_products[1]
    #                             else:
    #                                 track_crystal = split_products[1]
    #                                 new_crystal = split_products[0]
    #
    #                             # append this similar one to target
    #                             target_crys.add_crystalobject(track_crystal)
    #
    #                             # create CrystalRecog object of the other part and add to original tracking list
    #                             crystal_tracking_list.append(CrystalRecog(new_crystal))
    #
    #                             # c_crystal_areas_list.remove(split_products[0].area)
    #                             # c_crystal_areas_list.remove(split_products[1].area)
    #                             print(f'area combination: {area_combination}')
    #                             print(f'area combination sum: {sum(area_combination)}')
    #                             print(f'target area of {target_index}: {target_crys.areas[-1]}')
    #                             print(f"split input index and center coord: {target_index}, {target_center}")
    #                             print(f"I added both crystals of the split part for frame {i}")
    #                             print(f"let's start tracking {crystal_tracking_list[-1].center_arrays}")
    #                             print("------close split prints------")
    #                             break

        post_frame_center_coord_count = len(c_central_list)
        print('------------------------------------------------------')
        print(f'Frame # {i + 1}:')
        print(f'C coordinates went from {pre_frame_center_coord_count} to {post_frame_center_coord_count}  ')
        print(f'Used count: {pre_frame_center_coord_count - post_frame_center_coord_count}')
    crystal_linking_time = (time.time() - start_time) - img_processing_time  # Log time it took to link crystals

    crystal_tracking_count = len(crystal_tracking_list)
    for i, b in enumerate(crystal_tracking_list):

        print(f'Plotting Crystal {i}/{crystal_tracking_count}', end='\r')
        if b.c_count > MIN_PLOT_FRAMES:

            space_scale = 86.7 * 10 ** (-3)  # um
            gamma_0 = 29.8  # mJ/m^2
            d_tolman = 0.24 * 10 ** (-9)  # m
            solution_thickness = 2 * 10 ** (-6)

            fig = plt.figure()
            fig.tight_layout()
            gs1 = fig.add_gridspec(nrows=2, ncols=2)
            fig_ax1 = fig.add_subplot(gs1[0, 0])
            fig_ax1.title.set_text('Contours')

            for contour in b.s_contours:
                fig_ax1.scatter(contour[..., 0], contour[..., 1], s=1)

            fig_ax1.invert_yaxis()
            fig_ax1.title.set_fontsize(12)
            fig.suptitle(t=f'#{b.count_num}; FU{b.c_count}/{file_count}; '
                           f'start frame {b.frames_used[0]}', fontsize=10, va='top')

            fig_ax2 = fig.add_subplot(gs1[0, 1])
            fig_ax2.title.set_text('Area')
            fig_ax2.plot(np.asarray(b.areas) * space_scale * space_scale)
            fig_ax2.set_ylabel('area [um^2]', fontsize=10)
            fig_ax2.set_xlabel('time', fontsize=10)
            fig_ax2.title.set_fontsize(10)

            fig_ax3 = fig.add_subplot(gs1[1, 0])
            fig_ax3.title.set_text('Mean curvatures')
            fig_ax3.plot(np.asarray(b.mean_curvatures) / space_scale)
            fig_ax3.set_ylabel('mean curvature [1/um]', fontsize=10)
            fig_ax3.set_xlabel('time', fontsize=10)
            fig_ax3.title.set_fontsize(10)

            fig_ax4 = fig.add_subplot(gs1[1, 1])
            fig_ax4.title.set_text('Gibbs surface energy')
            G = 2 * gamma_0 * np.asarray(b.areas) * space_scale * space_scale * 10 ** (-12) + \
                (gamma_0 * np.asarray(b.lengths) * space_scale * 10 ** (-6) * solution_thickness *
                 (1 - ((np.asarray(b.mean_curvatures) / (space_scale * 10 ** (-6))) * 2 * d_tolman)))
            fig_ax4.plot(G)
            fig_ax4.set_ylabel('G_total [mJ]', fontsize=10)
            fig_ax4.set_xlabel('time', fontsize=10)
            fig_ax4.title.set_fontsize(10)

            frames_used = ','.join(b.frames_used)
            fig.text(0.02, 0.02, 'FU: ' + frames_used, color='grey', fontsize=4)
            fig.subplots_adjust(left=None, bottom=None, right=None, top=0.90,
                                wspace=0.3, hspace=0.5)
            fig.savefig(os.path.join(output_img_dir, f'{b.count_num}.pdf'))
            plt.close()

            df_i = pd.concat(b.s_contours_dfs)
            csv_file_name = f'{b.count_num}.csv'
            csv_export_dir_i = os.path.join(csv_export_dir, csv_file_name)
            df_i.to_csv(csv_export_dir_i)

    print('######################################################')
    print(f'img processing time: {img_processing_time} ')
    print(f'Crystal linking time : {crystal_linking_time}')
    print("Total runtime --- %s seconds ---" % (time.time() - start_time))  # To see how long program
    print('######################################################')
