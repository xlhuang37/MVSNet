from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
import scipy
import torch 
import cv2



def read_mask(mask_filename):
        img = cv2.imread(mask_filename, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) - np.min(img)) / (np.max(img) - np.min(img)) 
        img = img[:, :-4]
        return img

# the DTU dataset preprocessed by Yao Yao (only for training)
class DiligentDataset(Dataset):
    def __init__(self, obj ="bear", mode = "test", lighting = 2, nviews = 5, ndepths=192, interval_scale=1.06, **kwargs):
        super(DiligentDataset, self).__init__()
        self.obj = obj
        self.imgpath = {"bear" : "/playpen-nas-ssd/xiaolong/data/DiLiGenT-MV/mvpmsData/bearPNG"
                            ,"buddha" : "/playpen-nas-ssd/xiaolong/data/DiLiGenT-MV/mvpmsData/buddhaPNG"
                            ,"cow" : "/playpen-nas-ssd/xiaolong/data/DiLiGenT-MV/mvpmsData/cowPNG"
                            ,"pot2" : "/playpen-nas-ssd/xiaolong/data/DiLiGenT-MV/mvpmsData/pot2PNG"
                            , "reading" : "/playpen-nas-ssd/xiaolong/data/DiLiGenT-MV/mvpmsData/readingPNG"}   
        self.calib_path = f"/playpen-nas-ssd/xiaolong/data/DiLiGenT-MV/mvpmsData/{obj}PNG/Calib_Results.mat"     
        self.mask_path = f"/playpen-nas-ssd/xiaolong/data/DiLiGenT-MV/mvpmsData/{obj}PNG/mask_depth"                             
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.depth_min = 1450
        self.depth_interval = 1
        self.interval_scale = interval_scale

        assert self.mode == "test"
        self.metas = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        self.lighting = lighting

    def build_list(self):
        with open("/playpen-nas-ssd/xiaolong/data/DiLiGenT-MV/mvpmsData/cowPNG/view_20/filenames.txt") as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]
        return scans

    def __len__(self):
        return len(self.metas)

    def retrieving_view(self, num):
        index_list = [num, (num+1)//20 + (num+1)%20, (num+2)//20 + (num+2)%20, (num+3)//20 + (num+3)%20, (num+4)//20 + (num+4)%20]
        view_list = []
        for i in range(5):
            index = index_list[i]
            if index  < 10:
                view = f"/view_0{index}/"
            else:
                view = f"/view_{index}/"
            view_list.append(view)

        return view_list, index_list


    def read_cam_file(self, filename, i):
        calib_mat = scipy.io.loadmat(filename)
        # extrinsics: line [1,5), 4x4 matrix
        rc = np.asarray(calib_mat[f"Rc_{i}"])
        tc = np.asarray(calib_mat[f"Tc_{i}"])
        extrinsics = np.concatenate((rc, tc), -1)
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.asarray(calib_mat["KK"])
        return intrinsics, extrinsics

    def read_img(self, filename):
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) - np.min(img)) / (np.max(img) - np.min(img)) 
        return img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        lighting_path = self.build_list()

        imgs = []
        mask = []
        depth = None
        depth_values = None
        proj_matrices = []
        view_list, index_list = self.retrieving_view(meta)
        for i in range(len(index_list)):
            img_filename = self.imgpath[self.obj] + view_list[i] + lighting_path[self.lighting]
            proj_mat_filename = os.path.join(self.calib_path)
            imgs.append(self.read_img(img_filename)[:, :-4])
        
            intrinsics, extrinsics = self.read_cam_file(proj_mat_filename, index_list[i])
            # multiply intrinsics and extrinsics to get projection matrix
            intrinsics[0][2] -= 2
            intrinsics[:2, :] = intrinsics[:2, :]/4
            proj_mat = extrinsics.copy()
            proj_mat = np.matmul(intrinsics, proj_mat)
            proj_mat = np.concatenate((proj_mat, np.array([[0, 0, 0, 1]])), 0).astype(np.float32)
            proj_matrices.append(proj_mat)

            if i == 1:  # reference view
                t_vals = t_vals = np.linspace(0., 1., num=self.ndepths, dtype=np.float32)
                depth_values = 1450 * (1. - t_vals) + 1600 * t_vals  # (D,)

        mask_filename = self.mask_path + view_list[0][:-1] + ".png"
        mask.append(read_mask(mask_filename))
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)
        return {"imgs": imgs,
                "proj_matrices": proj_matrices,
                "depth_values": depth_values,
                "mask" : mask}


if __name__ == "__main__":
    # some testing code, just IGNORE it
    dataset = DiligentDataset()
    item = dataset[1]
    print(item)
