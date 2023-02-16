from torch.utils.data import Dataset
import numpy as np
import os
import random
import scipy.io as sio
import cv2
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt



class MVSDataset(Dataset):
    def __init__(self, root_dir, split, nviews, ndepths=192, load_normal=False, load_intrinsics=False):
        super(MVSDataset, self).__init__()
        self.root_dir = root_dir
        self.split = split
        self.nviews = nviews
        self.ndepth = ndepths
        self.load_normal = load_normal
        self.load_intrinsics = load_intrinsics

        self.img_wh = (512, 512)  # 'img_wh must both be multiples of 32!'

        assert split in ['train', 'val', 'test']

        self.scenes = ['bearPNG', 'buddhaPNG', 'cowPNG', 'pot2PNG', 'readingPNG']
        self.lights = np.arange(1, 97)  # array

        # depth range in mm
        self.near_far = {
            'bearPNG': (1450, 1600),
            'buddhaPNG': (1450, 1600),
            'cowPNG': (1450, 1600),
            'pot2PNG': (1450, 1600),
            'readingPNG': (1450, 1600),
        }

        self.build_metas()
        self.read_params_file()

    def build_metas(self):
        self.metas = []
        for scene in self.scenes:
            with open(os.path.join(self.root_dir,'diligent_mv_pairs.txt')) as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1:]]
                    if self.split == "train":
                        light = random.randint(1, 96)
                    else:
                        light = 3
                    self.metas += [(scene, light, ref_view, src_views)]

        print("dataset", self.split, "metas:", len(self.metas))

    def __len__(self):
        return len(self.metas)

    def read_params_file(self):
        from scipy.io import loadmat
        self.params = {}
        for scene in self.scenes:
            self.params[scene] = loadmat(os.path.join(self.root_dir, scene, 'Calib_Results.mat'))


    def read_img(self, filename):
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # scale 0~65535 to 0~1
        # img = img.astype(np.float32) / 65535.
        # img = linear_to_srgb(img)
        img = (img.astype(np.float32) - np.min(img)) / (np.max(img) - np.min(img))  # 0~1
        # img = reinhart(img.astype(np.float32))
        return img

    def __getitem__(self, item):
        meta = self.metas[item]
        scene, light, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_mats = []
        near = None
        far = None
        normal = None
        ref_intrinsics = None
        ref_proj_inv = None

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.root_dir, scene, f'view_{vid+1:02d}', f'{light:03d}.png')


            img = self.read_img(img_filename)
            # crop images from (512, 612) to (512, 512)
            img = img[:, 50:562, :]
            imgs.append(img)

            R = self.params[scene]['Rc_%d' % (vid+1)].astype(np.float32).copy()  # w2c
            t = self.params[scene]['Tc_%d' % (vid+1)].astype(np.float32).copy()  # w2c
            extrinsics = np.concatenate((R, t), axis=-1)
            intrinsics = self.params[scene]['KK'].astype(np.float32).copy()
            # modify intrinsics because of image cropping
            intrinsics[0][2] -= 50  # cx
            # downsize x4
            intrinsics[:2, :] = intrinsics[:2, :] / 4

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat = np.matmul(intrinsics, proj_mat)
            proj_mat = np.concatenate((proj_mat, np.array([[0, 0, 0, 1]])), 0).astype(np.float32)
            if i == 0:
                ref_proj_inv = np.linalg.inv(proj_mat)
                proj_mats.append(np.eye(4).astype(np.float32))
            else:
                proj_mats.append(proj_mat @ ref_proj_inv)

            if i == 0:  # reference view


                near_far = self.near_far[scene]  # tuple (2,)

                t_vals = np.linspace(0., 1., num=self.ndepth, dtype=np.float32)  # (D)
                near, far = near_far

                depth_values = near * (1. - t_vals) + far * t_vals  # (D,)

                if self.load_normal:
                    normal_filename = os.path.join(self.root_dir, scene, f'view_{vid + 1:02d}', 'Normal_gt.mat')
                    normal = sio.loadmat(normal_filename)['Normal_gt'].astype(np.float32)
                    norm = np.sqrt((normal * normal).sum(2, keepdims=True))
                    normal = normal / (norm + 1e-10)
                    normal = normal[:, 50:562, :]
                    normal = normal * np.array([1, -1, -1]).reshape((1, 1, 3))
                    normal = cv2.resize(normal, None, fx=1.0 / 2, fy=1.0 / 2)
                    normal = cv2.resize(normal, None, fx=1.0 / 2, fy=1.0 / 2)
                if self.load_intrinsics:
                    ref_intrinsics = intrinsics

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_mats = np.stack(proj_mats)

        sample = {}
        sample['imgs'] = imgs  # (nviews, 3, H, W)
        sample['proj_matrices'] = proj_mats  # (nviews, 4, 4)
        sample['depth'] = depth  # (h//4, w//4)
        sample['depth_values'] = depth_values  # (ndepth, )
        sample['mask'] = mask  # (h//4, w//4)
        sample['near'] = near
        sample['far'] = far
        sample['filename'] = scene + '/view{:0>2}'.format(view_ids[0])
        if self.load_normal:
            sample['normal'] = normal
        if self.load_intrinsics:
            sample['intrinsics'] = ref_intrinsics
        return sample


if __name__ == '__main__':
    dataset = MVSDataset(
        root_dir="/playpen-nas-ssd/xiaolong/project/MVSNet_pytorch/training_data/DiLiGenT-MV/mvpmsData",
        split='train',
        nviews=5,
    )

    print(f"Dataset size: {len(dataset)}")
    item = dataset[0]








