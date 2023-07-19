import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp

from lietorch import SE3
from .base import RGBDDataset
from .stream import RGBDStream





class GeneralRGBDataset(RGBDDataset):
    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, name, test_scene_list: list, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2
        self.test_scene_list = test_scene_list
        super(GeneralRGBDataset, self).__init__(name=name, **kwargs)


    def is_test_scene(self, scene):
        # print(scene, any(x in scene for x in test_split))
        return scene in self.test_scene_list

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building General dataset")

        scene_info = {}
        scenes = glob.glob(osp.join(self.root, '*/*/*/*'))
        for scene in tqdm(sorted(scenes)):
            images = sorted(glob.glob(osp.join(scene, 'image/*.png')))
            depths = sorted(glob.glob(osp.join(scene, 'depth/*.npy')))

            poses = np.loadtxt(osp.join(scene, 'pose.txt'), delimiter=' ')
            poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
            poses[:, :3] /= TartanAir.DEPTH_SCALE
            intrinsics = np.loadtxt(osp.join(scene, 'intrinsics.txt')) * len(images)

            # graph of co-visible frames based on flow
            graph = self.build_frame_graph(poses, depths, intrinsics)

            scene = '/'.join(scene.split('/'))
            scene_info[scene] = {'images': images, 'depths': depths,
                                 'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

        return scene_info