import cv2
import numpy as np
import torch
import logging


from utils.utils import DepthToSpace, flattenDetection
from utils.loader import modelLoader
from models.model_wrap import SuperPointFrontend_torch


@torch.no_grad()
class Val_model_heatmap(SuperPointFrontend_torch):
    def __init__(self, config, device='cpu', verbose=False):
        self.config = config
        self.model = self.config['name']
        self.params = self.config['params']
        self.weights_path = self.config['pretrained']
        self.device=device
        self.nms_dist = self.config['nms']
        self.conf_thresh = self.config['detection_threshold']
        self.nn_thresh = self.config['nn_thresh']
        self.top_k = self.config.get('top_k')
        self.cell = 8
        self.cell_size = 8
        self.border_remove = 4
        self.sparsemap = None
        self.heatmap = None
        self.pts = None
        self.pts_subpixel = None
        self.pts_nms_batch = None
        self.desc_sparse_batch = None
        self.patches = None


    def loadModel(self):
        self.net = modelLoader(model=self.model, **self.params)
        checkpoint = torch.load(self.weights_path,
                                map_location=lambda storage, loc: storage)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net = self.net.to(self.device)
        logging.info('successfully load pretrained model from: %s', self.weights_path)
        pass

    def flatten_64to1(semi, cell_size=8):
        """
        input: 
            semi: tensor[batch, cell_size*cell_size, Hc, Wc]
            (Hc = H/8)
        outpus:
            heatmap: tensor[batch, 1, H, W]
        """
        depth2space = DepthToSpace(cell_size)
        heatmap = depth2space(semi)
        return heatmap


    def run(self, images):
        """
        input: 
            images: tensor[batch(1), 1, H, W]

        """
        with torch.no_grad():
            outs = self.net(images)
        semi = outs['semi']
        self.outs = outs

        channel = semi.shape[1]
        if channel == 64:
            heatmap = self.flatten_64to1(semi, cell_size=self.cell_size)
        elif channel == 65:
            heatmap = flattenDetection(semi, tensor=True)
            
        heatmap_np = heatmap.detach().cpu().numpy()
        self.heatmap = heatmap_np
        return self.heatmap


    def heatmap_to_pts(self):
        heatmap_np = self.heatmap

        pts_nms_batch = [self.getPtsFromHeatmap(h) for h in heatmap_np]
        self.pts_nms_batch = pts_nms_batch
        return pts_nms_batch


    def get_keypoints_and_descriptors(self, sample):
        img = sample['image']
        heatmap_batch = self.run(img.to(self.device))
        pts = self.heatmap_to_pts()
        desc_sparse = self.desc_to_sparseDesc()
        if self.top_k:
            if pts[0].shape[1] > self.top_k:
                pts[0] = pts[0][:, :self.top_k]
                desc_sparse[0] = desc_sparse[0][:, :self.top_k]
        pts = pts[0]
        desc_sparse = desc_sparse[0]
        return pts, desc_sparse


    def desc_to_sparseDesc(self):
        desc_sparse_batch = [self.sample_desc_from_points(self.outs['desc'], pts) for pts in self.pts_nms_batch]
        self.desc_sparse_batch = desc_sparse_batch
        return desc_sparse_batch
    
    def getInliers_cv(self, matches, H=None, epi=3, verbose=False):
        H, inliers = cv2.findHomography(matches[:, [0, 1]],
                                        matches[:, [2, 3]],
                                        cv2.RANSAC)
        inliers = inliers.flatten()

        return inliers
