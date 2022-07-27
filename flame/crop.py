""" Module with functionality to crop images based on landmarks estimated by the
``face_alignment`` package by `Adrian Bulat <https://www.adrianbulat.com/face-alignment>`_ [1]_.
or the ``insightface`` package [2]_.

.. [1] Bulat, A., & Tzimiropoulos, G. (2017). How far are we from solving the 2d & 3d
       face alignment problem?(and a dataset of 230,000 3d facial landmarks).
       In *Proceedings of the IEEE International Conference on Computer Vision*
       (pp. 1021-1030).

.. [2] Deng, J., Guo, J., Ververas, E., Kotsia, I., & Zafeiriou, S. (2020). Retinaface:
       Single-shot multi-level face localisation in the wild. In Proceedings of the
       IEEE/CVF conference on computer vision and pattern recognition (pp. 5203-5212).
"""

import cv2
import math
import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.transform import estimate_transform, warp


class BaseModel:
    
    @staticmethod
    def to_numpy(img):
        """ 'Undoes' the preprocessing of the cropped image and returns an ordinary
        h x w x 3 numpy array. Useful for checking the cropping result. 
        
        Parameters
        ----------
        img : torch.Tensor
            The result from the cropping operation (i.e., whatever the ``__call__``
            method returns); should be a 1 x 3 x 224 x 224 tensor
        
        Returns
        -------
        img : np.ndarray
            A 224 x 224 x 3 numpy array with uint8 values
        """        
        
        img = img.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        img = (img * 255).astype(np.uint8)
        return img
    
    @staticmethod
    def to_torch(img, scale=255., device='cuda', add_batch_dim=True):
        import torch
        img = img.transpose(2, 0, 1) / scale
        img = torch.tensor(img).float().to(device)
        if add_batch_dim:
            img = img.unsqueeze(0)

        return img


class FanCropModel(BaseModel):
    """ Cropping model based on the estimated (2D) landmarks from  the ``face_alignment``
    package.
    
    Parameters
    ----------
    device : str
        Either 'cuda' (GPU) or 'cpu'
    target_size : tuple
        Length 2 tuple with desired width/heigth of cropped image; should be (224, 224)
        for EMOCA and DECA
    
    Attributes
    ----------
    model : FaceAlignment
        The initialized face alignment model from ``face_alignment``, using 2D landmarks    
    """

    def __init__(self, device='cuda', target_size=(224, 224)):
        from face_alignment import LandmarksType, FaceAlignment
        self.device = device
        self.target_size = target_size
        self.model = FaceAlignment(LandmarksType._2D, device=device)

    def _load_image(self, image):
        """Loads image using PIL if it's not already
        a numpy array."""
        if isinstance(image, (str, Path)):
            image = np.array(imread(image))

        return image

    def _create_bbox(self, lm, scale=1.25):
        """ Creates a bounding box (bbox) based on the landmarks by creating
        a box around the outermost landmarks (+10%), as done in the original
        DECA usage.

        Parameters
        ----------
        scale : float
            Factor to scale the bounding box with
        """
        left = np.min(lm[:, 0])
        right = np.max(lm[:, 0])
        top = np.min(lm[:, 1])
        bottom = np.max(lm[:, 1])

        orig_size = (right - left + bottom - top) / 2 * 1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(orig_size * scale)
        return np.array(
            [
                [center[0] - size / 2, center[1] - size / 2],  # bottom left
                [center[0] - size / 2, center[1] + size / 2],  # top left
                [center[0] + size / 2, center[1] - size / 2],  # bottom right
                [center[0] + size / 2, center[1] + size / 2],  # top right
            ]
        )  

    def _crop(self, img_orig, bbox):
        """ Using the bounding box (`self.bbox`), crops the image by warping the image
        based on a similarity transform of the bounding box to the corners of target size
        image. """
        w, h = self.target_size
        dst = np.array([[0, 0], [0, w - 1], [h - 1, 0]])
        self.tform = estimate_transform("similarity", bbox[:3, :], dst)

        # Note to self: preserve_range needs to be True, because otherwise `warp` will scale the data!
        return warp(img_orig, self.tform.inverse, output_shape=(w, h), preserve_range=True)

    def _preprocess(self, img_crop):
        """ Transposes (channels, width, height), rescales (/255) the data,
        casts the data to torch, and add a batch dimension (`unsqueeze`). """

        import torch
        img_crop = img_crop.transpose(2, 0, 1)
        img_crop = img_crop / 255.0
        img_crop = torch.tensor(img_crop, dtype=torch.float32).to(self.device)
        return img_crop.unsqueeze(0)  # add singleton batch dim

    def __call__(self, image):
        """ Runs all steps of the cropping / preprocessing pipeline
        necessary for use with Flame-based models such as DECA/EMOCA. 
        
        Parameters
        -----------
        image : str, Path, np.ndarray
            Either a string or ``pathlib.Path`` object to an image or a numpy array
            (width x height x 3) representing the already loaded RGB image

        Returns
        -------
        torch.Tensor
            The preprocessed (normalized) and cropped image as a ``torch.Tensor``
            of shape (1, 3, 224, 224), as EMOCA expects (the 1 is the batch size)
        
        Examples
        --------
        To preprocess (which includes cropping) an image:
        
        >>> from flame.data import get_example_img
        >>> crop_model = CropModel(device='cpu')
        >>> img = get_example_img()  # path to jpg image
        >>> cropped_img = crop_model(img)
        >>> cropped_img.shape
        torch.Size([1, 3, 224, 224])
        """

        # Load image if not already a h x w x 3 numpy array
        img_orig = self._load_image(image)

        # Estimate landmarks
        lm = self.model.get_landmarks_from_image(img_orig.copy())
        if lm is None:
            raise ValueError("No face detected!")
        elif len(lm) > 1:
            raise ValueError(f"More than one face (i.e., {len(lm)}) detected!")
        else:
            lm = lm[0]
        
        # Create bounding box based on landmarks, use that to crop image, and return
        # preprocessed (normalized, to tensor) image        
        bbox = self._create_bbox(lm)
        img_crop = self._crop(img_orig, bbox)
        return self._preprocess(img_crop)

    def viz_qc(self, f_out=None, return_rgba=False):
        """ Visualizes the inferred 3D landmarks & bounding box, as well as the final
        cropped image.

        Parameters
        ----------
        f_out : str, Path
            Path to save viz to; if ``None``, returned as an RGBA image
        return_rgba : bool
            Whether to return a numpy image with the raw pixel RGBA intensities
            (True) or not (False; return nothing)
            
        Returns
        -------
        img : np.ndarray
            The rendered image as a numpy array (if ``f_out`` is ``None``)
        
        Examples
        --------
        To visualize the landmark and (EMOCA-style) bounding box:
        
        >>> from medusa.data import get_example_frame
        >>> img = get_example_frame()
        >>> fan = FAN(lm_type='2D')
        >>> cropped_img = fan.prepare_for_emoca(img) 
        >>> viz_img = fan.viz_qc(return_rgba=True)
        >>> viz_img.shape
        (480, 640, 4)
        """
        import io
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        if f_out is None and return_rgba is False:
            raise ValueError("Either supply f_out or set return_rgb to True!")

        fig, axes = plt.subplots(nrows=2, constrained_layout=True)
        axes[0].imshow(self.img_orig)
        axes[0].axis("off")
        axes[0].plot(self.lm[:, 0], self.lm[:, 1], marker="o", ms=2, ls="")

        w = self.bbox[2, 0] - self.bbox[0, 0]
        h = self.bbox[3, 1] - self.bbox[2, 1]
        rect = Rectangle(self.bbox[0, :], w, h, facecolor="none", edgecolor="r")
        axes[0].add_patch(rect)

        axes[1].imshow(self.img_crop.astype(np.uint8))
        axes[1].axis("off")

        if f_out is not None:
            fig.savefig(f_out)
            plt.close()
        else:
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format="raw", dpi=100)
            io_buf.seek(0)
            img = np.reshape(
                np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
            )
            io_buf.close()
            plt.close()
            return img


class InsightFaceCropModel(BaseModel):
    """ Cropping model based on functionality from the ``insightface`` package, as used
    by MICA (https://github.com/Zielon/MICA).
    
    Parameters
    ----------
    device : str
        Either 'cuda' (GPU) or 'cpu'
    target_size : tuple
        Length 2 tuple with desired width/heigth of cropped image; should be (112, 112)
        for MICA
    """    
    
    def __init__(self, device='cuda', target_size=(112, 112)):
        """ Initialize InsightFaceCropModel. """
        from insightface.app import FaceAnalysis
        self.device = device
        self.target_size = target_size
        self.app = FaceAnalysis(name='antelopev2', providers=[f'{device.upper()}ExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(224, 224))  # must be 224x224 (not 112x112)

    def __call__(self, image):
        import torch
        from insightface.app.common import Face
        from insightface.utils import face_align
        
        img = cv2.imread(str(image))
        bboxes, kpss = self.app.det_model.detect(img, max_num=0, metric='default')
        i = self._get_center(bboxes, img)
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        kps = None
        if kpss is not None:
            kps = kpss[i]
        
        # Crop to target size using keypoints (kps)
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        af_img = face_align.norm_crop(img, landmark=face.kps, image_size=self.target_size[0])
        
        # Channel-wise mean subtraction (- 127.5), scaling (* 1 / 127.5), BGR -> RGB
        af_img = ((af_img - 127.5) * (1 / 127.5)).transpose(2, 0, 1)
        
        # Add singleton batch dim (shape: 1 x 3 x 112 x 112), cast to device
        af_img = torch.tensor(af_img[None, ::-1, ...].copy()).to(self.device)
        
        #deca_img = face_align.norm_crop(img, landmark=face.kps, image_size=224)
        
        return af_img

    def _dist(self, p1, p2):
        return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    
    def _get_center(self, bboxes, img):
        
        img_center = img.shape[0] // 2, img.shape[1] // 2
        size = bboxes.shape[0]
        distance = np.Inf
        j = 0
        for i in range(size):
            x1, y1, x2, y2 = bboxes[i, 0:4]
            dx = abs(x2 - x1) / 2.0
            dy = abs(y2 - y1) / 2.0
            current = self._dist((x1 + dx, y1 + dy), img_center)
            if current < distance:
                distance = current
                j = i

        return j
