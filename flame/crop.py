""" Module with functionality to crop images based on landmarks estimated by the
``face_alignment`` package by `Adrian Bulat <https://www.adrianbulat.com/face-alignment>`_ [1]_.

.. [1] Bulat, A., & Tzimiropoulos, G. (2017). How far are we from solving the 2d & 3d
       face alignment problem?(and a dataset of 230,000 3d facial landmarks).
       In *Proceedings of the IEEE International Conference on Computer Vision*
       (pp. 1021-1030).
"""

import torch
import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.transform import estimate_transform, warp
from face_alignment import LandmarksType, FaceAlignment


class CropModel:
    """ Cropping model based on the estimated (2D) landmarks from ``face_alignment``.
    
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
        self.device = device
        self.target_size = target_size
        self.model = FaceAlignment(
            LandmarksType._2D,
            flip_input=False,
            device=device,
            face_detector='sfd',
        )

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
        img_crop = img_crop.transpose(2, 0, 1)
        img_crop = img_crop / 255.0
        img_crop = torch.tensor(img_crop, dtype=torch.float32).to(self.device)
        return img_crop.unsqueeze(0)  # add singleton batch dim

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
