import torch
import numpy as np


def create_viewport_matrix(nx, ny, batch_size=1, device='cuda'):
    """Creates a viewport matrix that transforms vertices in NDC [-1, 1]
    space to viewport (screen) space. Based on a blogpost by Mauricio Poppe:
    https://www.mauriciopoppe.com/notes/computer-graphics/viewing/viewport-transform/
    except that I added the minus sign at [1, 1], which makes sure that the
    viewport (screen) space origin is in the top left.

    Parameters
    ----------
    nx : int
        Number of pixels in the x dimension (width)
    ny : int
        Number of pixels in the y dimension (height)

    Returns
    -------
    mat : np.ndarray
        A 4x4 numpy array representing the viewport transform
    """

    mat = torch.tensor(#np.array(
        [
            [nx / 2, 0, 0, (nx - 1) / 2],
            [0, -ny / 2, 0, (ny - 1) / 2],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device
    ).repeat(batch_size, 1, 1)

    return mat


def create_ortho_matrix(nx, ny, znear=0.05, zfar=100.0, batch_size=1, device='cuda'):
    """Creates an orthographic projection matrix, as
    used by EMOCA/DECA. Based on the pyrender implementaiton.
    Assumes an xmag and ymag of 1.

    Parameters
    ----------
    nx : int
        Number of pixels in the x-dimension (width)
    ny : int
        Number of pixels in the y-dimension (height)
    znear : float
        Near clipping plane distance (from eye/camera)
    zfar : float
        Far clipping plane distance (from eye/camera)

    Returns
    -------
    mat : np.ndarray
        A 4x4 affine matrix
    """
    n = znear
    f = zfar

    mat = torch.tensor(#np.array(
        [
            [1 / (nx / ny), 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 2.0 / (n - f), (f + n) / (n - f)],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device
    )
    return mat


def crop_matrix_to_3d(mat_33, device='cuda'):
    """Transforms a 3x3 matrix used for cropping (on 2D coordinates)
    into a 4x4 matrix that can be used to transform 3D vertices.
    It assumes that there is no rotation element.

    Parameters
    ----------
    mat_33 : np.ndarray
        A 3x3 affine matrix

    Returns
    -------
    mat_44 : np.ndarray
        A 4x4 affine matrix
    """
    # Define translation in x, y, & z (z = 0)
    #t_xyz = np.r_[mat_33[:2, 2], 0]
    t_xyz = torch.cat([mat_33[:2, 2], torch.tensor([0], device=device)])
    
    # Add column representing z at the diagonal
    #mat_44 = np.c_[mat_33[:, :2], [0, 0, 1]]
    mat_33 = torch.hstack([mat_33[:, :2],
                           torch.tensor([[0, 0, 1]], device=device).T])

    # Add back translation
    #mat_44 = np.c_[mat_44, t_xyz]
    mat_34 = torch.hstack([mat_33, t_xyz.unsqueeze(1)])

    # Make it a proper 4x4 matrix
    #mat_44 = np.r_[mat_44, [[0, 0, 0, 1]]]
    mat_44 = torch.vstack([mat_34, torch.tensor([0, 0, 0, 1], device=device)])

    return mat_44
