from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from typing_extensions import Self


def rot_to_quat(rotation: Rotation) -> Quaternion:
    """Convert 'scipy.spatial.transform.Rotation' object to 'pyquaternion.Quaternion'.

    Args:
        rotation (Rotation): 'scipy.spatial.transform.Rotation' object

    Returns:
        Quaternion:  'pyquaternion.Quaternion' object

    >>> from navio3ddata.utils import rot_to_quat
    >>> from scipy.spatial.transform import Rotation
    >>> import numpy as np
    >>> rot = Rotation.from_matrix(np.eye(3))
    >>> quat = rot_to_quat(rot)
    >>> quat
    Quaternion(1.0, 0.0, 0.0, 0.0)
    >>> quat.rotation_matrix
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    """
    # https://docs.scipy.org/doc/scipy-1.8.1/reference/generated/scipy.spatial.transform.Rotation.as_quat.html
    # scalar last order here for SDA version
    xyzw_arr = rotation.as_quat()
    return Quaternion(scalar=xyzw_arr[3], vector=xyzw_arr[:3])


class RigidTransform:  # noqa: PLR0904
    """
    Represents a 3D rigid body transformation, composed of a rotation and a translation.

    Internally, the transformation is stored as a 4x4 homogeneous matrix, which allows
    for easy composition and application of transformations to points in 3D space.
    """

    _DEFAULT_DTYPE = np.float64

    def __init__(
        self,
        matrix: npt.NDArray[Any],
        normalize: bool = True,
    ) -> None:
        """
        Initialize a RigidTransform instance from a 4x4 transformation matrix.

        Args:
            matrix (np.ndarray): A (4, 4) homogeneous transformation matrix.
            normalize (bool): Whether to orthonormalize the rotation matrix on initialization.

        Examples:
            >>> import numpy as np
            >>> from navio3ddata.utils import RigidTransform
            >>> mat = np.eye(4)
            >>> rt = RigidTransform(mat)
            >>> np.allclose(rt._matrix, mat)
            True

            >>> rot = np.array([[1.0, 0.01, 0.0],
            ...                 [0.0, 1.0, 0.01],
            ...                 [0.0, 0.0, 1.0]])
            >>> t = np.array([0.0, 0.0, 0.0])
            >>> mat_perturbed = np.eye(4)
            >>> mat_perturbed[:3, :3] = rot
            >>> mat_perturbed[:3, 3] = t
            >>> rt = RigidTransform(mat_perturbed, normalize=True)
            >>> r = rt._matrix[:3, :3]
            >>> is_orthonormal = np.allclose(r.T @ r, np.eye(3), atol=1e-6)
            >>> is_orthonormal
            True

            Normalization should not change already orthonormal matrix:
            >>> from scipy.spatial.transform import Rotation as R
            >>> rot_mat = R.from_euler("xyz", [30, 45, 60], degrees=True).as_matrix()
            >>> mat_ortho = np.eye(4)
            >>> mat_ortho[:3, :3] = rot_mat
            >>> mat_ortho[:3, 3] = [1.0, 2.0, 3.0]
            >>> rt_nochange = RigidTransform(mat_ortho.copy(), normalize=True)
            >>> np.allclose(rt_nochange._matrix, mat_ortho, atol=1e-12)
            True
        """
        if matrix.shape != (4, 4):
            msg = "Input matrix must be 4x4."
            raise ValueError(msg)

        if normalize:
            matrix = self.orthonormalize(matrix)
        self._matrix = matrix

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._matrix!r}, dtype={self._matrix.dtype})"

    @staticmethod
    def orthonormalize(matrix: np.ndarray) -> np.ndarray:
        """
        Orthonormalize the rotation part of the transformation matrix using SVD.

        Args:
            matrix (np.ndarray): A 4x4 transformation matrix.

        Returns:
            np.ndarray: The transformation matrix with orthonormalized rotation.
        """
        rot = matrix[:3, :3]
        u, _, vh = np.linalg.svd(rot)
        rot_ortho = u @ vh

        ortho_matrix = matrix.copy()
        ortho_matrix[:3, :3] = rot_ortho
        return ortho_matrix

    @classmethod
    def identity(cls, dtype: npt.DTypeLike = _DEFAULT_DTYPE) -> Self:
        """
        Create an identity transform.

        Args:
            dtype (np.dtype): The dtype for the matrix.

        Returns:
            RigidTransform: Identity transform (no translation or rotation).
        """
        return cls(np.eye(4, dtype=dtype))

    def is_noop(self) -> bool:
        """Indicator if transform does not change data

        Returns:
            bool: True if transform is identity False otherwise
        """
        return np.all(self._matrix == self.identity(dtype=self._matrix.dtype)).item()

    @classmethod
    def from_translation_rotation(
        cls,
        *,
        translation: npt.ArrayLike | None,
        rotation: npt.ArrayLike | Quaternion | Rotation | None,
        dtype: npt.DTypeLike = _DEFAULT_DTYPE,
        scalar_first: bool = True,
    ) -> Self:
        """
        Create a transform from translation and rotation.

        Args:
            translation (array-like or None): 3D translation vector.
            rotation (Quaternion, Rotation, or array-like): Rotation to apply.
            dtype (np.dtype): Matrix dtype.
            scalar_first (bool): Controls order of quaternion components, if True
                it corresponds to wxyz, xyzw otherwise. Defaults to True. See for more details:
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html#scipy.spatial.transform.Rotation.from_quat

        Returns:
            RigidTransform: The resulting transformation.
        """
        translation = cls.identity_translation(dtype) if translation is None else np.array(translation, dtype=dtype)
        assert translation.shape == (3,)
        if rotation is None:
            rotation = cls.identity_rotation()
        if isinstance(rotation, Quaternion):
            rot_matrix = rotation.rotation_matrix
        elif isinstance(rotation, Rotation):
            rot_matrix = rotation.as_matrix()
        else:
            if scalar_first:
                rotation = np.roll(rotation, -1)
            # https://docs.scipy.org/doc/scipy-1.8.1/reference/generated/scipy.spatial.transform.Rotation.as_quat.html
            # scalar last order here for SDA version
            rot_matrix = Rotation.from_quat(rotation).as_matrix()
        transform = np.eye(4, dtype=dtype)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = translation
        return cls(transform)

    @classmethod
    def from_matrix(cls, matrix: npt.NDArray[Any], dtype: npt.DTypeLike = _DEFAULT_DTYPE) -> Self:
        """
        Create a transform directly from a matrix.

        Args:
            matrix (np.ndarray): 4x4 transformation matrix.
            dtype (np.dtype): Matrix dtype.

        Returns:
            RigidTransform: The resulting transformation.
        """
        return cls(matrix=matrix.astype(dtype))

    @classmethod
    def from_dict(
        cls,
        translation_dict: dict[str, int | float | str],
        rotation_dict: dict[str, int | float | str],
        dtype: npt.DTypeLike = _DEFAULT_DTYPE,
    ) -> Self:
        """
        Create a transform from dicts of translation and rotation values.

        Args:
            translation_dict (dict): Keys must be 'x', 'y', 'z'.
            rotation_dict (dict): Keys must be 'w', 'x', 'y', 'z'.
            dtype (np.dtype): Matrix dtype.

        Returns:
            RigidTransform: The resulting transformation.
        """
        return cls.from_translation_rotation(
            translation=[translation_dict[key] for key in "xyz"],
            rotation=[rotation_dict[key] for key in "wxyz"],
            dtype=dtype,
            scalar_first=True,
        )

    @staticmethod
    def identity_rotation() -> Rotation:
        """
        Return an identity rotation.

        Returns:
            Rotation: Identity rotation object.
        """
        return Rotation.identity()

    @staticmethod
    def identity_translation(dtype: npt.DTypeLike) -> npt.NDArray[Any]:
        """
        Return a zero translation vector.

        Args:
            dtype (np.dtype): Array dtype.

        Returns:
            np.ndarray: Translation vector of shape (3,) with zeros.
        """
        return np.zeros((3,), dtype=dtype)

    @property
    def matrix(self) -> npt.NDArray[Any]:
        """
        Access the internal transformation matrix.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        return self._matrix

    @property
    def translation(self) -> npt.NDArray[Any]:
        """
        Access the translation component of the transform.

        Returns:
            np.ndarray: Translation vector of shape (3,).
        """
        return self._matrix[:3, 3]

    @property
    def rotation(self) -> Rotation:
        return Rotation.from_matrix(self._matrix[:3, :3])

    @classmethod
    def from_rotation(cls, rotation: Rotation, dtype: npt.DTypeLike = _DEFAULT_DTYPE) -> RigidTransform:
        """Initialize from a rotation, without a translation.

        When applying this transform to a vector ``v``, the result is the
        same as if the rotation was applied to the vector.
        ``Tf.from_rotation(r).apply(v) == r.apply(v)``

        Parameters
        ----------
        rotation : `Rotation` instance
            A single rotation or a stack of rotations.

        Returns
        -------
        transform : `RigidTransform` instance

        Examples
        --------
        >>> from navio3ddata.utils import RigidTransform as Tf
        >>> from scipy.spatial.transform import Rotation as R
        >>> import numpy as np

        Creating a transform from a single rotation:

        >>> r = R.from_euler("ZYX", [90, 30, 0], degrees=True)
        >>> np.allclose(r.apply([1, 0, 0]), np.array([0, 0.8660254, -0.5]))
        True
        >>> tf = Tf.from_rotation(r)
        >>> np.allclose(tf.apply([1, 0, 0]), np.array([0, 0.8660254, -0.5]))
        True

        The upper 3x3 submatrix of the transformation matrix is the rotation
        matrix:

        >>> np.allclose(tf.as_matrix()[:3, :3], r.as_matrix(), atol=1e-12)
        True
        """
        return cls.from_translation_rotation(translation=None, rotation=rotation, dtype=dtype)

    @classmethod
    def from_translation(cls, translation: npt.ArrayLike, dtype: npt.DTypeLike = _DEFAULT_DTYPE) -> RigidTransform:
        """Initialize from a translation numpy array, without a rotation.

        When applying this transform to a vector ``v``, the result is the same
        as if the translation and vector were added together. If ``t`` is the
        displacement vector of the translation, then:

        ``Tf.from_translation(t).apply(v) == t + v``

        Parameters
        ----------
        translation : array_like, shape (N, 3) or (3,)
            A single translation vector or a stack of translation vectors.

        Returns
        -------
        transform : `RigidTransform` instance

        Examples
        --------
        >>> from navio3ddata.utils import RigidTransform as Tf
        >>> import numpy as np

        Creating a transform from a single translation vector:

        >>> t = np.array([2, 3, 4])
        >>> t + np.array([1, 0, 0])
        array([3, 3, 4])
        >>> tf = Tf.from_translation(t)
        >>> tf.apply([1, 0, 0])
        array([3., 3., 4.])

        The top 3x1 points in the rightmost column of the transformation matrix
        is the translation vector:

        >>> tf.as_matrix()
        array([[1., 0., 0., 2.],
               [0., 1., 0., 3.],
               [0., 0., 1., 4.],
               [0., 0., 0., 1.]])
        >>> np.allclose(tf.as_matrix()[:3, 3], t)
        True
        """
        return cls.from_translation_rotation(translation=translation, rotation=None, dtype=dtype)

    def compose(self, following_transform: RigidTransform) -> RigidTransform:
        # T = T1 @ T2 means apply T2, then T1 (i.e., T(x) = T1(T2(x))).
        return RigidTransform.from_matrix(following_transform.matrix @ self.matrix, dtype=self._matrix.dtype)

    def apply(self, points: npt.ArrayLike, inverse: bool = False) -> npt.NDArray[Any]:
        """Apply the transform to a vector.

        If the original frame transforms to the final frame by this transform,
        then its application to a vector can be seen in two ways:

            - As a projection of vector components expressed in the final frame
              to the original frame.
            - As the physical transformation of a vector being glued to the
              original frame as it transforms. In this case the vector
              components are expressed in the original frame before and after
              the transformation.

        In terms of rotation matrices and translation vectors, this application
        is the same as
        ``self.translation + self.rotation.as_matrix() @ vector``.

        Parameters
        ----------
        vector : array_like, shape (N, 3) or (3,)
            A single vector or a stack of vectors.
        inverse : bool, optional
            If True, the inverse of the transform is applied to the vector.

        Returns
        -------
        transformed_vector : numpy.ndarray, shape (N, 3) or (3,)
            The transformed vector(s). Shape depends on the following cases:

                - If object contains a single transform (as opposed to a
                  stack with a single transform) and a single vector is
                  specified with shape ``(3,)``, then `transformed_vector` has
                  shape ``(3,)``.
                - In all other cases, `transformed_vector` has shape
                  ``(N, 3)``, where ``N`` is either the number of
                  transforms or vectors.

        Examples
        --------
        >>> from navio3ddata.utils import RigidTransform as Tf
        >>> from scipy.spatial.transform import Rotation as R
        >>> import numpy as np

        Apply a single transform to a vector. Here the transform is just a
        translation, so the result is the vector added to the translation
        vector.

        >>> t = np.array([1, 2, 3])
        >>> tf = Tf.from_translation(t)
        >>> t + np.array([1, 0, 0])
        array([2, 2, 3])
        >>> tf.apply([1, 0, 0])
        array([2., 2., 3.])

        Apply a single transform to a stack of vectors:

        >>> tf.apply([[1, 0, 0], [0, 1, 0]])
        array([[2., 2., 3.],
               [1., 3., 3.]])

        Apply the inverse of a transform to a vector, so the result is the
        negative of the translation vector added to the vector.

        >>> np.allclose(-t + np.array([1, 0, 0]), np.array([0, -2, -3]))
        True
        >>> np.allclose(tf.apply([1, 0, 0], inverse=True), np.array([0, -2, -3]))
        True

        For transforms which are not just pure translations, applying it to a
        vector is the same as applying the rotation component to the vector and
        then adding the translation component.

        >>> r = R.from_euler('z', 60, degrees=True)
        >>> tf = Tf.from_components(t, r)
        >>> np.allclose(t + r.apply([1, 0, 0]), np.array([1.5, 2.8660254, 3]))
        True
        >>> np.allclose(tf.apply([1, 0, 0]), np.array([1.5, 2.8660254, 3]))
        True

        When applying the inverse of a transform, the result is the negative of
        the translation vector added to the vector, and then rotated by the
        inverse rotation.

        >>> r.inv().apply(-t + np.array([1, 0, 0]))
        array([-1.73205081, -1.        , -3.        ])
        >>> tf.apply([1, 0, 0], inverse=True)
        array([-1.73205081, -1.        , -3.        ])
        """
        points = np.asarray(points, dtype=self._matrix.dtype)
        n_dim = len(points.shape)
        single = n_dim == 1
        if single:
            points = points[np.newaxis, :]
        elif n_dim > 2:
            msg = "3d tensors are not supported"
            raise NotImplementedError(msg)
        if self.is_noop():
            return points
        matrix = self.inv().matrix if inverse else self.matrix
        transformed = self.apply_transform_with_extras(matrix, points)
        if single:
            transformed = np.squeeze(transformed, axis=0)
        return transformed

    def inv(self) -> RigidTransform:
        """Invert this transform.

        Composition of a transform with its inverse results in an identity
        transform.

        A rigid transform is a composition of a rotation and a translation,
        where the rotation is applied first, followed by the translation. So the
        inverse transform is equivalent to the inverse translation followed by
        the inverse rotation.

        Returns
        -------
        `RigidTransform` instance
            The inverse of this transform.

        Examples
        --------
        >>> from navio3ddata.utils import RigidTransform as Tf
        >>> from scipy.spatial.transform import Rotation as R
        >>> import numpy as np

        A transform composed with its inverse results in an identity transform:

        >>> rng = np.random.default_rng(seed=123)
        >>> t = rng.random(3)
        >>> r = R.random(rng=rng)
        >>> tf = Tf.from_components(t, r)
        >>> tf.as_matrix()
        array([[-0.45431291,  0.67276178, -0.58394466,  0.68235186],
               [-0.23272031,  0.54310598,  0.80676958,  0.05382102],
               [ 0.85990758,  0.50242162, -0.09017473,  0.22035987],
               [ 0.        ,  0.        ,  0.        ,  1.        ]])

        >>> np.allclose((tf.inv() * tf).as_matrix(), np.identity(4))
        True
        >>> tf.inv().matrix.dtype == tf.matrix.dtype
        True

        The inverse rigid transform is the same as the inverse translation
        followed by the inverse rotation:

        >>> t, r = tf.as_components()
        >>> r_inv = r.inv()  # inverse rotation
        >>> t_inv = -t  # inverse translation
        >>> tf_r_inv = Tf.from_rotation(r_inv)
        >>> tf_t_inv = Tf.from_translation(t_inv)
        >>> np.allclose((tf_r_inv * tf_t_inv).as_matrix(),
        ...             tf.inv().as_matrix(),
        ...             atol=1e-12)
        True
        >>> np.allclose((tf_r_inv * tf_t_inv * tf).as_matrix(), np.identity(4))
        True
        """
        if self.is_noop():
            return self
        return RigidTransform(self.invert_transform_matrix(self.matrix))

    def as_components(self) -> tuple[npt.NDArray[Any], Rotation]:
        """Return the translation and rotation components of the transform,
        where the rotation is applied first, followed by the translation.

        4x4 rigid transformation matrices are of the form::

            [       tx]
            [   R   ty]
            [       tz]
            [ 0 0 0  1]

        Where ``R`` is a 3x3 orthonormal rotation matrix and
        ``t = [tx, ty, tz]`` is a 3x1 translation vector. This function
        returns the rotation corresponding to this rotation matrix
        ``r = Rotation.from_matrix(R)`` and the translation vector ``t``.

        Take a transform ``tf`` and a vector ``v``. When applying the transform
        to the vector, the result is the same as if the transform was applied
        to the vector in the following way:
        ``tf.apply(v) == translation + rotation.apply(v)``

        Returns
        -------
        translation : numpy.ndarray, shape (N, 3) or (3,)
            The translation of the transform.
        rotation : `Rotation` instance
            The rotation of the transform.

        Examples
        --------
        >>> from navio3ddata.utils import RigidTransform as Tf
        >>> from scipy.spatial.transform import Rotation as R
        >>> import numpy as np

        Recover the rotation and translation from a transform:

        >>> t = np.array([2, 3, 4])
        >>> r = R.from_matrix([[0, 0, 1],
        ...                    [1, 0, 0],
        ...                    [0, 1, 0]])
        >>> tf = Tf.from_components(t, r)
        >>> tf_t, tf_r = tf.as_components()
        >>> tf_t
        array([2., 3., 4.])
        >>> tf_r.as_matrix()
        array([[0., 0., 1.],
               [1., 0., 0.],
               [0., 1., 0.]])

        The transform applied to a vector is equivalent to the rotation applied
        to the vector followed by the translation:

        >>> r.apply([1, 0, 0])
        array([0., 1., 0.])
        >>> t + r.apply([1, 0, 0])
        array([2., 4., 4.])
        >>> tf.apply([1, 0, 0])
        array([2., 4., 4.])
        """
        return self.translation, self.rotation

    @staticmethod
    def apply_transform_with_extras(transform: npt.NDArray[Any], points: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Apply a 4x4 transformation matrix to the first 3 dimensions of N-D points.
        Extra dimensions (4th, 5th, etc.) are preserved in the output.

        Args:
            transform (np.ndarray): A (4, 4) transformation matrix.
            points (np.ndarray): A (N, D) array or (D,) point with D >= 3.

        Returns:
            np.ndarray: Transformed points of shape (N, D) or (D,) if input was a single point.
        """
        if transform.shape != (4, 4):
            msg = "Transform must be a 4x4 matrix"
            raise ValueError(msg)

        points = np.asarray(points)
        single_point = points.ndim == 1

        if single_point:
            if points.shape[0] < 3:
                msg = "Point must have at least 3 dimensions"
                raise ValueError(msg)
            points = points[np.newaxis, :]  # (1, D)

        N, D = points.shape
        if D < 3:
            msg = "Points must have at least 3 dimensions"
            raise ValueError(msg)

        # Separate spatial (XYZ) and extra dimensions
        xyz = points[:, :3]  # (N, 3)
        extras = points[:, 3:] if D > 3 else None

        # Homogeneous coordinates
        ones = np.ones((N, 1), dtype=xyz.dtype)
        xyz_hom = np.hstack([xyz, ones])  # (N, 4)

        # Apply transformation
        xyz_transformed = (transform @ xyz_hom.T).T[:, :3]  # (N, 3)

        # Reattach extras
        if extras is not None:  # noqa: SIM108
            result = np.hstack([xyz_transformed, extras])  # (N, D)
        else:
            result = xyz_transformed  # (N, 3)

        return result[0] if single_point else result

    @staticmethod
    def invert_transform_matrix(matrix: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Invert a 4x4 homogeneous transformation matrix (rigid transform).

        Assumes the matrix represents a rigid transformation:
        upper-left 3x3 is a rotation matrix, and last column is translation.

        Args:
            matrix (np.ndarray): A (4, 4) transformation matrix.

        Returns:
            np.ndarray: The inverse transformation matrix.
        """
        if matrix.shape != (4, 4):
            msg = "Input must be a 4x4 matrix"
            raise ValueError(msg)

        R = matrix[:3, :3]
        t = matrix[:3, 3]

        R_inv = R.T  # for rotation matrices, inverse is transpose
        t_inv = -R_inv @ t

        T_inv = np.eye(4, dtype=matrix.dtype)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv

        return T_inv

    @staticmethod
    def invert_rotation_translation(rot: Quaternion, trans: np.ndarray) -> tuple[Quaternion, np.ndarray]:
        """
        Invert a rigid transformation represented by a quaternion and translation.

        Args:
            rot (Quaternion): The rotation as a unit quaternion.
            trans (np.ndarray): The translation as a (3,) vector.

        Returns:
            tuple: (inverted Quaternion, inverted translation vector)
        """
        if trans.shape != (3,):
            msg = "Translation must be a 3-element vector"
            raise ValueError(msg)

        rot_inv = rot.inverse
        trans_inv = -rot_inv.rotate(trans)

        return rot_inv, trans_inv

    @classmethod
    def from_components(cls, translation: npt.ArrayLike, rotation: Rotation) -> RigidTransform:
        """Initialize a rigid transform from translation and rotation
        components.

        When creating a rigid transform from a translation and rotation, the
        translation is applied after the rotation, such that
        ``tf = Tf.from_components(translation, rotation)``
        is equivalent to
        ``tf = Tf.from_translation(translation) * Tf.from_rotation(rotation)``.

        When applying a transform to a vector ``v``, the result is the
        same as if the transform was applied to the vector in the
        following way: ``tf.apply(v) == translation + rotation.apply(v)``

        Parameters
        ----------
        translation : array_like, shape (N, 3) or (3,)
            A single translation vector or a stack of translation vectors.
        rotation : `Rotation` instance
            A single rotation or a stack of rotations.

        Returns
        -------
        `RigidTransform`
            If rotation is single and translation is shape (3,), then a single
            transform is returned.
            Otherwise, a stack of transforms is returned.

        Examples
        --------
        >>> from navio3ddata.utils import RigidTransform as Tf
        >>> from scipy.spatial.transform import Rotation as R
        >>> import numpy as np

        Creating from a single rotation and translation:

        >>> t = np.array([2, 3, 4])
        >>> r = R.from_euler("ZYX", [90, 30, 0], degrees=True)
        >>> expected = np.array([[ 0.       , -1.,  0.        ],
        ...                      [ 0.8660254,  0.,  0.5       ],
        ...                      [-0.5      ,  0.,  0.8660254 ]])
        >>> np.allclose(r.as_matrix(), expected)
        True
        >>> tf = Tf.from_components(t, r)
        >>> expected = np.array([[ 0.       , -1.,  0.        ],
        ...                      [ 0.8660254,  0.,  0.5       ],
        ...                      [-0.5      ,  0.,  0.8660254 ]])
        >>> np.allclose(tf.rotation.as_matrix(), expected)
        True
        >>> tf.translation
        array([2., 3., 4.])

        When applying a transform to a vector ``v``, the result is the same as
        if the transform was applied to the vector in the following way:
        ``tf.apply(v) == translation + rotation.apply(v)``

        >>> np.allclose(r.apply([1, 0, 0]), np.array([0, 0.8660254, -0.5]))
        True
        >>> np.allclose(t + r.apply([1, 0, 0]), np.array([2, 3.8660254, 3.5]))
        True
        >>> np.allclose(tf.apply([1, 0, 0]), np.array([2, 3.8660254, 3.5]))
        True
        """
        return cls.from_translation(translation) * cls.from_rotation(rotation)

    def as_matrix(self) -> npt.NDArray[Any]:
        """Return a copy of the matrix representation of the transform.

        4x4 rigid transformation matrices are of the form::

            [       tx]
            [   R   ty]
            [       tz]
            [ 0 0 0  1]

        where ``R`` is a 3x3 orthonormal rotation matrix and
        ``t = [tx, ty, tz]`` is a 3x1 translation vector.

        Returns
        -------
        matrix : numpy.ndarray, shape (4, 4) or (N, 4, 4)
            A single transformation matrix or a stack of transformation
            matrices.

        Examples
        --------
        >>> from navio3ddata.utils import RigidTransform as Tf
        >>> from scipy.spatial.transform import Rotation as R
        >>> import numpy as np

        A transformation matrix is a 4x4 matrix formed from a 3x3 rotation
        matrix and a 3x1 translation vector:

        >>> t = np.array([2, 3, 4])
        >>> r = R.from_matrix([[0, 0, 1],
        ...                    [1, 0, 0],
        ...                    [0, 1, 0]])
        >>> tf = Tf.from_components(t, r)
        >>> expected = np.array([[ 0., 0., 1., 2.],
        ...                      [ 1., 0., 0., 3.],
        ...                      [ 0., 1., 0., 4.],
        ...                      [ 0., 0., 0., 1.]])
        >>> np.allclose(tf.as_matrix(), expected)
        True
        >>> np.allclose(Tf.identity().as_matrix(), np.identity(4))
        True
        """
        return self.matrix.copy()

    def __mul__(self, other: RigidTransform) -> RigidTransform:
        """Compose this transform with the other.

        If ``p`` and ``q`` are two transforms, then the composition of '``q``
        followed by ``p``' is equivalent to ``p * q``. In terms of
        transformation matrices, the composition can be expressed as
        ``p.as_matrix() @ q.as_matrix()``.

        In terms of translations and rotations, the composition when applied to
        a vector ``v`` is equivalent to
        ``p.translation + p.rotation.apply(q.translation)
        + (p.rotation * q.rotation).apply(v)``.

        This function supports composition of multiple transforms at a
        time. The following cases are possible:

            - Either ``p`` or ``q`` contains a single or length 1 transform. In
              this case the result contains the result of composing each
              transform in the other object with the one transform. If both are
              single transforms, the result is a single transform.
            - Both ``p`` and ``q`` contain ``N`` transforms. In this case each
              transform ``p[i]`` is composed with the corresponding transform
              ``q[i]`` and the result contains ``N`` transforms.

        Parameters
        ----------
        other : `RigidTransform` instance
            Object containing the transforms to be composed with this one.

        Returns
        -------
        `RigidTransform` instance
            The composed transform.

        Examples
        --------
        >>> from navio3ddata.utils import RigidTransform as Tf
        >>> from scipy.spatial.transform import Rotation as R
        >>> import numpy as np

        Compose two transforms:

        >>> tf1 = Tf.from_translation([1, 0, 0])
        >>> tf2 = Tf.from_translation([0, 1, 0])
        >>> tf = tf1 * tf2
        >>> tf.translation
        array([1., 1., 0.])

        When applied to a vector, the composition of two transforms is applied
        in right-to-left order.

        >>> t1, r1 = [1, 2, 3], R.from_euler('z', 60, degrees=True)
        >>> t2, r2 = [0, 1, 0], R.from_euler('x', 30, degrees=True)
        >>> tf1 = Tf.from_components(t1, r1)
        >>> tf2 = Tf.from_components(t2, r2)
        >>> tf = tf1 * tf2
        >>> tf.apply([1, 0, 0])
        array([0.6339746, 3.3660254, 3.       ])
        >>> tf1.apply(tf2.apply([1, 0, 0]))
        array([0.6339746, 3.3660254, 3.       ])

        """
        return other.compose(self)

    @property
    def dtype(self) -> np.dtype:
        return self._matrix.dtype

    def astype(self, dtype: npt.DTypeLike) -> Self:
        """Create new `RigidTransform` object with specified `dtype`

        Example:
        >>> tf = RigidTransform.identity(np.float64)
        >>> tf.dtype
        dtype('float64')
        >>> new_tf = tf.astype(np.float32)
        >>> new_tf.dtype
        dtype('float32')
        """
        return RigidTransform(self.matrix.astype(dtype))


def interpolate_transforms(
    transform0: RigidTransform,
    timestamp0: int,
    transform1: RigidTransform,
    timestamp1: int,
    interpolation_ts: int,
) -> RigidTransform:
    """
    Linearly interpolate (or extrapolate) between two rigid transforms.

    Args:
        transform0 (RigidTransform): one of two closest to target transform with earliest timestamp
        timestamp0 (int): timestamp of transform0
        transform1 (RigidTransform): one of two closest to target transform with latest timestamp
        timestamp1 (int): timestamp of transform1
        interpolation_ts (int): target timestamp for interpolation or extrapolation

    Returns:
        RigidTransform: calculated transform at interpolation_ts

    Example:
        >>> from navio3ddata.utils import RigidTransform, interpolate_transforms
        >>> from scipy.spatial.transform import Rotation
        >>> import numpy as np
        >>> t0 = np.array([0.0, 0.0, 0.0])
        >>> r0 = Rotation.from_euler('z', 0, degrees=True)
        >>> t1 = np.array([1.0, 0.0, 0.0])
        >>> r1 = Rotation.from_euler('z', 90, degrees=True)
        >>> tf0 = RigidTransform.from_translation_rotation(translation=t0, rotation=r0)
        >>> tf1 = RigidTransform.from_translation_rotation(translation=t1, rotation=r1)
        >>> interp_tf = interpolate_transforms(tf0, 0, tf1, 2, 1)
        >>> np.allclose(interp_tf.translation, [0.5, 0.0, 0.0])
        True
        >>> np.allclose(interp_tf.rotation.as_euler('zxy', degrees=True), [45.0, 0.0, 0.0])
        True
    """
    if timestamp1 == timestamp0:
        return transform0

    rotation0 = transform0.rotation
    rotation1 = transform1.rotation

    # relative rotation from R0 to R1
    delta_R = rotation0.inv() * rotation1

    # rotation vector (axis * angle) of delta_R
    rotvec = delta_R.as_rotvec()

    # constant angular velocity vector
    omega = rotvec / (timestamp1 - timestamp0)

    # extrapolate by Δt = ts - ts0
    dt = interpolation_ts - timestamp0
    R_delta = Rotation.from_rotvec(omega * dt)

    # apply on top of R0
    extrapolated_rotation = rotation0 * R_delta

    translation0 = transform0.translation
    translation1 = transform1.translation

    alpha = (interpolation_ts - timestamp0) / (timestamp1 - timestamp0)

    extrapolated_translation = translation0 + alpha * (translation1 - translation0)

    return RigidTransform.from_translation_rotation(
        translation=extrapolated_translation,
        rotation=extrapolated_rotation,
        dtype=transform0.matrix.dtype,
    )


def angle_diff_da(a0: float, a1: float) -> float:
    diff = abs(a0 - a1) % (2 * np.pi)
    diff = min(diff, 2 * np.pi - diff)
    if diff > np.pi / 2:
        diff = np.pi - diff
    return diff
