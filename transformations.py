import abc
import itertools
import numpy as np
from keras.preprocessing.image import apply_affine_transform

class AffineTransformation(object):
    def __init__(self, flip, tx, ty, k_90_rotate):
        self.flip = flip
        self.tx = tx
        self.ty = ty
        self.k_90_rotate = k_90_rotate

    def __call__(self, x):
        res_x = x
        if self.flip:
            res_x = np.fliplr(res_x)
        if self.tx != 0 or self.ty != 0:
            res_x = apply_affine_transform(res_x, tx=self.tx, ty=self.ty, channel_axis=2, fill_mode='reflect')
        if self.k_90_rotate != 0:
            res_x = np.rot90(res_x, self.k_90_rotate)
        return res_x

class TabularTransformation(object):
    def __init__(self, type_id):
        self.type_id = type_id

    def __call__(self, x):
        x_out = x.copy()
        if self.type_id == 0:
            pass
        elif self.type_id == 1:
            x_out += np.random.normal(0, 0.005, size=x_out.shape)
        elif self.type_id == 2:
            if len(x_out) > 1:
                idx1, idx2 = np.random.choice(len(x_out), 2, replace=False)
                x_out[idx1], x_out[idx2] = x_out[idx2], x_out[idx1]
        elif self.type_id == 3:
            idx = np.random.randint(len(x_out))
            x_out[idx] = 0.0
        elif self.type_id == 4:
            x_out *= np.random.uniform(0.9, 1.1)
        elif self.type_id == 5:
            x_out = -x_out
        elif self.type_id == 6:
            x_out += np.random.normal(0, 0.05)
        elif self.type_id == 7:
            np.random.shuffle(x_out)
        return x_out


class AbstractTransformer(abc.ABC):
    def __init__(self):
        self._transformation_list = None
        self._create_transformation_list()

    @property
    def n_transforms(self):
        return len(self._transformation_list)

    @abc.abstractmethod
    def _create_transformation_list(self):
        pass

    def transform_batch(self, x_batch, t_inds):
        assert len(x_batch) == len(t_inds)
        transformed_batch = x_batch.copy()
        for i, t_ind in enumerate(t_inds):
            transformed_batch[i] = self._transformation_list[t_ind](transformed_batch[i])
        return transformed_batch

class Transformer(AbstractTransformer):
    """ Standard image transformer with translations and rotations """
    def __init__(self, translation_x=8, translation_y=8):
        self.max_tx = translation_x
        self.max_ty = translation_y
        super().__init__()

    def _create_transformation_list(self):
        transformation_list = []
        for is_flip, tx, ty, k_rotate in itertools.product((False, True),
                                                           (0, -self.max_tx, self.max_tx),
                                                           (0, -self.max_ty, self.max_ty),
                                                           range(4)):
            transformation = AffineTransformation(is_flip, tx, ty, k_rotate)
            transformation_list.append(transformation)
        self._transformation_list = transformation_list

class SimpleTransformer(AbstractTransformer):
    def _create_transformation_list(self):
        transformation_list = []
        for is_flip, k_rotate in itertools.product((False, True), range(4)):
            transformation = AffineTransformation(is_flip, 0, 0, k_rotate)
            transformation_list.append(transformation)
        self._transformation_list = transformation_list

class TabularTransformer(AbstractTransformer):
    """ Transformer for tabular data using 8 types of corruptions """
    def __init__(self, n_transforms=8):
        self._n_transforms = n_transforms
        super().__init__()

    def _create_transformation_list(self):
        self._transformation_list = [TabularTransformation(i) for i in range(self._n_transforms)]


class TabularTransformerIEEE(object):
    def __init__(self, n_transforms=8):
        self.n_transforms = n_transforms

    def transform_batch(self, x, t_inds):
        x_out = x.copy()
        for i, t_idx in enumerate(t_inds):
            if t_idx == 0:
                continue
            elif t_idx == 1:
                x_out[i] += np.random.normal(0, 0.02, size=x_out[i].shape)
            elif t_idx == 2:
                if len(x_out[i]) > 1:
                    idx1, idx2 = np.random.choice(len(x_out[i]), 2, replace=False)
                    x_out[i][idx1], x_out[i][idx2] = x_out[i][idx2], x_out[i][idx1]
            elif t_idx == 3:
                idx = np.random.randint(len(x_out[i]))
                x_out[i][idx] = 0.0
            elif t_idx == 4:
                x_out[i] *= np.random.uniform(0.5, 1.5)
            elif t_idx == 5:
                x_out[i] = -x_out[i]
            elif t_idx == 6:
                x_out[i] += np.random.normal(0, 0.1)
            elif t_idx == 7:
                np.random.shuffle(x_out[i])
        return x_out