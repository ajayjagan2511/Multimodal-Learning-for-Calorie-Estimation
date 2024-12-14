import numpy as np
import torch
from torchvision import transforms

import numpy as np
import torch

def transform_images(image_column,transform):
  processed_images = []
  for img in image_column:
      try:
          img_array = np.array(eval(img)).astype(np.float32)
          if img_array.ndim == 1:
              raise ValueError("Image data should be 2D or 3D, but found 1D.")
          if img_array.ndim == 2:
              img_array = np.stack([img_array] * 3, axis=-1)
          processed_images.append(transform(img_array))
      except Exception as e:
          processed_images.append(torch.zeros((3, 224, 224)))
  return processed_images