from math import exp
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.

    Returns:
        torch.Tensor: the labels in one hot tensor of shape :math:`(N, C, *)`,

    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> kornia.losses.one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    shape = labels.shape
    one_hot = torch.zeros(shape[0], num_classes, *shape[1:],
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def focal_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        gamma: float = 2.0,
        reduction: str = 'none',
        eps: float = 1e-8,
        ignore_index: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Function that computes Focal loss.
    
    See :class:`~kornia.losses.FocalLoss` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                         .format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(
            out_size, target.size()))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {}" .format(
                input.device, target.device))
        
    if ignore_index is not None:
        if 0 <= ignore_index <= input.shape[1]:
            raise ValueError("ignore index cannot be equal to class labels.")
            
    else:
        if torch.sum((target>input.shape[1])+(target<0)) != 0:
            raise ValueError("class labels must be between 0 and {}".format(
                    input.shape[1]-1))

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1) + eps

    # create the labels one hot tensor
    if ignore_index is None:
        
        target_one_hot: torch.Tensor = one_hot(
                target, num_classes=input.shape[1],
                device=input.device, dtype=input.dtype)
    else:
        
        if torch.sum(target==ignore_index) == 0:
            target_one_hot: torch.Tensor = one_hot(
                    target, num_classes=input.shape[1],
                    device=input.device, dtype=input.dtype)
        else:
            target_copy = torch.clone(target)
            target_copy[target == ignore_index] = input.shape[1]
            target_one_hot: torch.Tensor = one_hot(
                    target_copy, num_classes=input.shape[1]+1,
                    device=input.device, dtype=input.dtype)
            # create mask and implement it
            mask = 1. - target_one_hot[:,-1].unsqueeze(1)
            target_one_hot = target_one_hot[:,:-1] * mask
        
    
    # create weight denoted by alpha
    dims = (2, 3)
    target_class = target_one_hot.sum(dims)
    idx = target_class < 1.0
    dims = (1,)
    target_image = target_class.sum(dims).unsqueeze(-1).expand_as(target_class)
    target_image[idx] = 0
    dims = (0,)
    alpha = target_class.sum(dims) / (target_image.sum(dims) + eps)
    alpha = (alpha.mean() / alpha).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)


    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1., gamma)

    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.sum(loss_tmp) / torch.sum(target_one_hot)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}"
                                  .format(reduction))
    return loss



class WeightedFocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to [1], the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.


    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> loss = kornia.losses.FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(self, gamma: float = 2.0,
                 reduction: str = 'none',
                 ignore_index: Optional[torch.Tensor] = None) -> None:
        super(WeightedFocalLoss, self).__init__()
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = 1e-6
        self.ignore_index = ignore_index

    def forward(  # type: ignore
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.gamma,
                          self.reduction, self.eps,
                          self.ignore_index)
        

def gaussian(window_size, sigma):
	gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
	return gauss / gauss.sum()
# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
	_1D_window = gaussian(window_size, 1.5).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
	return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False):
	padd = 5
	(_, channel, height, width) = img1.size()
	if window is None:
		real_size = min(window_size, height, width)
		window = create_window(real_size, channel=channel).to(img1.device)
	
	mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
	mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
	
	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1 * mu2
	
	sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
	sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
	sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
	
	C1 = (0.01 ) ** 2
	C2 = (0.03 ) ** 2
	
	v1 = 2.0 * sigma12 + C2
	v2 = sigma1_sq + sigma2_sq + C2
	cs = torch.mean(v1 / v2)  # contrast sensitivity
	
	ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
	
	if size_average:
		ret = ssim_map.mean()
	else:
		ret = ssim_map.mean(1).mean(1).mean(1)
	
	if full:
		return ret, cs
	return ret

class SSIM(nn.Module):
	def __init__(self, windows_size=11, size_average=True, val_range=None):
		super(SSIM, self).__init__()
		self.windows_size = windows_size
		self.size_average = size_average
		self.val_range = val_range
		
		self.channel = 1
		self.window = create_window(windows_size)
	
	def forward(self, x, y):
		(_, channel, _, _) = x.size()
		if channel == self.channel and self.window.dtype == x.dtype:
			window = self.window.to(x.device)
		else:
			window = create_window(self.windows_size, channel).to(x.device).type(x.dtype)
			self.window = window
			self.channel = channel
		
		return 1-ssim(x, y, window=window, window_size=self.windows_size, size_average=self.size_average)

if __name__ == '__main__':
	data = torch.zeros(1, 1, 128,128)
	data += 0.01
	target = torch.zeros(1, 1, 128, 128)
	data = Variable(data, requires_grad=True)
	target = Variable(target)
	
	model = SSIM()
	loss = model(data, target)
	loss.backward()
	