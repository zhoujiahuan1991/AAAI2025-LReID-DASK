from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
import collections
import numpy as np


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))
class CM_weight(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y,mom in zip(inputs, targets,ctx.momentum):
            ctx.features[y] = mom * ctx.features[y] + (1. - mom) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None
def cm_weight(inputs, indexes, features, momentum=0.5):
    return CM_weight.apply(inputs, indexes, features, momentum)

def cm(inputs, indexes, features, momentum=0.5):
    if isinstance(momentum, torch.Tensor):
        return CM.apply(inputs, indexes, features, momentum.to(inputs.device))
    else:
        return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))

class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, indexes.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    if isinstance(momentum, torch.Tensor):
        return CM_Hard.apply(inputs, indexes, features, momentum.to(inputs.device))
    else:
        return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        # self.cluster_stds=cluster_stds.cuda()
        # self.cluster_stds=self.cluster_stds.clamp(min=self.cluster_stds.mean())/self.cluster_stds.mean()# 计算均值为下届

        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets):

        inputs = F.normalize(inputs, dim=1).cuda()
        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)

        outputs /= self.temp
        # outputs/=self.cluster_stds  # 标准差作为加权
        loss = F.cross_entropy(outputs, targets)
        return loss
class ClusterMemory_weight(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False,cluster_stds=None,weight=None):
        super(ClusterMemory_weight, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.cluster_stds=cluster_stds.cuda()
        self.cluster_stds=self.cluster_stds.clamp(min=self.cluster_stds.mean())/self.cluster_stds.mean()# 计算均值为下届

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.weight=weight.cuda()
        # self.momentum=self.momentum*self.weight

        
        

    def forward(self, inputs, targets,inds):

        inputs = F.normalize(inputs, dim=1).cuda()
        weight=self.weight[inds]
        momentum =self.momentum-weight*self.momentum    # 分数越低，新数据特征权重越小
        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, momentum.unsqueeze(-1))
        else:
            outputs = cm_weight(inputs, targets, self.features, momentum=momentum)

        outputs /= self.temp
        # outputs/=self.cluster_stds  # 标准差作为加权
        loss = F.cross_entropy(outputs, targets,reduction='none')
        loss=(loss*weight.to(loss.device)).mean()
        return loss

class ClusterMemory_part(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False, lambda2=None, mu=None):
        super(ClusterMemory_part, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        self.lambda2 = lambda2
        self.mu = mu

        #self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, inputs_up, inputs_down, inputs_teacher, inputs_up_teacher, inputs_down_teacher, targets, epoch):

        inputs = F.normalize(inputs, dim=1).cuda()
        inputs_up = F.normalize(inputs_up, dim=1).cuda()
        inputs_down = F.normalize(inputs_down, dim=1).cuda()
        inputs_teacher = F.normalize(inputs_teacher, dim=1).cuda()
        inputs_up_teacher = F.normalize(inputs_up_teacher, dim=1).cuda()
        inputs_down_teacher = F.normalize(inputs_down_teacher, dim=1).cuda()

        if epoch == -1:
            outputs = cm(inputs, targets, self.features, 1.0)
            outputs_up = cm(inputs_up, targets, self.features_up, 1.0)
            outputs_down = cm(inputs_down, targets, self.features_down, 1.0)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)
            outputs_up = cm(inputs_up, targets, self.features_up, self.momentum)
            outputs_down = cm(inputs_down, targets, self.features_down, self.momentum)

        outputs /= self.temp
        outputs_up /= self.temp
        outputs_down /= self.temp


        loss_distill = nn.MSELoss(reduce=False)(inputs, inputs_teacher).mean(0).sum()
        loss_distill_up = nn.MSELoss(reduce=False)(inputs_up, inputs_up_teacher).mean(0).sum()
        loss_distill_down = nn.MSELoss(reduce=False)(inputs_down, inputs_down_teacher).mean(0).sum() 
        loss = (1.0-self.lambda2) * (F.cross_entropy(outputs, targets) + self.mu * loss_distill) + self.lambda2 * (F.cross_entropy(outputs_up, targets) + self.mu * loss_distill_up) + self.lambda2 * (F.cross_entropy(outputs_down, targets) + self.mu * loss_distill_down) 

        return loss
class ClusterMemory_part_source(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False, lambda2=None, mu=None):
        super(ClusterMemory_part_source, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        self.lambda2 = lambda2
        self.mu = mu

        #self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, inputs_up, inputs_down, targets, epoch):
        inputs = F.normalize(inputs, dim=1).cuda()
        inputs_up = F.normalize(inputs_up, dim=1).cuda()
        inputs_down = F.normalize(inputs_down, dim=1).cuda()
        

        if epoch == -1:
            outputs = cm(inputs, targets, self.features, 1.0)
            outputs_up = cm(inputs_up, targets, self.features_up, 1.0)
            outputs_down = cm(inputs_down, targets, self.features_down, 1.0)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)
            outputs_up = cm(inputs_up, targets, self.features_up, self.momentum)
            outputs_down = cm(inputs_down, targets, self.features_down, self.momentum)

        outputs /= self.temp
        outputs_up /= self.temp
        outputs_down /= self.temp

       
        loss = (1.0-self.lambda2) * (F.cross_entropy(outputs, targets) ) + self.lambda2 * (F.cross_entropy(outputs_up, targets) ) + self.lambda2 * (F.cross_entropy(outputs_down, targets) ) 

        return loss