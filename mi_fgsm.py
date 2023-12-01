import torch
import torch.nn as nn
from utils import *
import numpy as np
import config as flags
from cam_model import *
import torchvision.transforms as transforms

import torch.nn.functional as F
import torch.autograd as autograd


def imagenet_resize_postfn(grad):
    grad = grad.abs().max(1, keepdim=True)[0]
    grad = F.avg_pool2d(grad, 4).squeeze(1)
    shape = grad.shape
    grad = grad.view(len(grad), -1)
    grad_min = grad.min(1, keepdim=True)[0]
    grad = grad - grad_min
    grad_max = grad.max(1, keepdim=True)[0]
    grad = grad / torch.max(grad_max, torch.tensor([1e-8], device='cuda'))
    return grad.view(*shape)


def generate_gs_per_batches(model, bx, by, post_fn=None, keep_grad=False):
    logit = model(bx)
    loss = F.nll_loss(F.log_softmax(logit), by)
    grad = autograd.grad([loss], [bx], create_graph=keep_grad)[0]
    if post_fn is not None:
        grad = post_fn(grad)
    return grad
    


class MI_FGSM_ENS(object):

    def __init__(self,models, weights=None, epsilon=0.1, stepsize=0.01, iters=10, mu=1,
                 random_start=True, loss_fn=nn.CrossEntropyLoss(),pnorm=np.inf,
                 clip_min=0, clip_max=1,targeted=False, position='loss'):
        '''
        :param models:
        :param weights:
        :param position: ensemble position, logits, probabilities, loss
        '''
        self.models = []
        for model in models:
            self.models.append(model.to(flags.device))

        if weights is None:
            num_ensemble = len(self.models)
            self.weights = [1./num_ensemble]*num_ensemble
        else:
            self.weights = weights

        self.epsilon = epsilon
        self.stepsize = stepsize
        self.iters = iters
        self.mu = mu
        self.random_start = random_start

        self.loss_fn = loss_fn
        self.pnorm = pnorm
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted

        self.ben_cams = []
        self.batch_size = None

        self.position = position
        
        self.preprocess = transforms.Compose([
	    transforms.Resize(224),
	])

    def perturb(self, x, y):
        self.targeted = False
        x = x.clone()
        x, y = x.to(flags.device), y.to(flags.device)
        self.batch_size = x.size(0)
        self.ben_cams = self.generate_cams(x.clone().detach().requires_grad_(), y)
        # Call MI_FGSM Attack
        target_label = y.clone()
        img = None

        while((target_label.cpu().detach().numpy() == y.cpu().detach().numpy()).all()):
          img = self.__mi_fgsm_attack(x, y)
          target_label = torch.argmax(self.models[0](self.preprocess(img)), 1)
        
        self.targeted = True
        print('Found')
        
        return self.__int_fgsm_attack(img, target_label)


    def ensemble_logits(self, x, y):
        # ensemble in logits with same weight
        logits = 0
        for model, w in zip(self.models, self.weights):
            logits += model(x) * w

        loss = self.loss_fn(logits, y)  # Calculate the loss
        # target or untarget
        if self.targeted:
            loss = -loss

        return loss


    def generate_cams(self, x, y):
        temp = []
        for model, w in zip(self.models, self.weights):
            gs = generate_gs_per_batches(model, self.preprocess(x).to(flags.device), y.to(flags.device), post_fn=imagenet_resize_postfn)
            #cam_model = CAM(model)
            #_, cams = cam_model(self.preprocess(x), y)
            temp.append(gs)
            
        
        return temp
    
    def interpretation_loss(self, adv_cams):

        int_loss = 0
        for ind, adv_cam in enumerate(adv_cams):
        	
        	ben_cam = self.ben_cams[ind]
        	
        	diff = adv_cam - ben_cam
        	loss = (diff * diff).view(len(adv_cam), -1).sum()
        	int_loss += loss * self.weights[ind]

        return int_loss


    def ensemble_probabilities(self):
        pass

    def ensemble_loss(self, x, y):
        # ensemble in logits with same weight
        ensemble_loss = 0
        for model, w in zip(self.models, self.weights):
            logits = model(self.preprocess(x))

            loss = self.loss_fn(logits, y)  # Calculate the loss
            # target or untarget
            if self.targeted:
                loss = -loss

            ensemble_loss += loss *w

        return ensemble_loss

    def __mi_fgsm_attack(self, x, y):
        
        if self.random_start:
            noise = np.random.uniform(-self.epsilon, self.epsilon, x.size()).astype('float32')
            noise = torch.from_numpy(noise).to(flags.device)
        else:
            noise = 0

        # perturbation
        delta = torch.zeros_like(x) + noise
        delta = torch.clamp(x + delta, self.clip_min, self.clip_max) - x
        delta.requires_grad = True

        g = 0
        for i in range(self.iters):

            x_nes = x + delta
            x_nes = torch.clamp(x_nes, 0, 1)

            for model in self.models:
                model.zero_grad()

            if self.position == 'logits':
                loss = self.ensemble_logits(x_nes, y)
            if self.position == 'loss':
                loss = self.ensemble_loss(x_nes, y)
            
            #adv_cams = self.generate_cams(x_nes.clone().detach().requires_grad_(), y)
            #int_loss = self.interpretation_loss(adv_cams)
            total_loss = loss# + 5 * int_loss
	
            total_loss.backward(retain_graph=True)
            grad_data = delta.grad.data

            # this is the wrong code, but it works better on mnist  todo verify it on imagenet
            # g = self.mu * g + grad_data / torch.batch_norm(grad_data, p=1)
            # this is the stadrad MI-FGSM
            g = self.mu * g + normalize(grad_data, p=1)

            if self.pnorm == np.inf:

                delta.data += self.stepsize * g.sign()
                # clamp accm perturbation to [-epsilon, epsilon]
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
            # pnorm = 2
            else:
                delta.data += self.stepsize * normalize(g, p=2)
                delta.data = clamp_by_2norm(delta.data, self.epsilon)

            delta.data = torch.clamp(
                x + delta.data, self.clip_min, self.clip_max) - x

            delta.grad.data.zero_()
            # clear cache,
            torch.cuda.empty_cache()

        return torch.clamp(x + delta.data, self.clip_min, self.clip_max)
        
        

    def __int_fgsm_attack(self, x, y):
        
        noise = 0

        # perturbation
        delta = torch.zeros_like(x) + noise
        delta = torch.clamp(x + delta, self.clip_min, self.clip_max) - x
        delta.requires_grad = True

        g = 0
        for i in range(self.iters):

            x_nes = x + delta
            x_nes = torch.clamp(x_nes, 0, 1)
            for model in self.models:
                model.zero_grad()

            if self.position == 'logits':
                loss = self.ensemble_logits(x_nes, y)
            if self.position == 'loss':
                loss = self.ensemble_loss(x_nes, y)
            
            adv_cams = self.generate_cams(x_nes.clone().detach().requires_grad_(), y)
            int_loss = self.interpretation_loss(adv_cams)
            total_loss = loss + 0.001 * int_loss
	
            total_loss.backward(retain_graph=True)
            grad_data = delta.grad.data

            # this is the wrong code, but it works better on mnist  todo verify it on imagenet
            # g = self.mu * g + grad_data / torch.batch_norm(grad_data, p=1)
            # this is the stadrad MI-FGSM
            g = self.mu * g + normalize(grad_data, p=1)

            if self.pnorm == np.inf:

                delta.data += self.stepsize * g.sign()
                # clamp accm perturbation to [-epsilon, epsilon]
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
            # pnorm = 2
            else:
                delta.data += self.stepsize * normalize(g, p=2)
                delta.data = clamp_by_2norm(delta.data, self.epsilon)

            delta.data = torch.clamp(
                x + delta.data, self.clip_min, self.clip_max) - x

            delta.grad.data.zero_()
            # clear cache,
            torch.cuda.empty_cache()

        return torch.clamp(x + delta.data, self.clip_min, self.clip_max)
