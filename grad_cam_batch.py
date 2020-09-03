import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import copy

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        
        def call_recurr(x, parent_module, ac_module_names, outputs, idx, target_layers):
            
            if len(parent_module._modules.items())==0:
                return parent_module(x)
            else:
                for name, module in parent_module._modules.items():
                    if name==target_layers[idx] and idx < len(target_layers)-1:
                        idx += 1
                        temp_ac_module_names = copy.deepcopy(ac_module_names)
                        temp_ac_module_names.append(name)
                        x = call_recurr(x, module, temp_ac_module_names, outputs, idx, target_layers)
                    else:
                        x = module(x)
                        if name == target_layers[idx] and ac_module_names==target_layers[:-1]:
                            x.register_hook(self.save_gradient)
                            outputs += [x]
                return x
        
        x = call_recurr(x, self.model, [], outputs, 0, self.target_layers)
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)
                #x = module(x, torch.zeros(1, 1), torch.zeros(1, 1))
        
        return target_activations, x


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda, trainable=False, norm_scale=True):
        self.model = model
        self.feature_module = feature_module
        self.trainable = trainable
        self.norm_scale = norm_scale
        if not trainable:
            self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)
    
    def get_grad_cam_trainable(self, grads_val, features, input):
        grads_val = grads_val[-1]
        
        #(n, 32, 20, 20)
        target = features[-1]
        #(n, 32)
        weights = torch.mean(grads_val, axis=(2, 3))
        #(n, 32, 1)
        weights = torch.unsqueeze(weights, axis=-1)
        #(n, 32, 20, 20)
        weights = weights.repeat(1, 1, target.shape[2]*target.shape[3])
        weights = torch.reshape(weights, target.shape)
        
        #(n, 32, 20, 20)
        cam = target*weights
        return cam
    
    
    def get_grad_cam_no_norm_scale(self, grads_val, features, input):
        
        grads_val = [gval.cpu().data.numpy() for gval in grads_val]
        grads_val = grads_val[-1]
        
        target = features[-1]
        #(n, 32, 20, 20)
        target = target.cpu().data.numpy()
        #(n, 32)
        weights = np.mean(grads_val, axis=(2, 3))
        #(n, 32, 1)
        weights = np.expand_dims(weights, axis=-1)
        #(n, 32, 20, 20)
        weights = np.repeat(weights, target.shape[2]*target.shape[3], axis=2)
        weights = weights.reshape(target.shape)
        
        #(n, 32, 20, 20)
        cam = target*weights
        return cam
    
    def get_grad_cam(self, grads_val, features, input, normalize_size):
        
        grads_val = [gval.cpu().data.numpy() for gval in grads_val]
        grads_val = grads_val[-1]
        
        target = features[-1]
        #(n, 32, 20, 20)
        target = target.cpu().data.numpy()
        #(n, 32)
        weights = np.mean(grads_val, axis=(2, 3))
        #(n, 32, 1)
        weights = np.expand_dims(weights, axis=-1)
        #(n, 32, 20, 20)
        weights = np.repeat(weights, target.shape[2]*target.shape[3], axis=2)
        weights = weights.reshape(target.shape)
        
        #(n, 32, 20, 20)
        cam = target*weights
        #(n,20,20)
        cam = np.sum(cam, axis=1)
        total_shape = tuple([cam.shape[0]] + list(input.shape[2:]))
        result_cam = []

        for i in range(len(cam)):
            cam_el = cam[i]
            
            cam_el = np.maximum(cam_el, 0)
            cam_el = cv2.resize(cam_el, dsize=input.shape[2:])
            
            cam_el = cam_el - np.min(cam_el)
            cam_el = cam_el / (np.max(cam_el) + 1e-10)
            
            cam_el = cam_el * (normalize_size["max"] - normalize_size["min"])
            cam_el = cam_el + normalize_size["min"]
            
            result_cam.append(cam_el)
        return np.array(result_cam)
            
            
    def __call__(self, input, index=None, normalize_size={"min":0, "max":1}):
        
        '''get feature and gradients'''
        
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros(output.size(), dtype=np.float32)
        one_hot[:, index] = 1
        
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        
        grads_val = self.extractor.get_gradients()
        
        '''get grad cam'''
        
        if not self.trainable:
            if self.norm_scale:
                return self.get_grad_cam(grads_val, features, input, normalize_size)
            else:
                return self.get_grad_cam_no_norm_scale(grads_val, features, input)
        else:
            return self.get_grad_cam_trainable(grads_val, features, input)


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply
                
        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    model = models.resnet50(pretrained=True)
    grad_cam = GradCam(model=model, feature_module=model.layer4, \
                       target_layer_names=["2"], use_cuda=args.use_cuda)

    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask = grad_cam(input, target_index)

    show_cam_on_image(img, mask)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    print(model._modules.items())
    gb = gb_model(input, index=target_index)
    gb = gb.transpose((1, 2, 0))
    cam_mask = cv2.merge([mask, mask, mask])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)

    cv2.imwrite('gb.jpg', gb)
    cv2.imwrite('cam_gb.jpg', cam_gb)