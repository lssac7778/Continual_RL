from utils import *
from grad_cam_batch import GradCam
import copy
import numpy as np
import pickle
import gc

def collect_traj(agent, env, steps, device):
    verbose_step = 1000
    
    with torch.no_grad():
        agent.eval()
        
        frame = env.reset()
        frames_, actions_, values_, logits_, features_, epinfos = [], [], [], [], [], []

        for s in range(steps):
            frame = np_to_pytorch_img(frame)
            frames_.append(frame)
            
            action_probs, state_estimate, logits, features = agent.forward_ftre_lgit(frame.to(device))
            
            action = get_action(action_probs).cpu().numpy()
            frame, reward, done, info = env.step(action)
            
            actions_.append(torch.from_numpy(action))
            logits_.append(logits)
            features_.append(features)
            values_.append(state_estimate)

            for i_info in info:
                episode_info = i_info.get('episode')
                if episode_info: epinfos.append(episode_info)
                    
            if s%verbose_step==0:
                print("collecting trajectory {}/{}".format(s, steps))
        
        results = [frames_, actions_, values_, logits_, features_]
        
        for i in range(len(results)):
            results[i] = torch.cat(results[i], 0)
        
        return results

class simple_a2cppo_gradcam:
    
    feature_modules = ["block1", "block2", "block3"]
    
    def __init__(self, model, 
                 feature_module_str="block2", 
                 target_layer_str="res2", 
                 use_cuda=True, 
                 trainable=False, 
                 norm_scale = True):
        
        cam_model = copy.deepcopy(model)
        
        self.cam_model = cam_model
        del cam_model.critic
        
        self.action_size = cam_model.actor.out_features
        self.use_cuda = use_cuda
        self.model = model
        
        assert feature_module_str in self.feature_modules
        
        self.grad_cam = GradCam(model=cam_model,
                        feature_module=getattr(cam_model, feature_module_str),
                        target_layer_names=target_layer_str,
                        use_cuda=use_cuda,
                        trainable=trainable,
                        norm_scale=norm_scale)
    
    def __call__(self, inputs, action_index, normalize_size={"min":0, "max":1}):
        if self.use_cuda:
            inputs = inputs.cuda()
        return self.grad_cam(inputs, action_index, normalize_size=normalize_size)
    
    def batch_call(self, inputs, action_index, batch_size, normalize_size={"min":0, "max":1}):
            
        data_len = inputs.size()[0]
        cams = []
        checkpoint = 0
        
        for idx in range(0, data_len, batch_size):
            
            if idx+batch_size > data_len:
                
                batch_inputs = inputs[idx:]
                batch_action_index = action_index[idx:]
            else:
                
                batch_inputs = inputs[idx:idx+batch_size]
                batch_action_index = action_index[idx:idx+batch_size]
            
            '''
            if self.use_cuda:
                batch_inputs = copy.deepcopy(batch_inputs)
                
                batch_inputs = batch_inputs.cuda()
            '''
            
            cam = self.grad_cam(batch_inputs,
                                batch_action_index,
                                normalize_size=normalize_size)
                
            cams.append(cam)
            
            del batch_inputs
            del batch_action_index
            
            if idx > checkpoint:
                print("generating grad-cam {}/{}".format(idx, data_len))
                checkpoint += 50000
        
        if type(cams[0])==np.ndarray:
            cams = np.concatenate(cams, axis=0)
        else:
            cams = torch.cat(cams, 0)
        return cams
        

class Storage:
    def __init__(self, batch_size, traj_size, device, memeffi=False):
        self.data = {}
        self.data_names = {}
        self.env_indexs = []
        self.batch_size = batch_size
        self.traj_size = traj_size
        
        self.device = device
        
        self.memeffi = memeffi
    
    def append(self, env_index, frames, actions, values, features, logits, cam):
        ''' all inputs are tensors except gradcam '''
        
        self.env_indexs.append(env_index)
        
        data = {'frames':frames.detach(),
                  'actions':actions.detach(),
                  'values':values.detach(),
                  #'features':features.detach(),
                  'logits':logits.detach(),
                  'cams':cam.detach()}
        
        if not self.memeffi:
            self.data[env_index] = data
        else:
            data_name = "traj" + str(env_index) + ".pickle"
            self.data_names[env_index] = data_name
            print("saving traj")
            self.save_obj(data_name, data)
            del frames
            del actions
            del features
            del logits
            del cam
            gc.collect()
        print("...done")

    def get_batch(self):
        print("get batch from saved data")
        results = {'frames':[],
                  'actions':[],
                  'values':[],
                  #'features':[],
                  'logits':[],
                  'cams':[]}
        
        perm = torch.randperm(self.traj_size)
        indices = perm[:self.batch_size]
        
        for idx in self.env_indexs:
            
            if not self.memeffi:
                traj = self.data[idx]
            else:
                traj = self.load_obj(self.data_names[idx])
            
            for key in results.keys():
                temp = copy.deepcopy(torch.index_select(traj[key].cpu(), 0, indices.cpu()))
                del traj[key]
                gc.collect()
                
                temp = temp.to(self.device)
                
                results[key].append(temp)
            
            del traj
        
        for key in results.keys():
            results[key] = torch.cat(results[key], 0)
            
            
        return results.values()
    
    def save_obj(self, name, data):
        with open(name, 'wb') as f:
            pickle.dump(data, f, protocol=4)
    
    def load_obj(self, name):
        with open(name, 'rb') as f:
            data = pickle.load(f)
        return data
        
        
        
        