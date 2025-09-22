# neural bayesian framework for transition & cost model
import torch
import numpy as np
import os
import time
import torch

from datetime import datetime
from stein_thinning.thinning import thin
from stein_thinning.kernel import make_imq
from numpy import linalg as LA

from scipy.spatial.distance import pdist
from scipy.stats import invgamma

import pickle
import warnings
import math

#ksd
from .ksdp import *
from .ksdp import utils
from .ksdp import ksd

warnings.filterwarnings("ignore")



def _to_np(a):
    # unwrap tensors, scalars, masked arrays, whatever—give me a bare ndarray
    return np.array(a)
    #return np.array(a, copy=False)


    

class neural_bays_dx_tf(object):
    def __init__(self, args, model, model_type, output_shape, device=None, train_x=None, train_y=None, rew = None, sigma_n2=0.1,
                 sigma2=0.1):
        self.model = model
        self.model_type = model_type
        self.args = args
        self.device = device
        self.train_x = train_x
        self.train_y = train_y
        self.rew = rew
        #def rewards
        # self.rew = rew
        self.output_shape = output_shape
        self.hidden_dim = 200 if self.model_type == "SAC" else 2*model.layers[0].get_input_dim()
        self.beta_s = None
        self.latent_z = None
        self.sigma2 = sigma2  # W prior variance
        self.sigma_n2 = sigma_n2  # noise variacne
        self.eye = np.eye(self.hidden_dim)
        self.mu_w = np.random.normal(loc=0, scale=.01, size=(output_shape, self.hidden_dim))
        self.cov_w = np.array([self.sigma2 * np.eye(self.hidden_dim) for _ in range(output_shape)])
        self.train_x_s = None
        self.train_y_s = None
        self.rew_s = None

    #primary main code where data is added
    def add_data(self, new_x, new_y, new_r, real = True):
        if real:
            if self.train_x is None:
                self.train_x = new_x
                self.train_y = new_y
                #add reward
                self.rew =  new_r

            else:
                 
                #add the thinning condition : Based on Posterior variane (if posterior variance > threshold)
                tx = _to_np(self.train_x)
                nx = _to_np(new_x)
                self.train_x = np.vstack((tx, nx))
                #print(tx.shape)
                #print(nx.shape)
                #print(self.train_x.shape)
                ty = _to_np(self.train_y)
                ny = _to_np(new_y)
                tr = _to_np(self.rew)
                nr = _to_np(new_r)

                self.train_x = np.vstack((tx, nx))
                            
                self.train_y = np.vstack((ty, ny))
                # print (torch.is_tensor(self.train_x))
                #add rewards
                self.rew = np.vstack((tr, nr))
                return self.train_x.shape
        else:
            if self.train_x_s is None:
                self.train_x_s = new_x
                self.train_y_s = new_y
                #add reward
                self.rew_s =  new_r

            else:

                #add the thinning condition : Based on Posterior variane (if posterior variance > threshold)
                tx = _to_np(self.train_x_s)
                nx = _to_np(new_x)
                self.train_x_s = np.vstack((tx, nx))
                #print(tx.shape)
                #print(nx.shape)
                #print(self.train_x_s.shape)
                ty = _to_np(self.train_y_s)
                ny = _to_np(new_y)
                tr = _to_np(self.rew_s)
                nr = _to_np(new_r)

                self.train_x_s = np.vstack((tx, nx))

                self.train_y_s = np.vstack((ty, ny))
                # print (torch.is_tensor(self.train_x))
                #add rewards
                self.rew_s = np.vstack((tr, nr))
                return self.train_x_s.shape


                

    def get_shape(self):
        return self.train_x.shape[0]


    #Add sorted data ###############################################################
    def get_sorted_data(self, episode):
        #reshape
        sort_rew = self.rew.flatten()
        ids_reward = np.argsort(sort_rew)[::-1]

        # print (np.argsort(self.rew)[::-1])
        # print (np.sort(self.rew)[::-1])
        # print (sort_rew)
        # print (self.x.shape, self.y.shape, self.rew)
        
        #get sorted
        self.train_x = self.train_x[ids_reward]
        self.train_y = self.train_y[ids_reward]
        self.rew = self.rew[ids_reward]

        # print (self.train_x.shape)

        #subset the samples by taking the top
        total_samples = self.train_x.shape[0]
        rew_nsamples =  int(30 + 0.03 * 0.1 * episode * total_samples)
        keep_samples = total_samples - rew_nsamples

        #update the dist
        self.train_x = self.train_x[0 : keep_samples]
        self.train_y = self.train_y[0 : keep_samples]
        self.rew = self.rew[0 : keep_samples]
    

    def generate_latent_z(self, real = True):
        # Update the latent representation of every datapoint collected so far
        if real:
            new_z = self.get_representation(self.train_x)
            self.latent_z = new_z
        else:
            # print ('the shape is' + str(self.train_x.shape))   ## 200 * 4
            # MODIFIED FOR REAL DATA
            # IN PREVIOS (PROBABLY WRONG ) VERSION WAS FOR SYNTHETIC DATA
            new_z = self.get_representation(self.train_x)
            self.latent_z_s = new_z


    #training the latent representation 
    def train(self, epochs = 5, real = True):
        if self.model_type == "SAC":
            self.model.learn(total_timesteps=10000)
        else:
            #print("TRAIN ", self.train_x.shape[0])
            #print("TRAIN TARGETs ", self.train_y.shape[0])
            self.model.train(self.train_x,self.train_y,epochs=epochs)
        self.generate_latent_z()


    #get the representation
    def get_representation(self, input):
        """
        Returns the latent feature vector from the neural network.
        """
        if self.model_type == "SAC":
            z = self.model.predict(input)
            """
            print("EXTRACTED")
            print(input.shape)
            print(self.model.policy.features_extractor)
            obs_tensor = torch.as_tensor(input, dtype=torch.float32).to(self.model.device)
            features = self.model.actor.extract_features(
                    obs_tensor, self.model.actor.features_extractor
                    )
            z_tensor = self.model.policy.actor.latent_pi(features)
            z = z_tensor.detach().cpu().numpy()
            """
        else:
            z = self.model.predict(input, layer = True)
        z = z.squeeze()

        return z


    def check_dim(self):
        print("prior to sampling, check dim as follows: ")
        if self.output_shape == 1:
            print("sampling from cost model")
        else:
            print("sampling from transition model")
        


    def sample(self, parallelize=False):
        d = self.mu_w[0].shape[0]  # hidden_dim
        beta_s = []
        try:
            
            # Here output denotes the dimension of s' or s_t+1.
            # For each output dimension : self.mu_w[i] :  
            #mu_w[i] , cov_w[i] are the mean of the posterior distribution for the bayesian model
            #This samples from the posterior distribution of weights and the samples are saved as beta_s
            #beta_s[i] represents the weight for the first dim

            for i in range(self.output_shape):
                mus = self.mu_w[i]
                covs = self.cov_w[i][np.newaxis, :, :]
                multivariates = np.random.multivariate_normal(mus, covs[0])
                beta_s.append(multivariates)

        except np.linalg.LinAlgError as e:
            # Sampling could fail if covariance is not positive definite
            print('Details: {} | {}.'.format(e.message, e.args))
            multivariates = np.random.multivariate_normal(np.zeros((d)), np.eye(d))
            beta_s.append(multivariates)
        self.beta_s = np.array(beta_s)

    def predict(self, x):
        # Compute last-layer representation for the current context
        z_context = self.get_representation(x)

        # z_context = z_context[np.newaxis, :]
        
        # Apply Thompson Sampling
        vals = (self.beta_s.dot(z_context.T))
        if self.model_type == "dx":
            state = x[:vals.shape[0]] if len(x.shape) == 1 else x[:, :vals.shape[0]]
            return vals.T + state + self.model.layers[len(self.model.layers)-1].biases.eval(session =self.model.sess).squeeze()[:self.output_shape]+ np.random.normal(loc=0, scale=np.sqrt(self.sigma_n2),size = vals.T.shape)
        if self.model_type == "SAC":
            final_linear = self.model.policy.action_net[-1]
            bias = final_linear.bias[: self.output_shape]  
            noise = torch.randn_like(vals.t()) * (self.sigma_n2 ** 0.5)  
            return vals.t() + bias + noise
        return vals.T + self.model.layers[len(self.model.layers)-1].biases.eval(session =self.model.sess).squeeze()[:self.output_shape]+np.random.normal(loc=0, scale=np.sqrt(self.sigma_n2),size = vals.T.shape)



    def update_bays_reg(self):

        for i in range(self.output_shape):
            
            # Update  posterior with formulas: \beta | z,y ~ N(mu_q, cov_q)
            z = self.latent_z
            if self.model_type == "SAC":
                y = self.train_y[:, i] - self.model.actor.mu[0].bias[i].item()
            else:
                y = self.train_y[:, i] - self.model.layers[len(self.model.layers)-1].biases.eval(session =self.model.sess).squeeze()[i]
            s = np.dot(z.T, z)
            print(s.shape)
            # inv = np.linalg.inv((s/self.sigma_n + 1/self.sigma*self.eye))
            A = s / self.sigma_n2 + 1 / self.sigma2 * self.eye
            B = np.dot(z.T, y) / self.sigma_n2
            reg_coeff = 0

            

            for _ in range(10):
                try:
                    # Compute inv
                    A = A + reg_coeff * self.eye
                    inv = np.linalg.inv(A)
                except Exception as e:
                    # in case computation failed
                    print(e)
                    reg_coeff += 10

                # Store new posterior distributions using inv
                else:
                    self.mu_w[i] = inv.dot(B).squeeze()
                    self.cov_w[i] = inv
                    break
        return self.cov_w

#  : Check Posterior Variance computation
    def compute_posterior_variance(self, new_point):

        #print shape
        new_point = torch.reshape(new_point, (1,-1))

        #get the representation
        z = self.get_representation(new_point)
        z = z.reshape(1,-1)
        
        #compute phi phi trans
        s = np.dot(z.T, z)
        A = s / self.sigma_n2 + 1 / self.sigma2 * self.eye

        #compute inv
        reg_coeff = 0

        for _ in range(10):
            try:
                # Compute inv
                A = A + reg_coeff * self.eye
                inv = np.linalg.inv(A)
            
            except Exception as e:
                # in case computation failed
                print(e)
                reg_coeff += 10
        
        #compute the post var
        # inv = np.linalg.inv(A)   
        # inv_new =   0.5 * (inv + inv.T)
        post_var = np.trace(inv)
        # eig_val, _ = LA.eig(inv)
        # post_var = np.sum(eig_val)
        # print (post_var)
        
        return post_var

    
    
    def thin_data(self, thin_type, thin_samples):
        
        #some condition
        #get the ids
    
        #grad
        nabla_z = []
        nabla_y = []
        reg_y = []
        
        if thin_type == 'ksd' :
            
            #get the x and y first
            for i in range(self.output_shape):
                
                #x and y
                z = self.latent_z
                if self.model_type == "SAC":
                    y = self.train_y[:, i] - self.model.policy.mu.bias[i]
                else:
                    y = self.train_y[:, i] - self.model.layers[len(self.model.layers)-1].biases.eval(session =self.model.sess).squeeze()[i]
 
                #get the w_likelihood
                r1 = np.linalg.inv(np.dot(z.T, z))
                r2 = np.dot(z.T,y)
                w_likelihood =  np.dot(r1, r2)
                
                #gradient computation-----> 200 * 1
                g_y = -2 * (y - np.dot(z, w_likelihood))/ self.sigma_n2
                nabla_y.append(g_y)
                
                #gradient computation for z -----> 200*8
                y = y.reshape(y.shape[0],1)
                w_likelihood = w_likelihood.reshape(w_likelihood.shape[0],1)
                g_z = 2 * ( - np.dot(y, w_likelihood.T) + np.dot(z,w_likelihood)@w_likelihood.T)/ self.sigma_n2
                nabla_z.append(g_z)

                #regression y
                reg_y.append(y)

            
            #get the gradients as np array
            nabla_y_f = np.array(nabla_y).T

            # norm = 1.0 / np.array(nabla_z).shape[0]
            nabla_z_f = np.mean(nabla_z, 0)

            #concat
            # nabla_z_f = np.hstack(nabla_z)
            print ("NABLA Z SHAPE ", nabla_z_f.shape)
            
            grad = np.concatenate((nabla_z_f,nabla_y_f), axis=1)
            reg_y = np.squeeze(np.array(reg_y),2).T
            smpl = np.concatenate((self.latent_z,reg_y ), axis=1)

            #stein thinning
            ids = thin(smpl, grad, thin_samples)
            
        elif thin_type == 'random'  :
            ids = np.random.choice(self.train_x.shape[0], 50, replace=False)      

        #get thinned
        print ('before' + str(self.train_x.shape), str(self.train_y.shape))

        self.train_x = self.train_x[ids]
        self.train_y = self.train_y[ids]
        
        # self.rew = self.rew[ids_f]
        print ('after' + str(self.train_x.shape), str(self.train_y.shape))
        # return check_ksd
    
    def select_samples(pruning_container,new_samples,new_gradients,addition_rule):

        if addition_rule=='std':
            index = 0 
        elif addition_rule=='thin':
            index=-1
        elif addition_rule=='spmcmc':
            index = pruning_container.best_index(candidate_points=new_samples, candidate_gradients=new_gradients)

        return new_samples[index],new_gradients[index]


    def select_samples_d(pruning_container,new_samples,new_gradients,addition_rule):

        if addition_rule=='std':
            index = 0
        elif addition_rule=='thin':
            index=-1
        elif addition_rule=='spmcmc':
            index = pruning_container.best_index_del(candidate_points=new_samples, candidate_gradients=new_gradients)

        return new_samples[index],new_gradients[index]

    
    def thin_data_new(self, thin_type, real = True):
        if real:
            #def thin_data_new(self, thin_type, thin_samples):
            
            #some condition
            #get the ids
        
            #grad
            nabla_z = []
            nabla_y = []
            reg_y = []
            
            if thin_type == 'ksd' :
                
                #get the x and y first
                for i in range(self.output_shape):
                    
                    #x and y
                    z = self.latent_z
                    if self.model_type == "SAC":
                        y = self.train_y[:, i] - self.model.actor.mu[0].bias[i].item()
                    else:
                        y = self.train_y[:, i] - self.model.layers[len(self.model.layers)-1].biases.eval(session =self.model.sess).squeeze()[i]
     
                    #get the w_likelihood
                    r1 = np.linalg.inv(np.dot(z.T, z))
                    r2 = np.dot(z.T,y)
                    w_likelihood =  np.dot(r1, r2)
                    
                    #gradient computation-----> 200 * 1
                    g_y = -2 * (y - np.dot(z, w_likelihood))/ self.sigma_n2
                    nabla_y.append(g_y)
                    
                    #gradient computation for z -----> 200*8
                    y = y.reshape(y.shape[0],1)
                    w_likelihood = w_likelihood.reshape(w_likelihood.shape[0],1)
                    g_z = 2 * ( - np.dot(y, w_likelihood.T) + np.dot(z,w_likelihood)@w_likelihood.T)/ self.sigma_n2
                    nabla_z.append(g_z)

                    #regression y
                    reg_y.append(y)

                
                #get the gradients as np array
                nabla_y_f = np.array(nabla_y).T

                # norm = 1.0 / np.array(nabla_z).shape[0]
                nabla_z_f = np.mean(nabla_z, 0)

                #concat
                # nabla_z_f = np.hstack(nabla_z)
                
                grad = np.concatenate((nabla_z_f,nabla_y_f), axis=1)
                reg_y = np.squeeze(np.array(reg_y),2).T
                smpl = np.concatenate((self.latent_z,reg_y ), axis=1)



                ###########################  New Thinning Method #################################################


                #check ksd value
                # samples = torch.Tensor(smpl)
                # gradients = torch.Tensor(grad)
                samples = smpl
                gradients = grad

                check_ksd = ksd.get_KSD(torch.Tensor(smpl), torch.Tensor(grad), kernel_type = 'rbf', h_method = 'dim')
                print('the ksd is' + str(check_ksd))

                #write : Update pruning container
                kernel_type = 'rbf'
                pruning_container = ksdp.PruningContainer(kernel_type=kernel_type,
                                                  h_method='dim' if kernel_type=='rbf' else None,
                                                  )
                
                #set the device
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                init_sample = torch.tensor(samples[0], dtype=torch.double).to(device)
                init_gradient = torch.tensor(gradients[0], dtype=torch.double).to(device)
                pruning_container.add_point(point=init_sample, gradient=init_gradient)

                #Define the generatpr
                sample_generator = ((torch.tensor(samples[i:i + 10],dtype=torch.double).to(device),
                torch.tensor(gradients[i:i + 10], dtype=torch.double).to(device)) for i in range(0, samples.shape[0], 10))

                #implement new thining
                addition_rule = 'spmcmc'
                prune = [False, None]
                eval_every = 1
                # samples_per_iter = 10
                EPSILON = 0
                pruned_samples = []
                exponent = 1.0

                #Main loop
                for step, (batch_samples, batch_gradients) in enumerate(sample_generator):

        
                    #part 1
                    _, idx = batch_samples.unique_consecutive(dim=0,return_inverse=True)
                    idx = idx.unique()
                    # print (idx)
                    batch_samples = batch_samples[idx]
                    batch_gradients = batch_gradients[idx]
                    
                    #get next
                    if i == 100:
                        pdb.set_trace()
                    #print("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII ", i,"  IIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
                    next_sample, next_gradient = neural_bays_dx_tf.select_samples(pruning_container=pruning_container,
                                                                new_samples=batch_samples,
                                                                new_gradients=batch_gradients,
                                                                addition_rule= addition_rule)
                            
                    
                    #add to cont
                    pruning_container.add_point(point=next_sample, gradient=next_gradient)

                    
                    if exponent>(2.0-1e-10):
                        min_samples = step/2.0

                    else:
                        min_samples = math.sqrt((step**(exponent)) * max(math.log(step + 1.0), 1.0))
                    
                    #implement pruning
                    pruned = pruning_container.retain_to_cutoff(cutoff=EPSILON, min_samples=300)
                    
                    #save the pruned samples
                    pruned_samples.append(pruned)

                
                #clean the pruned samples
                
                pruned_new = [x[0].cpu().numpy()[0].tolist() for x in pruned_samples if x != []]
                # print ('pruned ', len(pruned_new))

                #get the ids of the pruned samples
                ids_pruned = [samples.tolist().index(i) for i in pruned_new]
                #print ('ids pruned ', ids_pruned)

                #total samples
                ids_total = list(np.arange(0,self.train_x.shape[0]))
                print ('ids total ', len(ids_total))

                #get the ids to keep
                ids = [x for x in ids_total if x not in ids_pruned]

                
            elif thin_type == 'random'  :
                ids = np.random.choice(self.train_x.shape[0], 50, replace=False)      



            #get the updated data
            self.train_x = self.train_x[ids]
            self.train_y = self.train_y[ids]
            self.rew = self.rew[ids]
            print ('after' + str(self.train_x.shape), str(self.train_y.shape))
        else:
                
            #def thin_data_new(self, thin_type, thin_samples):

            #some condition
            #get the ids

            #grad
            nabla_z = []
            nabla_y = []
            reg_y = []

            if thin_type == 'ksd' :

                #get the x and y first
                for i in range(self.output_shape):

                    #x and y
                    z = self.latent_z_s
                    y = self.train_y_s[:, i] - self.model.layers[len(self.model.layers)-1].biases.eval(session =self.model.sess).squeeze()[i]

                    #get the w_likelihood
                    r1 = np.linalg.inv(np.dot(z.T, z))
                    r2 = np.dot(z.T,y)
                    w_likelihood =  np.dot(r1, r2)

                    #gradient computation-----> 200 * 1
                    g_y = -2 * (y - np.dot(z, w_likelihood))/ self.sigma_n2
                    nabla_y.append(g_y)

                    #gradient computation for z -----> 200*8
                    y = y.reshape(y.shape[0],1)
                    w_likelihood = w_likelihood.reshape(w_likelihood.shape[0],1)
                    g_z = 2 * ( - np.dot(y, w_likelihood.T) + np.dot(z,w_likelihood)@w_likelihood.T)/ self.sigma_n2
                    nabla_z.append(g_z)

                    #regression y
                    reg_y.append(y)


                #get the gradients as np array
                nabla_y_f = np.array(nabla_y).T

                # norm = 1.0 / np.array(nabla_z).shape[0]
                nabla_z_f = np.mean(nabla_z, 0)

                #concat
                # nabla_z_f = np.hstack(nabla_z)

                grad = np.concatenate((nabla_z_f,nabla_y_f), axis=1)
                reg_y = np.squeeze(np.array(reg_y),2).T
                smpl = np.concatenate((self.latent_z_s,reg_y ), axis=1)



                ###########################  New Thinning Method #################################################


                #check ksd value
                # samples = torch.Tensor(smpl)
                # gradients = torch.Tensor(grad)
                samples = smpl
                gradients = grad

                check_ksd = ksd.get_KSD(torch.Tensor(smpl), torch.Tensor(grad), kernel_type = 'rbf', h_method = 'dim')
                print('the ksd is' + str(check_ksd))

                #write : Update pruning container
                kernel_type = 'rbf'
                pruning_container = ksdp.PruningContainer(kernel_type=kernel_type,
                                                  h_method='dim' if kernel_type=='rbf' else None,
                                                  )

                #set the device
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                init_sample = torch.tensor(samples[0], dtype=torch.double).to(device)
                init_gradient = torch.tensor(gradients[0], dtype=torch.double).to(device)
                pruning_container.add_point(point=init_sample, gradient=init_gradient)

                #Define the generatpr
                sample_generator = ((torch.tensor(samples[i:i + 10],dtype=torch.double).to(device),
                torch.tensor(gradients[i:i + 10], dtype=torch.double).to(device)) for i in range(0, samples.shape[0], 10))

                #implement new thining
                addition_rule = 'spmcmc'
                prune = [False, None]
                eval_every = 1
                # samples_per_iter = 10
                EPSILON = 0
                pruned_samples = []
                exponent = 1.0

                #Main loop
                for step, (batch_samples, batch_gradients) in enumerate(sample_generator):


                    #part 1
                    _, idx = batch_samples.unique_consecutive(dim=0,return_inverse=True)
                    idx = idx.unique()
                    # print (idx)
                    batch_samples = batch_samples[idx]
                    batch_gradients = batch_gradients[idx]

                    #get next
                    next_sample, next_gradient = neural_bays_dx_tf.select_samples(pruning_container=pruning_container,
                                                                new_samples=batch_samples,
                                                                new_gradients=batch_gradients,
                                                                addition_rule= addition_rule)


                    #add to cont
                    pruning_container.add_point(point=next_sample, gradient=next_gradient)


                    if exponent>(2.0-1e-10):
                        min_samples = step/2.0

                    else:
                        min_samples = math.sqrt((step**(exponent)) * max(math.log(step + 1.0), 1.0))

                    #implement pruning
                    pruned = pruning_container.retain_to_cutoff(cutoff=EPSILON, min_samples=300)

                    #save the pruned samples
                    pruned_samples.append(pruned)


                #clean the pruned samples

                pruned_new = [x[0].cpu().numpy()[0].tolist() for x in pruned_samples if x != []]
                # print ('pruned ', len(pruned_new))

                #get the ids of the pruned samples
                ids_pruned = [samples.tolist().index(i) for i in pruned_new]
                #print ('ids pruned ', ids_pruned)

                #total samples
                ids_total = list(np.arange(0,self.train_x_s.shape[0]))
                #print ('ids total ', len(ids_total))

                #get the ids to keep
                ids = [x for x in ids_total if x not in ids_pruned]


            elif thin_type == 'random'  :
                ids = np.random.choice(self.train_x_s.shape[0], 50, replace=False)



            #get the updated data
            self.train_x_s = self.train_x_s[ids]
            self.train_y_s = self.train_y_s[ids]
            self.rew_s = self.rew_s[ids]
            #print ('after' + str(self.train_x_s.shape), str(self.train_y_s.shape))


            
        return ids_pruned
        
    def get_ksd(self, thin_type, real = True):
        if real:
            #def thin_data_new(self, thin_type, thin_samples):

            #some condition
            #get the ids

            #grad
            nabla_z = []
            nabla_y = []
            reg_y = []

            if thin_type == 'ksd' :

                #get the x and y first
                for i in range(self.output_shape):

                    #x and y
                    z = self.latent_z
                    if self.model_type == "SAC":
                        y = self.train_y[:, i] - self.model.actor.mu[0].bias[i].item()
                    else:
                        y = self.train_y[:, i] - self.model.layers[len(self.model.layers)-1].biases.eval(session =self.model.sess).squeeze()[i]

                    #get the w_likelihood
                    r1 = np.linalg.inv(np.dot(z.T, z))
                    r2 = np.dot(z.T,y)
                    w_likelihood =  np.dot(r1, r2)

                    #gradient computation-----> 200 * 1
                    g_y = -2 * (y - np.dot(z, w_likelihood))/ self.sigma_n2
                    nabla_y.append(g_y)

                    #gradient computation for z -----> 200*8
                    y = y.reshape(y.shape[0],1)
                    w_likelihood = w_likelihood.reshape(w_likelihood.shape[0],1)
                    g_z = 2 * ( - np.dot(y, w_likelihood.T) + np.dot(z,w_likelihood)@w_likelihood.T)/ self.sigma_n2
                    nabla_z.append(g_z)

                    #regression y
                    reg_y.append(y)


                #get the gradients as np array
                nabla_y_f = np.array(nabla_y).T

                # norm = 1.0 / np.array(nabla_z).shape[0]
                nabla_z_f = np.mean(nabla_z, 0)

                #concat
                # nabla_z_f = np.hstack(nabla_z)

                grad = np.concatenate((nabla_z_f,nabla_y_f), axis=1)
                reg_y = np.squeeze(np.array(reg_y),2).T
                smpl = np.concatenate((self.latent_z,reg_y ), axis=1)



                ###########################  New Thinning Method #################################################


                #check ksd value
                # samples = torch.Tensor(smpl)
                # gradients = torch.Tensor(grad)
                samples = smpl
                gradients = grad

                check_ksd = ksd.get_KSD(torch.Tensor(smpl), torch.Tensor(grad), kernel_type = 'rbf', h_method = 'dim')
        else:
            #def thin_data_new(self, thin_type, thin_samples):

            #some condition
            #get the ids

            #grad
            nabla_z = []
            nabla_y = []
            reg_y = []

            if thin_type == 'ksd' :

                #get the x and y first
                for i in range(self.output_shape):

                    #x and y
                    z = self.latent_z_s
                    if self.model_type == "SAC":
                        y = self.train_y[:, i] - self.model.actor.mu[0].bias[i].item()
                    else:
                        y = self.train_y_s[:, i] - self.model.layers[len(self.model.layers)-1].biases.eval(session =self.model.sess).squeeze()[i]

                    #get the w_likelihood
                    r1 = np.linalg.inv(np.dot(z.T, z))
                    r2 = np.dot(z.T,y)
                    w_likelihood =  np.dot(r1, r2)

                    #gradient computation-----> 200 * 1
                    g_y = -2 * (y - np.dot(z, w_likelihood))/ self.sigma_n2
                    nabla_y.append(g_y)

                    #gradient computation for z -----> 200*8
                    y = y.reshape(y.shape[0],1)
                    w_likelihood = w_likelihood.reshape(w_likelihood.shape[0],1)
                    g_z = 2 * ( - np.dot(y, w_likelihood.T) + np.dot(z,w_likelihood)@w_likelihood.T)/ self.sigma_n2
                    nabla_z.append(g_z)

                    #regression y
                    reg_y.append(y)


                #get the gradients as np array
                nabla_y_f = np.array(nabla_y).T

                # norm = 1.0 / np.array(nabla_z).shape[0]
                nabla_z_f = np.mean(nabla_z, 0)

                #concat
                # nabla_z_f = np.hstack(nabla_z)

                grad = np.concatenate((nabla_z_f,nabla_y_f), axis=1)
                reg_y = np.squeeze(np.array(reg_y),2).T
                smpl = np.concatenate((self.latent_z_s,reg_y ), axis=1)



                ###########################  New Thinning Method #################################################


                #check ksd value
                # samples = torch.Tensor(smpl)
                # gradients = torch.Tensor(grad)
                samples = smpl
                gradients = grad

                check_ksd = ksd.get_KSD(torch.Tensor(smpl), torch.Tensor(grad), kernel_type = 'rbf', h_method = 'dim')

        return check_ksd






    def selectrn_ggvns_nieuw(self, thin_type):
        #def thin_data_new(self, thin_type, thin_samples):

        #some condition
        #get the ids

        #grad
        nabla_z = []
        nabla_y = []
        reg_y = []

        if thin_type == 'ksd' :

            #get the x and y first
            for i in range(self.output_shape):

                #x and y
                z = self.latent_z
                if self.model_type == "SAC":
                    y = self.train_y[:, i] - self.model.actor.mu[0].bias[i].item()
                else:
                    y = self.train_y[:, i] - self.model.layers[len(self.model.layers)-1].biases.eval(session =self.model.sess).squeeze()[i]

                #get the w_likelihood
                r1 = np.linalg.inv(np.dot(z.T, z))
                r2 = np.dot(z.T,y)
                w_likelihood =  np.dot(r1, r2)

                #gradient computation-----> 200 * 1
                g_y = -2 * (y - np.dot(z, w_likelihood))/ self.sigma_n2
                nabla_y.append(g_y)

                #gradient computation for z -----> 200*8
                y = y.reshape(y.shape[0],1)
                w_likelihood = w_likelihood.reshape(w_likelihood.shape[0],1)
                g_z = 2 * ( - np.dot(y, w_likelihood.T) + np.dot(z,w_likelihood)@w_likelihood.T)/ self.sigma_n2
                nabla_z.append(g_z)

                #regression y
                reg_y.append(y)


            #get the gradients as np array
            nabla_y_f = np.array(nabla_y).T

            # norm = 1.0 / np.array(nabla_z).shape[0]
            nabla_z_f = np.mean(nabla_z, 0)

            #concat
            # nabla_z_f = np.hstack(nabla_z)

            grad = np.concatenate((nabla_z_f,nabla_y_f), axis=1)
            reg_y = np.squeeze(np.array(reg_y),2).T
            smpl = np.concatenate((self.latent_z,reg_y ), axis=1)



            ###########################  New Thinning Method #################################################


            #check ksd value
            # samples = torch.Tensor(smpl)
            # gradients = torch.Tensor(grad)
            samples = smpl
            gradients = grad

            check_ksd = ksd.get_KSD(torch.Tensor(smpl), torch.Tensor(grad), kernel_type = 'rbf', h_method = 'dim')
            print('the ksd is' + str(check_ksd))

            #write : Update pruning container
            kernel_type = 'rbf'
            pruning_container = ksdp.PruningContainer(kernel_type=kernel_type,
                                              h_method='dim' if kernel_type=='rbf' else None,
                                              )

            #set the device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            init_sample = torch.tensor(samples[0], dtype=torch.double).to(device)
            init_gradient = torch.tensor(gradients[0], dtype=torch.double).to(device)
            pruning_container.add_point(point=init_sample, gradient=init_gradient)

            #Define the generatpr
            sample_generator = ((torch.tensor(samples[i:i + 10],dtype=torch.double).to(device),
            torch.tensor(gradients[i:i + 10], dtype=torch.double).to(device)) for i in range(0, samples.shape[0], 10))

            #implement new thining
            addition_rule = 'spmcmc'
            prune = [False, None]
            eval_every = 1
            # samples_per_iter = 10
            EPSILON = 0
            pruned_samples = []
            exponent = 1.0

            #Main loop
            for step, (batch_samples, batch_gradients) in enumerate(sample_generator):


                #part 1
                _, idx = batch_samples.unique_consecutive(dim=0,return_inverse=True)
                idx = idx.unique()
                # print (idx)
                batch_samples = batch_samples[idx]
                batch_gradients = batch_gradients[idx]

                #get next
                next_sample, next_gradient = neural_bays_dx_tf.select_samples(pruning_container=pruning_container,
                                                            new_samples=batch_samples,
                                                            new_gradients=batch_gradients,
                                                            addition_rule= addition_rule)


                #add to cont
                pdb.set_trace()
                pruning_container.add_point(point=next_sample, gradient=next_gradient)


                if exponent>(2.0-1e-10):
                    min_samples = step/2.0

                else:
                    min_samples = math.sqrt((step**(exponent)) * max(math.log(step + 1.0), 1.0))

                #implement pruning
                pruned = pruning_container.prune_to_cutoff(cutoff=EPSILON, min_samples=max(min_samples, 5))

                #save the pruned samples
                pruned_samples.append(pruned)

            for step, (batch_samples, batch_gradients) in enumerate(sample_generator):
                next_sample, next_gradient = neural_bays_dx_tf.select_samples_d(pruning_container=pruning_container,
                                                            new_samples=batch_samples,
                                                            new_gradients=batch_gradients,
                                                            addition_rule= addition_rule)

            #clean the pruned samples

            pruned_new = [x[0].cpu().numpy()[0].tolist() for x in pruned_samples if x != []]
            # print ('pruned ', len(pruned_new))

            #get the ids of the pruned samples
            ids_pruned = [samples.tolist().index(i) for i in pruned_new]
            # print ('ids pruned ', ids_pruned)

            #total samples
            ids_total = list(np.arange(0,self.train_x.shape[0]))
            #print ('ids total ', len(ids_total))

            #get the ids to keep
            ids = [x for x in ids_total if x not in ids_pruned]


        elif thin_type == 'random'  :
            ids = np.random.choice(self.train_x.shape[0], 50, replace=False)



        #get the updated data
        self.train_x = self.train_x[ids]
        self.train_y = self.train_y[ids]
        self.rew = self.rew[ids]
        #print ('after' + str(self.train_x.shape), str(self.train_y.shape))

        return check_ksd




