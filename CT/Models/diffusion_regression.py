import torch.nn as nn
import torch
from tqdm import tqdm
import torch
        
class Diffusion_regression(nn.Module):
    def __init__(self,config,model,silence=True):
        super().__init__()
        self.config=config
        
        self.diffusion_steps = self.config["diffusion_steps"]
            
        self.in_channels = self.config["input_channels"]
        self.image_width = self.config["image_width"]
        self.image_height = self.config["image_height"]

        self.silence=silence
        
        betas=self._cosine_variance_schedule(self.diffusion_steps)
        alphas=1.-betas
        alphas_cumprod=torch.cumprod(alphas,dim=-1)
        self.register_buffer("betas",betas)
        self.register_buffer("alphas",alphas)
        self.register_buffer("alphas_cumprod",alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod",torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",torch.sqrt(1.-alphas_cumprod))
        
        self.model=model

    def forward(self, x, noise_t, short_lag, long_lag):
        x_t = x[:,-1:]
        x_t_short_lag = x[:,-short_lag-1:-short_lag,:,:]
        x_t_long_lag = x[:,-long_lag-1:-long_lag,:,:]
        s = torch.randint(0,self.diffusion_steps-1,(x.shape[0],)).to(x.device)
        short_lag = torch.full((x.shape[0],), short_lag, dtype=torch.int64, device=x.device)
        if self.config.get('noise_short_lag') == True:
            if self.config["different_noise"] == False:
                x_t_short_lag = self._forward_diffusion(x_t_short_lag, s, noise_t)
            else:
                ranges = self.diffusion_steps - s
                s_prime = torch.floor(torch.rand_like(s.float()) * ranges.float()).long() + s
                noise_prime = torch.randn((x.shape[0],1,x.shape[2],x.shape[3]),device = x.device)
                x_t_short_lag = self._forward_diffusion(x_t_short_lag, s_prime, noise_prime)
        long_lag = torch.full((x.shape[0],), long_lag, dtype=torch.int64, device=x.device)
        
        x_t_noised = self._forward_diffusion(x_t, s, noise_t)

        x_noised_plus_x_t_conditionals = torch.cat((x_t_noised, x_t_short_lag, x_t_long_lag), dim=1)
        pred_noise, pred_s = self.model(x_noised_plus_x_t_conditionals, short_lag, long_lag, None, True)
            
        return pred_noise, pred_s, s
            
    @torch.no_grad()
    def denoising(self, x, short_lag, long_lag, denoising_timesteps, stop=-1, forward_diff=False):
        x_t = x[:,0:1]
        x_t_short_lag = x[:,1:2]
        x_t_long_lag = x[:,2:3]

        start_step=max(denoising_timesteps)
        
        if forward_diff:
            noise=torch.randn_like(x_t).to(x_t.device)
            x_t = self._forward_diffusion(x_t,denoising_timesteps,noise)
            if self.config.get('noise_short_lag') == True:
                if self.config.get('different_noise') == False:
                    x_t_short_lag = self._forward_diffusion(x_t_short_lag, denoising_timesteps, noise)
                else:
                    noise_1 = torch.randn_like(x_t).to(x_t.device)
                    x_t_short_lag = self._forward_diffusion(x_t_short_lag, denoising_timesteps, noise_1)

        x = torch.cat((x_t,x_t_short_lag,x_t_long_lag), dim = 1)

        for i in tqdm(range(start_step-1,stop,-1),desc="Denoising",disable=self.silence):
            selected=denoising_timesteps>=i
            selected_images=x[selected]
            noise=torch.randn_like(selected_images[:,0:1],device=x.device)
            s = torch.tensor([i for _ in range(len(selected_images))]).to(x.device)
            short_lag_selected = short_lag[selected]
            long_lag_selected = long_lag[selected]
            denoised_conditional = self._reverse_diffusion(selected_images, short_lag_selected, long_lag_selected, s, noise)

            inc=0
            for xx,sel in enumerate(selected):
                if sel:
                    x[xx]= denoised_conditional[inc]
                    inc+=1
        return x[:,0:1]
        
    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*torch.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas

    def _forward_diffusion(self,x_0,t,noise=None):
        """ Run forward diffusion process, i.e. add noise to some input images
        x_0:    input tensors to add noise to
        t:      noise level to add. Can be either a tensor with same length x_0, in which case
                each image can be noised differently. Or just pass a scalar, and the same level of noise
                will be added to each image
        noise:  Tensor of random noise. Can be None, in which case we will generate noise here
        
        returns a tensor of the same shape x_0, where each image has been noised """

        ## If t is just an int, create a tensor for the forward process
        if type(t)==int:
            t=t*torch.ones(len(x_0),device=x_0.device,dtype=torch.int64)

        if noise==None:
            noise=torch.randn_like(x_0).to(x_0.device)

        assert x_0.shape==noise.shape
        #q(x_{t}|x_{0})
        return self.sqrt_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*x_0+ \
                self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*noise

    @torch.no_grad()
    def _reverse_diffusion(self,x_conditional, short_lag, long_lag, s, noise):
        if self.config.get('noise_short_lag') == True and self.config.get('different_noise') == False:
            x_t = x_conditional[:,0:1]
            x_t_minus_short_lag = x_conditional[:,1:2]
            x_t_minus_long_lag = x_conditional[:,2:3]
            pred = self.model(x_conditional, short_lag, long_lag, None, False)
    
            alpha_s = self.alphas.gather(-1,s).reshape(x_t.shape[0],1,1,1)
            alpha_s_cumprod = self.alphas_cumprod.gather(-1,s).reshape(x_t.shape[0],1,1,1)
            beta_s = self.betas.gather(-1,s).reshape(x_t.shape[0],1,1,1)
            sqrt_one_minus_alpha_cumprod_s = self.sqrt_one_minus_alphas_cumprod.gather(-1,s).reshape(x_t.shape[0],1,1,1)
            mean_t = (1./torch.sqrt(alpha_s))*(x_t-((1.0-alpha_s)/sqrt_one_minus_alpha_cumprod_s)*pred)
            mean_t_minus_short_lag = (1./torch.sqrt(alpha_s))*(x_t_minus_short_lag-((1.0-alpha_s)/sqrt_one_minus_alpha_cumprod_s)*pred)
    
            if s.min()>0:
                alpha_s_cumprod_prev = self.alphas_cumprod.gather(-1,s-1).reshape(x_t.shape[0],1,1,1)
                std = torch.sqrt(beta_s*(1.-alpha_s_cumprod_prev)/(1.-alpha_s_cumprod))
            else:
                std = 0.0
            x_t = mean_t + std * noise
            x_t_minus_short_lag = mean_t_minus_short_lag + std * noise
            x_conditional = torch.cat((x_t,x_t_minus_short_lag, x_t_minus_long_lag), dim = 1)
            return x_conditional
        else:
            x_t = x_conditional[:,0:1]
            x_t_minus_short_lag = x_conditional[:,1:2]
            x_t_minus_long_lag = x_conditional[:,2:3]
            pred = self.model(x_conditional,short_lag, long_lag, None, False)
    
            alpha_s=self.alphas.gather(-1,s).reshape(x_t.shape[0],1,1,1)
            alpha_s_cumprod=self.alphas_cumprod.gather(-1,s).reshape(x_t.shape[0],1,1,1)
            beta_s=self.betas.gather(-1,s).reshape(x_t.shape[0],1,1,1)
            sqrt_one_minus_alpha_cumprod_s=self.sqrt_one_minus_alphas_cumprod.gather(-1,s).reshape(x_t.shape[0],1,1,1)
            mean_t=(1./torch.sqrt(alpha_s))*(x_t-((1.0-alpha_s)/sqrt_one_minus_alpha_cumprod_s)*pred)
    
            if s.min()>0:
                alpha_s_cumprod_prev=self.alphas_cumprod.gather(-1,s-1).reshape(x_t.shape[0],1,1,1)
                std=torch.sqrt(beta_s*(1.-alpha_s_cumprod_prev)/(1.-alpha_s_cumprod))
            else:
                std=0.0
            x_t = mean_t + std * noise
            x_conditional = torch.cat((x_t,x_t_minus_short_lag, x_t_minus_long_lag), dim = 1)
            return x_conditional