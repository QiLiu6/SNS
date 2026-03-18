import torch.nn as nn
import torch
from tqdm import tqdm
import torch
        
class ACDM_QG(nn.Module):
    def __init__(self,config,model,silence=True):
        super().__init__()
        self.config=config
        
        self.diffusion_steps = self.config["diffusion_steps"]
        self.silence=silence
        
        # Set noising schedule
        if self.config.get("noise_schedule", "Cosine") == "Cosine":
            betas=self._cosine_variance_schedule(self.diffusion_steps)
        elif self.config.get("noise_schedule") == "Linear":
            beta_start = self.config.get("beta_start", 1e-4 * (500/self.diffusion_steps))
            beta_end   = self.config.get("beta_end", 2e-2 * (500/self.diffusion_steps))
            betas = self._linear_variance_schedule(self.diffusion_steps, beta_start=beta_start, beta_end=beta_end)
        else:
            raise ValueError("Noise Schedule not recognized.")
            
        alphas=1.-betas
        alphas_cumprod=torch.cumprod(alphas,dim=-1)
        self.register_buffer("betas",betas)
        self.register_buffer("alphas",alphas)
        self.register_buffer("alphas_cumprod",alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod",torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",torch.sqrt(1.-alphas_cumprod))
        
        self.model=model

    def forward(self, x, noise_t):
        #x shape: [batch_size, 2, 2, 64 ,64 ]
        x_t = x[:,-1:].squeeze(1)
        x_t_short_lag = x[:,-2:-1].squeeze(1)
        s = torch.randint(0, self.diffusion_steps-1, (x.shape[0],), device=x.device)
        if self.config.get("noise_short_lag", False):
            if self.config.get("different_noise", False):
                noise_short = torch.randn_like(x_t_short_lag, device=x.device)
                x_t_short_lag = self._forward_diffusion(x_t_short_lag, s, noise_short)
            else:
                x_t_short_lag = self._forward_diffusion(x_t_short_lag, s, noise_t)

        x_t_noised = self._forward_diffusion(x_t, s, noise_t)
        x_t_conditionals = torch.cat((x_t_short_lag, x_t_noised), dim=1)
        pred_noise = self.model(x_t_conditionals, None, None, s)
        return pred_noise
            
    @torch.no_grad()
    def denoising(self, x, denoising_timesteps, stop=-1):
        x_t = x[:,-1]
        x_t_short_lag = x[:,0]
        
        noise=torch.randn_like(x_t).to(x_t.device)
        x_t = self._forward_diffusion(x_t, denoising_timesteps, noise)
        
        for i in tqdm(range(denoising_timesteps-1,-1,-1),desc="Denoising",disable=self.silence):
            noise=torch.randn_like(x_t,device=x.device)
            s = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
            if self.config.get('noise_short_lag',False) == True:
                if self.config.get('different_noise') == False:
                    x_t_short_lag_noised = self._forward_diffusion(x_t_short_lag, s, noise)
                else:
                    noise_1 = torch.randn_like(x_t_short_lag).to(x_t.device)
                    x_t_short_lag_noised = self._forward_diffusion(x_t_short_lag, s, noise_1)
            else:
                x_t_short_lag_noised = x_t_short_lag
            x_in = torch.cat((x_t_short_lag_noised, x_t), dim = 1)
            x_t = self._reverse_diffusion(x_in, s, noise)

        x_t = x_t.reshape(x_t.shape[0], 1, 2, 64, 64)
        return x_t
        
    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*torch.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)
        return betas
    def _linear_variance_schedule(self, timesteps, beta_start=1e-4, beta_end=2e-2):
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        return betas

    def _forward_diffusion(self,x_0,t,noise=None):
        if type(t)==int:
            t=t*torch.ones(len(x_0),device=x_0.device,dtype=torch.int64)
        if noise is None:
            noise=torch.randn_like(x_0).to(x_0.device)
        assert x_0.shape==noise.shape
        return self.sqrt_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*x_0+ \
                self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*noise

    @torch.no_grad()
    def _reverse_diffusion(self,x_conditional, s, noise):
        x_t = x_conditional[:,-2:]
        pred = self.model(x_conditional, None, None, s)

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
        return x_t