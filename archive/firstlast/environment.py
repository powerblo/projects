import torch
import numpy as np
from scipy import constants as const

class sphere_env:
    def __init__(self, pos:torch.float64, rad:torch.float64, rad_inf:torch.float64, num_agent:int, device:None):
        self.pos = pos # Position of conducting spherical surface on the central axis
        self.rad = rad # Radius of counducting spherical surface
        self.rad_inf = rad_inf # Radius of infinity surfate, centered at the origin
        self.num_agent = num_agent # Number of random walking agents
        self.device = device
        self.enc_pos = self.spherical_uniform_sample(num_agent=self.num_agent) # Initialize random walk position by uniform random sampling on the infinity surface
        self.inner_pos = torch.ones(size=(self.num_agent, 3), dtype=torch.double, device=self.device) # Initialize sample positions on the inner sphere.
        self.termination = torch.zeros(size=(self.num_agent,), dtype=torch.bool, device=self.device) # Termination flag of random walk. True when inner sample point is specified.
        self.infinity = torch.zeros(size=(self.num_agent,), dtype=torch.bool, device=self.device) # Flag for 'escape to infinity' case. True when the initial sample point is rejected.
        '''
        Allowed cases:
        (1) infinity = True, termination = False: Random walk point escaped to infinity. Requires resetting the whole process
        (2) infinity = False, termination = True: Sample point is determined. Requires terminal position collection and resetting the whole process

        Reset marker:
        (1) OR(infinity, termination): Reset the whole process when True.
        '''
        
    def step(self):
        '''
        For each points in self.enc_pos, check 'escape to infinity' if self.infinity = False. 
        Set self.infinity = True if rejected.
        For points in self.enc_pos[torch.logical_not(self.infinity)], compute distance and direction from the inner sphere center.
        Run bsurf() and obtain the sample position on the inner sphere and then set self.termination = True.
        Transform the coordinates and extract terminal z-coordinates.
        Reset the whole random walk process according to OR(infinity, termination). Update both masks to False.
        
        RETURNS
        termination_num = int
        infinity_num = int
        terminal_z: torch.DoubleTensor(termination_num,)
        '''
        infinity_num = self.infinity_check()
        termination_num = self.bsurf()
        terminal_z = self.inner_pos[self.termination,2].clone()
        self.reset()
        return termination_num, infinity_num, terminal_z


    def reset(self):
        '''
        Resets the random walk when in hits the inner sphere or escape to infinity.
        Reset to uniform random position on the infinity sphere at all cases.
        '''
        mask = torch.logical_or(self.termination, self.infinity)
        num_reset = torch.sum(mask.to(torch.int)).item()
        self.enc_pos[mask] = self.spherical_uniform_sample(num_agent=num_reset) # Initialize random walk position by uniform random sampling on the infinity surface
        self.inner_pos[mask] = torch.ones(size=(num_reset,3), dtype=torch.double, device=self.device) # Initialize the sample positions
        self.termination[mask] = torch.zeros(size=(num_reset,), dtype=torch.bool, device=self.device) # Termination flag of random walk
        self.infinity[mask] = torch.zeros(size=(num_reset,), dtype=torch.bool, device=self.device) # Flag for 'escape to infinity' case

    def bsurf(self):
        '''
        Determine the sample position on the inner sphere, in cartesian coordinates centered on the inner sphere.
        RETURNS
        num_hit: Single integer. Number of terminated random walk point
        '''
        termination_mask = torch.logical_not(self.infinity)
        r = self.compute_dist(termination_mask)
        num_hit = torch.sum(termination_mask.to(torch.int)).item()
        ratio = self.rad/r
        p2 = torch.rand(size=(num_hit,), dtype=torch.double, device=self.device)
        costh = -(1-ratio)**2 + 2*(1-ratio)*(1+ratio**2)*p2 + 2*ratio*(1+ratio**2)*p2**2
        costh = costh/(1-ratio+2*ratio*p2)**2
        sinth = torch.sqrt(torch.abs(1-costh**2))

        phi = 2*const.pi*torch.rand(size=(num_hit,), dtype=torch.double, device=self.device)
        cosph = torch.cos(phi)
        sinph = torch.sin(phi)
        x_old = self.enc_pos[termination_mask,0]
        y_old = self.enc_pos[termination_mask,1]
        z_old = self.enc_pos[termination_mask,2] - self.pos # Center the coordinate on the inner sphere
        p = torch.sqrt(x_old**2 + y_old**2)
        x = self.rad*(sinth*cosph*x_old*z_old/(p*r) - sinth*sinph*y_old/p + costh*x_old/r)
        y = self.rad*(sinth*cosph*y_old*z_old/(p*r) + sinth*sinph*x_old/p + costh*y_old/r)
        z = self.rad*(-sinth*cosph*p/r + costh*z_old/r)
        self.termination[termination_mask] = torch.ones(size=(num_hit,), dtype = torch.bool, device=self.device)
        self.inner_pos[termination_mask,0] = x
        self.inner_pos[termination_mask,1] = y
        self.inner_pos[termination_mask,2] = z
        return num_hit

    def infinity_check(self):
        '''
        Checks if the agent escaped to the infinity
        RETURNS
        Single integer, number of escaped agents
        '''
        inf_mask = torch.logical_not(self.infinity)
        r = self.compute_dist(inf_mask)
        # ratio = self.rad/r # Use this line for first-passage case
        ratio = torch.ones_like(r) # Use this line for last-passage case
        mask = torch.rand_like(ratio, dtype=torch.double, device=self.device)
        self.infinity[inf_mask] = torch.le(ratio, mask)
        return torch.sum(self.infinity[inf_mask].to(torch.int)).item()

    def compute_dist(self, mask):
        '''
        Computes distance from sample points on the enclosing sphere to the conducting surface center
        INPUT
        mask=torch.Tensor(num_agent, dtype=torch.bool)
        RETURNS
        torch.DoubleTensor(num_agent,)
        '''
        disp = self.enc_pos[mask].clone()
        disp[:,2] -= self.pos
        dist = torch.linalg.norm(disp, dim=1).flatten()
        return dist

    def spherical_uniform_sample(self, num_agent:int):
        '''
        Uniform sampling on the enclosing sphere in spherical coordinates (theta, phi)
        RETURNS
        torch.DoubleTensor(num_agent, 3) with uniformly sampled positions in cartesian coordinates 
        '''
        z = torch.clamp(1 - 2*torch.rand(size=(num_agent,), dtype=torch.float64, device=self.device), min=-1, max=1)
        phi = 2*const.pi*torch.rand(size=(num_agent,), dtype=torch.float64, device=self.device)
        sin_th = torch.sqrt(1-z**2)
        out = self.rad_inf*torch.stack((sin_th*torch.cos(phi), sin_th*torch.sin(phi), z), dim=1)
        return out