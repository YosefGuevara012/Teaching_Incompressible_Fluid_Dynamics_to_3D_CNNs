from get_param import params,get_hyperparam
import matplotlib.pyplot as plt
from Logger import Logger,t_step
from pde_cnn import get_Net
import torch
import numpy as np
from setups import Dataset
import derivatives as d
from derivatives import vector2HSV,rot_mac,toCuda,toCpu
from torch.optim import Adam
import cv2
import math
import numpy as np
import time
import os
from datetime import datetime
from numpy2vtk import imageToVTK

torch.manual_seed(1)
torch.set_num_threads(4)
np.random.seed(2)


# initialize logger
logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=False)

# initialize fluid model
pde_cnn = toCuda(get_Net(params))

# load fluid model
date_time,index = logger.load_state(pde_cnn,None,datetime=params.load_date_time,index=params.load_index)
pde_cnn.eval()

print(f"loaded date_time: {date_time}; index: {index}")
model_parameters = filter(lambda p: p.requires_grad, pde_cnn.parameters())
model_parameters = sum([np.prod(p.size()) for p in model_parameters])
print(f"n_model_parameters: {model_parameters}")

# plot color legend
cv2.namedWindow('legend',cv2.WINDOW_NORMAL)
vector = toCuda(torch.cat([torch.arange(-1,1,0.01).unsqueeze(0).unsqueeze(2).repeat(1,1,200),torch.arange(-1,1,0.01).unsqueeze(0).unsqueeze(1).repeat(1,200,1)]))
image = vector2HSV(vector)
image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
cv2.imshow('legend',image)

# initialize windows for averaged velocity / pressure fields
cv2.namedWindow('v xy',cv2.WINDOW_NORMAL)
cv2.namedWindow('v xz',cv2.WINDOW_NORMAL)
cv2.namedWindow('v yz',cv2.WINDOW_NORMAL)
cv2.namedWindow('p xy',cv2.WINDOW_NORMAL)
cv2.namedWindow('p xz',cv2.WINDOW_NORMAL)
cv2.namedWindow('p yz',cv2.WINDOW_NORMAL)

def mousePosition_xy(event,x,y,flags,param):
	global dataset
	if (event==cv2.EVENT_MOUSEMOVE or event==cv2.EVENT_LBUTTONDOWN) and flags==1:
		dataset.mousex = y
		dataset.mousey = x

def mousePosition_xz(event,x,y,flags,param):
	global dataset
	if (event==cv2.EVENT_MOUSEMOVE or event==cv2.EVENT_LBUTTONDOWN) and flags==1:
		dataset.mousex = y
		dataset.mousez = x

def mousePosition_yz(event,x,y,flags,param):
	global dataset
	if (event==cv2.EVENT_MOUSEMOVE or event==cv2.EVENT_LBUTTONDOWN) and flags==1:
		dataset.mousey = y
		dataset.mousez = x

cv2.setMouseCallback("v xy",mousePosition_xy)
cv2.setMouseCallback("v xz",mousePosition_xz)
cv2.setMouseCallback("v yz",mousePosition_yz)
cv2.setMouseCallback("p xy",mousePosition_xy)
cv2.setMouseCallback("p xz",mousePosition_xz)
cv2.setMouseCallback("p yz",mousePosition_yz)


last_FPS = 0
quit = False
vtk_iteration = 0
p_pressed = False
recording = False
animation_iteration = 0.0
w_pressed = False
animating = False # animate mu / rho
animation_freq = 45
animation_type = "sin"#"triangle"#
aggregator = "mean"# "max"#

#My variables

abs_divergence = []
sq_divergence = []
Ld_list = []
Lp_list = []

eps = 0.00000001

def compute_Ld(v_new, v_cond, cond_mask_mac):
    # Crop v_cond and cond_mask to match v_new's size
    v_cond = v_cond[:, :, :v_new.shape[2], :v_new.shape[3], :v_new.shape[4]]
    cond_mask_mac = cond_mask_mac[:, :, :v_new.shape[2], :v_new.shape[3], :v_new.shape[4]]

    # Compute boundary loss
    boundary_loss = torch.mean((cond_mask_mac * (v_new - v_cond))**2)
    return boundary_loss


def loss_function(x):
	if params.loss=="square":
		return torch.pow(x,2)
	if params.loss=="exp_square":
		x = torch.pow(x,2)
		return torch.exp(x/torch.max(x).detach()*5)
	if params.loss=="abs":
		return torch.abs(x)
	if params.loss=="log_square":
		return torch.log(torch.pow(x,2)+eps)


def compute_Lp(v_new, v_old, p_new, mu, rho, dt, flow_mask_mac):
	
    # Time-stepping with IMEX: implicit-explicit scheme
    v_old = v_old[:, :, :v_new.shape[2], :v_new.shape[3], :v_new.shape[4]]
    p_new = p_new[:, :, :v_new.shape[2], :v_new.shape[3], :v_new.shape[4]]
    flow_mask_mac = flow_mask_mac[:, :, :v_new.shape[2], :v_new.shape[3], :v_new.shape[4]]
    # print(flow_mask_mac.shape)

    v = (v_new + v_old) / 2

    loss_nav =  torch.mean(loss_function(flow_mask_mac[:,0:1]*(rho*((v_new[:,0:1]-v_old[:,0:1])/params.dt+v[:,0:1]*d.dx(v[:,0:1])+0.5*(d.map_vy2vx_p(v[:,1:2])*d.dy_p(v[:,0:1])+d.map_vy2vx_m(v[:,1:2])*d.dy_m(v[:,0:1]))+0.5*(d.map_vz2vx_p(v[:,2:3])*d.dz_p(v[:,0:1])+d.map_vz2vx_m(v[:,2:3])*d.dz_m(v[:,0:1])))+d.dx_m(p_new)-mu*d.laplace(v[:,0:1])))[:,:,1:-1,1:-1,1:-1],dim=(1,2,3,4))+\
					torch.mean(loss_function(flow_mask_mac[:,1:2]*(rho*((v_new[:,1:2]-v_old[:,1:2])/params.dt+v[:,1:2]*d.dy(v[:,1:2])+0.5*(d.map_vx2vy_p(v[:,0:1])*d.dx_p(v[:,1:2])+d.map_vx2vy_m(v[:,0:1])*d.dx_m(v[:,1:2]))+0.5*(d.map_vz2vy_p(v[:,2:3])*d.dz_p(v[:,1:2])+d.map_vz2vy_m(v[:,2:3])*d.dz_m(v[:,1:2])))+d.dy_m(p_new)-mu*d.laplace(v[:,1:2])))[:,:,1:-1,1:-1,1:-1],dim=(1,2,3,4))+\
					torch.mean(loss_function(flow_mask_mac[:,2:3]*(rho*((v_new[:,2:3]-v_old[:,2:3])/params.dt+v[:,2:3]*d.dz(v[:,2:3])+0.5*(d.map_vx2vz_p(v[:,0:1])*d.dx_p(v[:,2:3])+d.map_vx2vz_m(v[:,0:1])*d.dx_m(v[:,2:3]))+0.5*(d.map_vy2vz_p(v[:,1:2])*d.dy_p(v[:,2:3])+d.map_vy2vz_m(v[:,1:2])*d.dy_m(v[:,2:3])))+d.dz_m(p_new)-mu*d.laplace(v[:,2:3])))[:,:,1:-1,1:-1,1:-1],dim=(1,2,3,4))

    loss_nav_value = loss_nav.item()
    loss_nav_value= "{:.5f}".format(loss_nav_value * 1e-2)  

    return loss_nav_value





with torch.no_grad():
		while True:
			
			# initialize dataset
			dataset = Dataset(params.width,params.height,params.depth,1,1,interactive=True,average_sequence_length=params.average_sequence_length,max_speed=params.max_speed,dt=params.dt,
						types=["image"],images=["cube"],mu_range=[params.mu_min,params.mu_max],rho_range=[params.rho_min,params.rho_max])
			# options for setup types: "magnus_y","magnus_z","no_box","rod_y","rod_z","moving_rod_y","moving_rod_z","box","benchmark","image","ball"
			# options for images: "submarine","fish","cyber","wing","2_objects","3_objects"
			
			FPS=0
			last_time = time.time()
			
			# Start: fluid simulation loop
			
			for t in range(params.average_sequence_length):
				
				# get dirichlet boundary conditions, fluid domain, vector potential (streamfunction), pressure field, mu, rho from dataset:
				v_cond,cond_mask,a_old,p_old,mu,rho = toCuda(dataset.ask())
				
				v_cond = d.normal2staggered(v_cond) # map dirichlet boundary conditions onto staggered grid
				
				a_new,p_new = pde_cnn(a_old,p_old,v_cond,cond_mask,mu,rho) # apply fluid model on fluid state and boundary conditions
				
				dataset.tell(toCpu(a_new),toCpu(p_new)) # update dataset with new predicted fluid state
				
				# End: fluid simulation loop
				
				# Visualization Code:
				if t%1==0:
					print(f"t:{t}")
					
					cond_mask_mac = (d.normal2staggered(cond_mask.repeat(1,3,1,1,1))==1).float()
					flow_mask_mac = 1-cond_mask_mac
					
					# show velocity field
					
					v_new = d.staggered2normal(flow_mask_mac*d.rot_mac(a_new)+cond_mask_mac*v_cond)[:,:,1:-1,1:-1,1:-1]
					
					if aggregator == "mean":
						vector = (v_new[0])[(0,1),].mean(3).clone()
					elif aggregator == "max":
						vector = torch.max((v_new[0])[(0,1),],dim=3)[0].clone()
					image = vector2HSV(vector)
					image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
					cv2.imshow('v xy',image)
					
					if aggregator == "mean":
						vector = (v_new[0])[(0,2),].mean(2).clone()
					if aggregator == "max":
						vector = torch.max((v_new[0])[(0,2),],dim=2)[0].clone()
					image = vector2HSV(vector)
					image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
					cv2.imshow('v xz',image)
					
					if aggregator == "mean":
						vector = (v_new[0])[(1,2),].mean(1).clone()
					if aggregator == "max":
						vector = torch.max((v_new[0])[(1,2),],dim=1)[0].clone()
					image = vector2HSV(vector)
					image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
					cv2.imshow('v yz',image)
					
					# show pressure field
					
					if aggregator == "mean":
						p = (p_new[0,0]*(1-cond_mask[0,0])).mean(2).clone()
					elif aggregator == "max":
						p = torch.max((p_new[0,0]*(1-cond_mask[0,0])),dim=2)[0].clone()
					p = p-torch.min(p)
					p = p/torch.max(p)
					p = toCpu(p).unsqueeze(2).repeat(1,1,3).numpy()
					cv2.imshow('p xy',p)
					
					if aggregator == "mean":
						p = (p_new[0,0]*(1-cond_mask[0,0])).mean(1).clone()
					elif aggregator == "max":
						p = torch.max((p_new[0,0]*(1-cond_mask[0,0])),dim=1)[0].clone()
					p = p-torch.min(p)
					p = p/torch.max(p)
					p = toCpu(p).unsqueeze(2).repeat(1,1,3).numpy()
					cv2.imshow('p xz',p)
					
					if aggregator == "mean":
						p = (p_new[0,0]*(1-cond_mask[0,0])).mean(0).clone()
					elif aggregator == "max":
						p = torch.max((p_new[0,0]*(1-cond_mask[0,0])),dim=0)[0].clone()
					p = p-torch.min(p)
					p = p/torch.max(p)
					p = toCpu(p).unsqueeze(2).repeat(1,1,3).numpy()
					cv2.imshow('p yz',p)
					
					divergence_v = d.div(flow_mask_mac*d.rot_mac(a_new)+cond_mask_mac*v_cond)[:,:,1:-1,1:-1,1:-1]
					
					#My variables
					v_old = d.rot_mac(a_old)
					# print(v_new.shape)
					# v_old = v_old[:, :, :v_new.shape[2], :v_new.shape[3], :v_new.shape[4]]
					# print(v_old.shape)

					Ld = compute_Ld(v_new, v_cond, cond_mask_mac)
					Lp = compute_Lp(v_new, v_old, p_new, mu, rho, params.dt, flow_mask_mac)

					abs_divergence.append(torch.mean(torch.abs(divergence_v)))
					sq_divergence.append(torch.mean(divergence_v**2))
			
					

					Ld_list.append(Ld)
					Lp_list.append(Lp)


					print(f"FPS: {last_FPS}; E[|div(v)|] = {torch.mean(torch.abs(divergence_v))}; E[div(v)^2] = {torch.mean(divergence_v**2)}; Ld = {Ld}; Lp = {Lp}")
					print(f"mu: {dataset.mousemu.numpy()[0,0,0,0]}; rho: {dataset.mouserho.numpy()[0,0,0,0]}; v: {dataset.mousev}")
					
					key = cv2.waitKey(1)
					
					if key==ord('x'):
						dataset.mousev+=0.1
					elif key==ord('y'):
						dataset.mousev-=0.1
					
					if key==ord('s'):
						dataset.mousew+=0.1
					elif key==ord('a'):
						dataset.mousew-=0.1
					
					if key==ord('f'):
						dataset.mousemu*=1.05
					elif key==ord('d'):
						dataset.mousemu/=1.05
					
					if key==ord('v'):
						dataset.mouserho*=1.05
					elif key==ord('c'):
						dataset.mouserho/=1.05
						
					elif key==ord('1'): # Re: 64 time reversible flow
						dataset.mousemu= torch.tensor([[[[5]]]])
						dataset.mouserho= torch.tensor([[[[0.2]]]])
						dataset.mousev=-1
					elif key==ord('2'): 
						dataset.mousemu=torch.tensor([[[[0.5]]]])
						dataset.mouserho=torch.tensor([[[[1]]]])
						dataset.mousev=-1
					elif key==ord('3'): # Re: 80 Laminar Flow
						dataset.mousemu= torch.tensor([[[[0.2]]]])
						dataset.mouserho=torch.tensor([[[[1]]]])
						dataset.mousev=-1
					elif key==ord('4'): # Re: 800 Turbulent flow
						dataset.mousemu=torch.tensor([[[[0.1]]]])
						dataset.mouserho=torch.tensor([[[[5]]]])
						dataset.mousev=-1
					elif key==ord('5'):
						dataset.mousemu=torch.tensor([[[[0.02]]]])
						dataset.mouserho=torch.tensor([[[[10]]]])
						dataset.mousev=-1
					
					if key==ord('r'):
						if dataset.env_info[0]["type"] == "image":
							dataset.mousex=96
							dataset.mousey=32
							dataset.mousez=32
							dataset.mousev=-1
							dataset.mousew=0
						
						if dataset.env_info[0]["type"] == "magnus_y":
							dataset.mousex=100
							dataset.mousey=32
							dataset.mousez=32
							dataset.mousev=-1
							dataset.mousew=1
					
					# animate mu / rho (for movie)
					if key==ord('w') or animating:
						animation_time = animation_iteration/15
						# triangle animation (value between 0 and 1)
						if animation_type == "triangle":
							value = animation_time % animation_freq
							if value>animation_freq/2:
								value = animation_freq-value
							value /= animation_freq/2
						elif animation_type == "sin":
							value = np.sin(animation_time/animation_freq*2*np.pi)/2+0.5
						
						dataset.mousemu = np.exp(value*(np.log(5)-np.log(0.01))+np.log(0.01))
						dataset.mouserho = np.exp(value*(np.log(0.2)-np.log(8))+np.log(8))
						
						animation_iteration += 1
						if not w_pressed and key==ord('w'):
							animating = not animating
						w_pressed = True
					if key != ord('w'):
						w_pressed = False
					
					# print to VTK
					if key==ord('p') or recording:
						name = dataset.env_info[0]["type"]
						if name=="image":
							name = name+"_"+dataset.env_info[0]["image"]
						os.makedirs(f"vtk/{name}/{get_hyperparam(params)}",exist_ok=True)
						
						pressure = toCpu((p_new[0,0]*(1-cond_mask[0,0])).clone()).numpy()
						imageToVTK(f"vtk/{name}/{get_hyperparam(params)}/pressure.{vtk_iteration}",pointData={"pressure":pressure})
						v_new = toCpu(d.staggered2normal(flow_mask_mac*d.rot_mac(a_new)+cond_mask_mac*v_cond)[:,:,1:-1,1:-1,1:-1])
						imageToVTK(f"vtk/{name}/{get_hyperparam(params)}/velocity.{vtk_iteration}",cellData={"velocity":(v_new[0,0].numpy(),v_new[0,1].numpy(),v_new[0,2].numpy())})
						boundary_object = cond_mask[0,0].clone()
						imageToVTK(f"vtk/{name}/{get_hyperparam(params)}/boundary.{vtk_iteration}",pointData={"boundary":toCpu(1-boundary_object).numpy()})
						print(f"saved fluid state to vtk-files in: vtk/{name}/{get_hyperparam(params)} ({vtk_iteration})")
						vtk_iteration += 1
						if not p_pressed and key==ord('p'):
							recording = not recording
						p_pressed = True
					if key != ord('p'):
						p_pressed = False
					
					if key==ord('n'):
						break
					
					if key==ord('q'):
						quit=True
						break
					
					FPS += 1
					if time.time()-last_time>=1:
						last_time = time.time()
						last_FPS=FPS
						FPS = 0
			
			if quit:
				if len(abs_divergence) == len(sq_divergence) == len(Ld_list) == len(Lp_list):
					with open('log_output.txt', 'w') as file:
						file.write("abs_divergence sq_divergence Ld Lp\n")
						for abs_divergence, sq_divergence, Ld_list , Lp_list in zip(abs_divergence, sq_divergence, Ld_list, Lp_list):
							file.write(f'{abs_divergence} {sq_divergence} {Ld_list} {Lp_list}\n')
				else:
					print("The divergence lists are not of equal length")
				break


cv2.destroy_all_windows()
