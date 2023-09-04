from scipy.optimize import minimize
import flow_solver_act as fs

# Parameters to be optimized (Amplitude and Frequency)
init_params = [1.25, 0.5, 0.5]

# Setting parameters like Reynolds number, time. timesteps...
settings = {'Re':100,'sim_time':100.0,'num_steps':20000,'ave_start_iter':15000}

res = minimize(fs.cost_fct, init_params, args=settings, method='BFGS', options={'disp':True})
# fs.cost_fct(init_params,settings)
print(res.x)

