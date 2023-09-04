import pysindy as ps
import numpy as np
import matplotlib.pyplot as plt
import argparse 
import pandas as pd
# from sklearn.linear_model import Lasso

parser = argparse.ArgumentParser(description="Data driven ODE system identification for vortex street simulation")
parser.add_argument("--fname", help="Name of the file containing time history of lift and drag values",
                    type=str, default='output_baseline.csv')
parser.add_argument("--slen", help="number of timesteps/length of the signal to be given to the model",
                    type=int, default=250)


filename = parser.parse_args().fname
slen = parser.parse_args().slen

data = pd.read_csv(filename)


t = np.array(data['t'],dtype=float)
CD = np.array(data['CD'], dtype=float)
CL = np.array(data['CL'], dtype=float)


# Signal used as input to SINDy 
idx1 = 10025
idx2 = idx1 + slen

t_train = np.array(data['t'],dtype=float)[idx1:idx2]
CD_train = np.array(data['CD'], dtype=float)[idx1:idx2]
CL_train = np.array(data['CL'], dtype=float)[idx1:idx2]

P_back_train = np.array(data['P_back'], dtype=float)[idx1:idx2]
P1_train = np.array(data['p1'], dtype=float)[idx1:idx2]
P2_train = np.array(data['p2'], dtype=float)[idx1:idx2]



# Original signal upto certain length for Simulation
idx1_test = idx1
idx2_test = idx1_test + 2000

t_test = np.array(data['t'],dtype=float)[idx1_test:idx2_test]
CD_test = np.array(data['CD'], dtype=float)[idx1_test:idx2_test]
CL_test = np.array(data['CL'], dtype=float)[idx1_test:idx2_test]

# P_backtest =  np.array(data['P_back'], dtype=float)[idx1_test:idx2_test]
P1test =  np.array(data['p1'], dtype=float)[idx1_test:idx2_test]
P2test =  np.array(data['p2'], dtype=float)[idx1_test:idx2_test]


#--------------SINDy---------------#


X = np.stack((
              CD_train,
              CL_train,
              # P_back_train,
              P1_train
              # P2_train
              ), 
              axis=-1)


# Differentiation
differentiation_method = ps.SmoothedFiniteDifference()



# Combined Libraries
fourier_library = ps.FourierLibrary()
polynomial_library = ps.PolynomialLibrary(degree=3)
identity_library = ps.IdentityLibrary()

libraries = [           fourier_library,
                        polynomial_library,
                        identity_library * fourier_library,
                        identity_library * polynomial_library
]

feature_library = libraries[0]

# Optimizers
optimizer = ps.STLSQ(threshold=0.05)

model = ps.SINDy(
    differentiation_method=differentiation_method,
    feature_library=feature_library,
    optimizer= optimizer,
    feature_names= ["CD",'CL','P1'],
)


model.fit(X, t=t_train)
model.print()


# SIMULATE
sim = model.simulate([CD_test[650],
                      CL_test[650], 
                      # P_backtest[0], 
                      P1test[650] 
                      # P2test[0]
                      ],
                      t=t_test)


fig = plt.figure(figsize=(9,6))
plt.ylim((3.15,3.28))
plt.plot(t_test, CD_test, color='red', label="Original signal")
plt.plot(t_test[645:], sim[645:,0],"-.b", label="SINDy prediction")
plt.plot(t_train, CD_train, "--k", label="Input signal", linewidth=2.5)

plt.text.usetex = True
plt.legend(loc='upper right',framealpha=1, fontsize=15)


plt.text(50.0,3.27,'Signal length=%i'%slen, bbox=dict(facecolor='white'),fontsize=16)

plt.xlabel("$t/t_c$", fontsize=20)
plt.ylabel("$C_d$", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# fig.savefig('sindy_slen_%i.eps'%slen, format='eps', dpi=1000)

plt.show()



