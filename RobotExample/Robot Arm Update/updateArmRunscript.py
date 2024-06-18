# Import Relevant Libraries

# For Data Processing
import numpy as np
import pandas as pd
from src.FoKL import FoKLRoutines
from Rotations import Arm

model = FoKLRoutines.FoKL()




# For Graphing
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"

df = pd.DataFrame()

# Import Robot Arm Data

u_test = pd.read_csv("u_test.csv", header=None).to_numpy()
u_train = pd.read_csv("u_train.csv", header=None).to_numpy()
y_test = pd.read_csv("y_test.csv", header=None).to_numpy()
y_train = pd.read_csv("y_train.csv", header=None).to_numpy()

# Initialize Arm
l1 = [350, 0, 0]
l2 = [1150, 0, 0]
l3 = [1000,0,0]
l4 = [5000,0,0]

initial_angles = np.array(u_train[:,0])

ArmO = Arm(l1,l2,l3,l4,initial_angles,u_train)

M,V,A = ArmO.simulate()
inputs = [M,V,A]
data = y_train[0,:]
# Perform

# sigsqd0
model.sigsqd0 = 0.009

# a (inverse gamma distribution shape factor for data observation error variance)
model.a = 9

# b (inverse gamma distribution scale factor for data observation error variance)
model.b = 0.01

# atau (inverse gamma distribution shape factor for beta prior variance)
model.atau = 3

# btau (inverse gamma distribution scale factor for beta prior variance)
model.btau = 4000

# tolerance (number of times needed to run Gibbs Sampling with a higher BIC than previous lowest (best) value)
model.tolerance = 3

# Allows for exclusion of term combinations (in this case none)
model.relats_in = []

# draws from the posterior for each tested model
model.draws = 1000

model.gimmie = False
model.aic = False

model.update = True
model.built = False

model.burnin = 0
# Un-normalized and un-formatted 'raw' input variables of dataset:


# Automatically normalize and format:
model.cleanFun(inputs,data)

x = model.inputs
y = model.data

model.inputs = model.inputs[0:500]
model.data = model.data[0:500]

model.fit()

# Use model to predict values until next data engress

avg_betas = np.array(np.mean(model.betas, axis = 0))

prediction= []

mean, bounds, rmse  = model.coverage3(inputs=x, data = y, plot = False)

model.inputs = x[500:700]
model.data = y[500:700]

model.fit()

mean, bounds, rmse  = model.coverage3(inputs=x, data = y, plot = False)

h = 1