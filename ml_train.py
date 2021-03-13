import pandas as pd
from scipy import stats 
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

# to save model
import pickle


df = pd.read_csv("training.csv")

# build ML model using scipy LinearRegression
reg = DecisionTreeRegressor(max_depth=12, 
                            min_samples_split=75, 
                            min_samples_leaf=46)
var = df[['util', 'cp', 'bboxArea','bboxAr', 'numPins']]
sc = StandardScaler()
var = sc.fit_transform(var)
label = df['numVias']

reg.fit(var, label)
# print("trained coeff:", reg.coef_)
# print(reg.score(var, label))

# save ML model
fileName = "DecisionTreeRegressor.sav"
pickle.dump(reg, open(fileName, "wb"))
