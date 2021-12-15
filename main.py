import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


#get df
train_df = pd.read_csv("internship_train.csv")

#get x y value
train_x = train_df.copy()
train_y = train_x.pop("target")

#normalize data
x_max = train_x.max()
train_x_norm = train_x / x_max
y_max = train_y.max()
train_y_norm = train_y / y_max

#split on train/valid
train_x_norm, valid_x_norm, train_y_norm, valid_y_norm = train_test_split(train_x_norm, train_y_norm, test_size=0.10, random_state=18)
# create model
model = ExtraTreesRegressor(n_estimators=15,
                            max_depth=5,
                            random_state=18,
                            criterion="squared_error")

# fit model with normalized parameters
model.fit(train_x_norm, train_y_norm)

# predict on validation data
predicted = model.predict(valid_x_norm)
print(f"Accurance of valid data: {rmse(predicted, valid_y_norm)} using rmse metric")

# get test data
hidden_test = pd.read_csv("internship_hidden_test.csv")

# rescale test data
hidden_test = hidden_test/x_max

#predict test
predicted_test = model.predict(hidden_test) * y_max

# create csv with prediction
pd.DataFrame(predicted_test).to_csv("resulted_test.csv")







