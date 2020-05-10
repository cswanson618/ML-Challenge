from joblib import load
import pandas as pd

# note all training was done in the jupyter notebook "training.ipynb." It also contains my process for getting to my model.

def predict_from_csv(path_to_csv):

    df = pd.read_csv(path_to_csv)
    df = df.assign(length_in_inches = (df["Length3"]*0.39370))
    df = df.assign(weightfactor = (df["length_in_inches"]*(df["length_in_inches"]*.58)*(df["length_in_inches"]*.58)/900))
    X = df[["length_in_inches", "weightfactor"]].values
    y = df[["Weight"]].values.ravel()
    reg = load("reg.joblib")

    predictions = reg.predict(X)

    return predictions

if __name__ == "__main__":
    predictions = predict_from_csv("fish_holdout_demo.csv")
    print(predictions)
######

# ### WE WRITE THIS ###
    from sklearn.metrics import mean_squared_error, r2_score
    ho_predictions = predict_from_csv("fish_holdout_demo.csv")
    ho_truth = pd.read_csv("fish_holdout_demo.csv")["Weight"].values
    ho_mse = mean_squared_error(ho_truth, ho_predictions)
    rsqu = r2_score(ho_truth, ho_predictions)
    print(ho_mse)
    print(rsqu)
# ######

