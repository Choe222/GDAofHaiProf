import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class MyData(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        mapping_dict = {"High Risk": 2, "Moderate Risk": 1, "Low Risk" : 0}
        df_input = dataframe.iloc[:, 1:21].to_numpy() 
        #df_disruption = dataframe['disruption_likelihood_score'] #Regression/Ranking
        #df_delay = dataframe['delay_probability']   #Regression/Probability Calibration
        df_risk = dataframe['risk_classification'].map(mapping_dict) #Classificasion
        df_delivery = dataframe['delivery_time_deviation']  #Regression
        
        class_indices = torch.tensor(df_risk.values, dtype=torch.long)
        self.risk_labels = torch.nn.functional.one_hot(class_indices, num_classes=3).float()
        #self.disruption_labels = torch.tensor(df_disruption, dtype=torch.float)
        #self.delay_labels = torch.tensor(df_delay, dtype=torch.float)
        self.delivery_time_labels = torch.tensor(df_delivery,dtype=torch.float)
        self.inputs = torch.tensor(df_input, dtype=torch.float)

        
    def __len__(self):
        return len(self.risk_labels)
    
    def __getitem__(self, idx):  
       return {"input": self.inputs[idx], "risk": self.risk_labels[idx], "time": self.delivery_time_labels[idx]}
    
