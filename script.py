# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 22:00:18 2020

@author: urmay
"""
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

# =============================================================================
# Task : 1 Predicting the l -( tuning parameter ) for score prediction
# =============================================================================
# Input data using Panda library
data = pd.read_csv(r'C:\Users\urmay\Downloads\ODI_data\ODI_data.csv')

# Selecting the required columns as per the requirement.
sel_data = pd.DataFrame(data, columns = ['Match', 'Innings', 'Runs.Remaining', 'Wickets.in.Hand','Over', 'Innings.Total.Runs'])

# selecting the data only for the first innings as mentioned
inning_first = sel_data['Innings'] == 1
sel_data = sel_data[inning_first]
# As we are more focused on the remaining overs we are converting the over to remaining overs
sel_data['Over'] = 50-sel_data['Over']

# As we are focused on finding out of the mean of the runs. We are Applying groupby with overs and wickets in hand an ==> average on runs remaining 
new_data =sel_data.groupby(by=['Over','Wickets.in.Hand'])['Runs.Remaining'].mean()
# Converting into the dataframce
p1=new_data.to_frame().reset_index()
# Renaming the column as y ==> we consider it as the original Y value
p1.rename(columns={'Runs.Remaining':'y'},inplace = True)

# Applying group by for only wkt in hand  & converting it into the dataframe
new_data_1 =sel_data.groupby(by=['Wickets.in.Hand'])['Runs.Remaining'].mean()
p2=new_data_1.to_frame().reset_index()

# Joining the two data frame 
p3 =p1.join(p2.set_index('Wickets.in.Hand'),on ='Wickets.in.Hand',how='left')
# rw ===>r{0}w
p3.rename(columns={'Runs.Remaining':'rw'},inplace = True)

# This function will give the avg score with n overs remaining & n wickets in hand. 
def give_avg_score(remaining_overs,remaining_wkt):
    return p1[(p1['Over']==remaining_overs) & (p1['Wickets.in.Hand']==remaining_wkt)]

#avg_score=(give_avg_score(0,0)['Innings.Total.Runs']).tolist()

# We are removing the 0 over remaining Entries as it want be required @ any point of time
# And even we dont want to train model on that.
p3=p3[p3.Over!=0]  

# This Function will give us the Tuned parameter -Which is l using gradient descent method.
def calculate_hyperparameter(v,rw,y):
    #remaining_overs=v
    #avg_score=rw
    #y= original value
    #v=p3.ix[:,1]
    #rw=p3.ix[:,3]
    #y=p3.ix[:,2]
    
    l=0 #Initializing tuning parameter as 0
    L = 0.0001  # The learning Rate
    epochs = 100000  # The number of iterations to perform gradient descent

    n = float(len(v)) # Number of elements in v
    
    # Performing Gradient Descent 
    for i in range(epochs): 
        y_pred = rw * ( 1- np.exp(-v*l/rw)) # The current predicted value of y
        D_l= (-2/n) * sum( v* np.exp(-l*v/rw)*(y-y_pred))  # Derivative wrt l
        l = l - L * D_l # Update l
        print(i)
        print(l)
    
    return l

#give_avg_score()
    
# my_l is the tuned l which we will put in the  equation 
# Input :  Remaining over ranges from : 0 to 49
#       :  Remaining wkts ranges from : 0 to 10 ( though 0 reamining  wkt doesnt make any sense but we have also used it to make model understand that )
#       : total combination 490 out which we got final 467 entry because few columns doesnt  have data for that (over and wkt combination)
    
my_l=calculate_hyperparameter(p3.ix[:,0],p3.ix[:,3],p3.ix[:,2])
print(my_l)
##############
#my_l = 14.46
##############
# over remaing 7  : with 10 wkts in had : avg score ==> 226.858
y_pred = 226.858 * ( 1- np.exp(-7*14.46/226.858))
print(y_pred)
# =============================================================================
# Task : 2  Ploting the Run graph based on the remainig over vs remaining wkts(0-10)
# =============================================================================
# Plot for over no 6 to 15

x6_15=p3[(p3['Over'] ==6) | (p3['Over'] ==7) | (p3['Over'] ==8) | (p3['Over'] ==9) |(p3['Over'] ==10) |(p3['Over'] ==11) | (p3['Over'] ==12) | (p3['Over'] ==13) | (p3['Over'] ==14) | (p3['Over'] ==15)] 

def predicting_score():
    predicted_score=[]
    for i in range(x6_15['Over'].size):
        #y_pred = rw * (1- np.exp(- v *14.46/rw))
        y_pred = x6_15['rw'].iloc[i] * (1- np.exp(- x6_15['Over'].iloc[i] *14.46/x6_15['rw'].iloc[i]))
        predicted_score.append(y_pred)
    return predicted_score

predicted_score = pd.DataFrame(predicting_score())
predicted_score.columns = ['y_pred']

#Preparing final data for ploting
plot_data = pd.concat([x6_15.reset_index(drop=True), predicted_score], axis=1)
plot_data.rename(columns={'Over':'Remaining_Over'},inplace = True)

#Saving the data to csv file
plot_data.to_csv('ploting_data.csv',index=False) 

#Plotting Remainin Overs vs Wickets in Hand vs Predicted Score
g= sns.FacetGrid(plot_data,col="Remaining_Over",height=5,col_wrap=5,margin_titles=True,hue="Remaining_Over")
g.map(sns.pointplot,"Wickets.in.Hand","y_pred",alpha=0.7)
g.set_axis_labels("Wickets.in.Hand","Predicted_Score")
g.add_legend()
g.savefig("output.png")