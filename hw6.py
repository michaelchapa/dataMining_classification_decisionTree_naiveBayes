import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

####################### expandData_byCount ################################
# Purpose: 
#   Takes each instance and get's integer value of 'Count' attribute
#   Appends instance to a new dataFrame times 'Count' value
# Parameters:
#   I       DataFrame       data
# Returns:
#   DataFrame
# Notes:
#   None
def expandData_byCount(df):
    df2 = pd.DataFrame()
    
    for instance in df.iterrows():
        count = instance[1][4]
        
        for x in range(count):
            df2 = df2.append(instance[1], ignore_index = True)
            
    return df2


######################## encodeCategorical #################################
# Purpose:
#   Encodes Categorical attributes (features) into one-of-kinds. 
#   If an attribute has three values 'A', 'B', 'C'; it encodes
#   'A' as 0, 'B' as 1, 'C' as 2. 
# Parameters:
#   I       DataFrame       df
# Returns:
#   DataFrame with Encoded attributes and non-categorical attributes.
# Notes:
#   The Encode section appends the encoded values,
#   the Encoded column will be named '{attributeName}_code'.
def encodeCategorical(df):
    obj_df = df.select_dtypes(include = ['object']).copy()
    categoricalAttr = list(obj_df.columns)
    lb_make = LabelEncoder()
    
    # Encode
    for attr in categoricalAttr:
        df[attr + '_code'] = lb_make.fit_transform(obj_df[attr])
        
    # Remove categorical attributes
    df = df.drop(columns = categoricalAttr)
    
    return df


####################### create_displayDecisionTree #########################
# Purpose:
#   Creates a decision tree classifier, display the tree
# Parameters:
#   I       DataFrame       data
#   I       String          target      Target Class attribute name
# Returns:
#   None
# Notes:
#   The target class should be a column name of the dataFrame
def create_displayDecisionTree(data, target):
    plt.figure(figsize=(15, 15)) 
    clf = DecisionTreeClassifier(random_state = 0).fit( \
                                 data.drop(columns = target), data[target])
    plot_tree(clf, filled = True)
    plt.show()
    
################ create_display_naiveBayes_classifier #####################
# Purpose:
#   ##
# Parameters:
#   ##
#   ##
# Returns:
#   ##
# Notes:
#   None
def create_display_naiveBayes_classifier(data):
    return data



def main():
    d = [['department', 'status', 'age', 'salary', 'count'], 
    ['sales', 'senior', '31..35', '46k..50k', 30],
    ['sales', 'junior', '26..30', '26k..30k', 40],
    ['sales', 'junior', '31..35', '31k..35k', 40],
    ['systems', 'junior', '21..25', '46k..50k', 20],
    ['systems', 'senior', '31..35', '66k..70k', 5],
    ['systems', 'junior', '26..30', '46k..50k', 3],
    ['systems', 'senior', '41..45', '66k..70k', 3],
    ['marketing', 'senior', '36..40', '46k..50k', 10],
    ['marketing', 'junior', '31..35', '41k..45k', 4],
    ['secretary', 'senior', '46..50', '36k..40k', 4],
    ['secretary', 'junior', '26..30', '26k..30k', 6]]

    df1 = pd.DataFrame(d[1:], columns = d[0])
    df2 = expandData_byCount(df1)
    df2 = encodeCategorical(df2)
    
    # Per assignment, remove 'Count' column from df2
    df2 = df2.drop(columns = 'count')
    
    create_displayDecisionTree(df2, 'department_code')
    
# Context the file is running in is __main__
if __name__ == "__main__":
    main()
