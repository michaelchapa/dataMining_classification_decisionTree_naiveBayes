import numpy as np
import pandas as pd

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

data = pd.DataFrame(d[1:], columns = d[0])

def expandData_byCount(data):
    df = pd.DataFrame()
    
    for instance in data.iterrows():
        count = int(instance[1][4])
        
        for x in range(count):
            df = df.append(pd.Series(instance[1]))
            
    print(df)
    
def main():
    df2 = expandData_byCount(data)
    
# Context the file is running in is __main__
if __name__ == "__main__":
    main()