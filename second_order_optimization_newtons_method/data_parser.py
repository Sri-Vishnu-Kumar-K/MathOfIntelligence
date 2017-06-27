import pandas as pd #To load in the data

def get_data(path='cal_data.csv', cols = ['dist_cycled','calories'], n_rows = 1000):
    df = pd.read_csv(path) #Reads in the CSV file specified
    df = df[cols] #Gets only the specified columns
    df.fillna(0, inplace = True) #Replaces missing values with 0.
    print('Loaded df of size %d'%(len(df)))
    arr = df.as_matrix() #returns the dataframe as a python array.
    # print(arr[:2,:])

    return arr

if __name__=='__main__':
    get_data('cal_data.csv') #Driver to test the function.
