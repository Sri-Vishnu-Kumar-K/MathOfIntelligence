import pandas as pd

def get_data(path='cal_data.csv', cols = ['dist_cycled','calories'], n_rows = 1000):
    df = pd.read_csv(path)
    df = df[cols]
    df.fillna(0, inplace = True)
    arr = df.iloc[:n_rows].as_matrix()
    # print(arr[:2,:])

    return arr

if __name__=='__main__':
    get_data('cal_data.csv')
