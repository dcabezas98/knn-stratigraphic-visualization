import pandas as pd
import numpy as np

dat=pd.read_excel('horizontal sections data.xlsx')
basements=dat[dat.Valor=='S']
new_basements=pd.DataFrame(columns=dat.columns)
for i in range(len(basements)):
    row=basements.iloc[i]
    for c in range(row['Cota']-1, np.min(dat.Cota)-1,-1):
        new_basements=new_basements.append({'UTM_X':row['UTM_X'], 'UTM_Y':row['UTM_Y'], 'Valor':row['Valor'],
                                    'Clase':row['Clase'], 'Cota':c, 'Codi':row['Codi']}, ignore_index=True)
                                
dat=dat.append(new_basements,ignore_index=True,sort=False)

dat.to_excel('hsd new basements.xls', index=False)