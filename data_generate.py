import numpy as np
import pandas as pd
import random

np.random.seed(42)

brands = ['Toyota','Hyundai','Honda','Ford','Skoda','Mahindra']

rows = []
for i in range(1200):
    brand = random.choice(brands)
    year = np.random.randint(2005, 2021)
    age = 2021 - year
    mileage = int(np.random.normal(60000 - age*3000, 15000))
    mileage = max(2000, abs(mileage))
    engine_cc = random.choice([1000, 1200, 1400, 1500, 1600, 1800, 2000])
    hp = int(engine_cc * np.random.uniform(0.06,0.09))
    base = {'Toyota':800000, 'Hyundai':600000, 'Honda':700000, 'Ford':650000, 'Skoda':750000, 'Mahindra':500000}[brand]
    price = base * np.exp(-0.12*age) - 0.8*mileage + np.random.normal(0,30000)
    price = max(20000, int(price))
    rows.append([brand, year, mileage, engine_cc, hp, price])

cols = ['brand','year','mileage','engine_cc','hp','price']
df = pd.DataFrame(rows, columns=cols)
df.to_csv('used_cars_sample.csv', index=False)
print('wrote used_cars_sample.csv with', len(df), 'rows')
