ACTSAT_2020_Cols = [col for col in data_2020['ACT/SAT Math'].columns if col != 'Key']
for col in ACTSAT_2020_Cols:
    if col != 'Key':
        m = data_2020['ACT/SAT Math'][col].apply(float).max()
        data_2020['ACT/SAT Math'][col] = data_2020['ACT/SAT Math'][col].apply(float).apply(lambda x: x/m)

data_2020['ACT/SAT Math'] = data_2020['ACT/SAT Math'][ACTSAT_2020_Cols].mean(axis=1)


ACTSAT_2021_Cols = [col for col in data_2021['ACT/SAT Math'].columns if col != 'Key']
for col in ACTSAT_2021_Cols:
    if col != 'Key':
        m = data_2021['ACT/SAT Math'][col].apply(float).max()
        data_2021['ACT/SAT Math'][col] = data_2021['ACT/SAT Math'][col].apply(float).apply(lambda x: x/m)

data_2021['ACT/SAT Math'] = data_2021['ACT/SAT Math'][ACTSAT_2021_Cols].mean(axis=1)