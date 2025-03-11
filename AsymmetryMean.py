import pandas as pd

# Leer el archivo CSV
df = pd.read_csv('assimetry_results.csv')

# Agrupar por la clasificación y calcular los promedios
resultados = df.groupby(df.iloc[:, 1]).agg({
    df.columns[2]: 'mean',
    df.columns[3]: 'mean'
})

resultados.columns = ['Promedio Vertical Assymetry', 'Promedio Horizontal Assimetry']

print(resultados)

resultados.to_csv('resultados_promedios.csv')

print("Los resultados han sido guardados en 'resultados_promedios.csv'")