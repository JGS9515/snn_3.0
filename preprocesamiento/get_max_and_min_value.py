# Inicializar variables para almacenar los valores máximo y mínimo
max_value = float('-inf')
min_value = float('inf')

# Abrir el archivo test.csv y leer los datos
with open('test.csv', 'r') as file:
    next(file)  # Saltar la línea de encabezado
    for line in file:
        # Extraer el valor de cada línea
        _, value = line.strip().split(',')
        value = float(value)
        
        # Actualizar el valor máximo y mínimo
        if value > max_value:
            max_value = value
        if value < min_value:
            min_value = value

# Imprimir los valores máximo y mínimo
print(f'Valor máximo: {max_value}')
print(f'Valor mínimo: {min_value}')