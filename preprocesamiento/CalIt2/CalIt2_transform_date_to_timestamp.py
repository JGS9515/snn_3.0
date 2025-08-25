import csv
import time
import pandas as pd
from datetime import datetime
import pytz
import os


# Function to convert date and time to Unix timestamp
def convert_to_timestamp(date_str, time_str):

    gmt_plus_1 = pytz.timezone('Etc/GMT-2')

    # Convertir las fechas a objetos datetime con la zona horaria GMT+01:00
    fecha1 = gmt_plus_1.localize(datetime.strptime(date_str + ' ' + time_str, '%m/%d/%y %H:%M:%S'))
    result = int(fecha1.timestamp())
    return result

    # dt = datetime.strptime(f"{date_str} {time_str}", "%m/%d/%y %H:%M:%S")
    # return int(time.mktime(dt.timetuple()))

    # date_parts = date_str.split('/')    

    # # print(date_parts) # Output: ['07', '24', '2005']
    
    # time_parts = time_str.split(':')
    # # Convertir las partes de la fecha y hora a enteros
    # year = int(date_parts[2])
    # if year < 50:  # Asumiendo que los años menores a 50 son del siglo 21
    #     year += 2000
    # else:
    #     year += 1900

    # month = int(date_parts[0])
    # day = int(date_parts[1])
    # hour = int(time_parts[0])
    # minute = int(time_parts[1])
    # second = int(time_parts[2])

    # fecha1 = datetime(year, month, day, hour, minute, second)
    # # fecha1 = datetime(2005, 7, 24, 0, 0, 0)
    # timestamp1 = int(time.mktime(fecha1.timetuple()))
    # return timestamp1

# Read the CSV file using pandas
def CalIt2_transform_date_to_timestamp():
    base_path = os.path.dirname(__file__)
    input_path = os.path.join(base_path, '..', '..', 'Nuevos datasets', 'Callt2', 'preliminar', 'CalIt2.csv')
    output_path = os.path.join(base_path, '..', '..', 'Nuevos datasets', 'Callt2', 'preliminar', 'train.csv')


    df = pd.read_csv(input_path)

    # Open the output CSV file
    with open(output_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        
        # Write the header of the CSV file
        writer.writerow(['timestamp','value','label'])
        
        # Iterate over the rows of the DataFrame
        for index, row in df.iterrows():
            date_event = row['Date']
            time_event = row['Time']
            if index == 9421:
                print(date_event, time_event)
            timestamp = convert_to_timestamp(date_event, time_event)
            # label = 'out flow' if flow_id == '7' else 'in flow'
            writer.writerow([timestamp, row['Count']])

    print(f'Archivo con tranin con dates transformados a timestamps guardado en: {output_path}')

# CalIt2_transform_date_to_timestamp()