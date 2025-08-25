import pandas as pd
import os


# Function to check if a timestamp is within any given range
def is_within_range(timestamp, ranges):
    for begin, end in ranges:
        if begin <= timestamp <= end:
            return True
    return False

def is_near_the_start(timestamp, ranges):
    for begin, end in ranges:
        # Check if the timestamp is within the first 15 minutes of the event
        if begin <= timestamp <= begin + 15 * 60:
            return True
    return False


def CalIt2_fill_label_field(personsCount, reasonOfAnomaly):

    # Ruta del archivo a revisar LINUX
    # train_filled_path = f'/home/javier/Practicas/Nuevos datasets/Callt2/preliminar/train_filled.csv'
    # train_events_path = f'/home/javier/Practicas/Nuevos datasets/Callt2/preliminar/train_events.csv'
    # output_path = f'/home/javier/Practicas/Nuevos datasets/Callt2/preliminar/train_label_filled.csv'
    
    base_path = os.path.abspath(os.path.dirname(__file__))

    train_filled_path = os.path.join(base_path, '..', '..', 'Nuevos datasets', 'Callt2', 'preliminar', 'train_filled.csv')
    train_events_path = os.path.join(base_path, '..', '..', 'Nuevos datasets', 'Callt2', 'preliminar', 'train_events.csv')
    output_path = os.path.join(base_path, '..', '..', 'Nuevos datasets', 'Callt2', 'preliminar', 'train_label_filled.csv')


    # Read the first CSV file
    df1 = pd.read_csv(train_filled_path)

    # Read the second CSV file
    df2 = pd.read_csv(train_events_path)
  

    # Extract ranges from the second CSV
    ranges = list(zip(df2['Begin'], df2['End']))

    # Iterate through each row in the first CSV
    for index, row in df1.iterrows():
        timestamp = row['timestamp']
        value = row['value']
        
        # Check if the row is even or odd
        is_even_row = (index % 2 == 0)
        
        if value > personsCount:
            if is_within_range(timestamp, ranges):
                if is_even_row:
                    # Even row: people are exiting
                    if is_near_the_start(timestamp, ranges):
                        df1['label'] = df1['label'].fillna(0).astype(int)
                        df1.at[index, 'label'] = 1  # Is an anomaly if many people are exiting near the start of an event
                        if reasonOfAnomaly:
                            df1.at[index, 'reason'] = 'Is an anomaly if many people are exiting near the start of an event'
                    else:
                        # Even row: people are exiting
                        df1['label'] = df1['label'].fillna(0).astype(int)
                        df1.at[index, 'label'] = 0
                else:
                    # Odd row: people are entering
                    if is_near_the_start(timestamp, ranges):
                        df1['label'] = df1['label'].fillna(0).astype(int)
                        df1.at[index, 'label'] = 0
                    else:
                        # Odd row: people are entering
                        df1['label'] = df1['label'].fillna(0).astype(int)
                        df1.at[index, 'label'] = 1 # Is an anomaly if many people are entering near the end of an event
                        if reasonOfAnomaly:
                            df1.at[index, 'reason'] = 'Is an anomaly if many people are entering near the end of an event'
            else:
                if is_even_row:
                    # Even row: people are exiting
                    df1['label'] = df1['label'].fillna(0).astype(int)
                    df1.at[index, 'label'] = 1  # Is an anomaly if many people are exiting when there is no event
                    if reasonOfAnomaly:
                        df1.at[index, 'reason'] = 'Is an anomaly if many people are exiting when there is no event'

                else:
                    # Odd row: people are entering
                    df1['label'] = df1['label'].fillna(0).astype(int)
                    df1.at[index, 'label'] = 1 # Is an anomaly if many people are entering when there is no event
                    if reasonOfAnomaly:
                        df1.at[index, 'reason'] = 'Is an anomaly if many people are entering when there is no event'
        else:
            df1['label'] = df1['label'].fillna(0).astype(int)
            df1.at[index, 'label'] = 0  # Not an anomaly if there is no movement or little movement when there is no event

    # Write the updated data to a new CSV file
    df1.to_csv(output_path, index=False)
    print(f'Archivo con label guardado en: {output_path}')

# CalIt2_fill_label_field(4,True)