import pandas as pd

boning_data = pd.read_csv('./ampc2/Boning.csv')
slicing_data = pd.read_csv('./ampc2/Slicing.csv')


boning_data['class'] = 0
slicing_data['class'] = 1


combined_data = pd.concat([boning_data, slicing_data], axis=0).reset_index(drop=True)


# I ASSUMED THE VALUES IN THE FRAME COLUMN TO BE THE STUDENT NUMBERS
column_mapping = {
    0: (["Neck x", "Neck y", "Neck z"], ["Head x", "Head y", "Head z"]),
    1: (["Right Shoulder x", "Right Shoulder y", "Right Shoulder z"], ["Left Shoulder x", "Left Shoulder y", "Left Shoulder z"]),
    2: (["Right Upper Arm x", "Right Upper Arm y", "Right Upper Arm z"], ["Left Upper Arm x", "Left Upper Arm y", "Left Upper Arm z"]),
    3: (["Right Forearm x", "Right Forearm y", "Right Forearm z"], ["Left Forearm x", "Left Forearm y", "Left Forearm z"]),
    4: (["Right Hand x", "Right Hand y", "Right Hand z"], ["Left Hand x", "Left Hand y", "Left Hand z"]),
    5: (["Right Upper Leg x", "Right Upper Leg y", "Right Upper Leg z"], ["Left Upper Leg x", "Left Upper Leg y", "Left Upper Leg z"]),
    6: (["Right Lower Leg x", "Right Lower Leg y", "Right Lower Leg z"], ["Left Lower Leg x", "Left Lower Leg y", "Left Lower Leg z"]),
    7: (["Right Foot x", "Right Foot y", "Right Foot z"], ["Left Foot x", "Left Foot y", "Left Foot z"]),
    8: (["Right Toe x", "Right Toe y", "Right Toe z"], ["Left Toe x", "Left Toe y", "Left Toe z"]),
    9: (["L5 x", "L5 y", "L5 z"], ["T12 x", "T12 y", "T12 z"])
}

filtered_rows = []

for index, row in combined_data.iterrows():
    student_number = str(int(float(row['Frame'])))  
    student_ending = int(student_number[-1])  
    column_set_1, column_set_2 = column_mapping[student_ending]
    
    try:
        selected_data = row[['Frame'] + column_set_1 + column_set_2].tolist()
        
        selected_data.append(row['class'])
        
        filtered_rows.append(selected_data)
    except KeyError as e:
        print(f"KeyError: {e} in row {index}")

final_columns = ['Frame'] + column_set_1 + column_set_2 + ['class']

final_data = pd.DataFrame(filtered_rows, columns=final_columns)

final_data.to_csv('combined_dataset.csv', index=False)

print("Data collection completed and saved to 'combined_dataset.csv'.")
