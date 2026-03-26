import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os


json_files = glob.glob('experiments/*.json') # Finds all .json files ---> unine shell wild card
data_list = []

for file in json_files:
    with open(file, 'r') as f:
        data = json.load(f)
        name =os.path.splitext(os.path.basename(file))[0] # split return ('model1', '.json') , take the name
        data_list.append({
            'Model': name,
            'Accuracy': data.get('accuracy') / 100,  # Convert to decimal as plot excepts 0-1
            'Time': data.get('train_time') 
        })
print (data_list)
df = pd.DataFrame(data_list) # list of dictionary , each dictionary is a row , keys are columns
df = df.sort_values('Accuracy', ascending=True) 

fig, ax1 = plt.subplots(figsize=(10, 6))

bars = ax1.barh(df['Model'], df['Accuracy'], color='skyblue', alpha=0.8)
ax1.set_xlabel('Test Accuracy', color='tab:blue', fontsize=12)
ax1.set_xlim(0, 1.05) # Accuracy is 0-1
ax1.tick_params(axis='x', labelcolor='tab:blue')


for bar in bars:
    width = bar.get_width()
    ax1.text(width - 0.1, bar.get_y() + bar.get_height()/2, 
             f'{width:.1%}', ha='center', va='center', color='black', fontweight='bold')


# We create a second x-axis that shares the same y-axis
ax2 = ax1.twiny() 
ax2.plot(df['Time'], df['Model'], color='tab:red', marker='o', linewidth=2)
ax2.set_xlabel('Training Time (seconds)', color='tab:red', fontsize=12)
ax2.tick_params(axis='x', labelcolor='tab:red')

# 3. FINISH
# ----------------------------
plt.title("Model Showdown: Accuracy vs. Speed")
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.show()