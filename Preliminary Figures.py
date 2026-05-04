import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load my dataset
df = pd.read_csv('SeoulBikeData.csv', encoding='unicode_escape')
df = df[df['Functioning Day'] == 'Yes']
df = df.drop(columns=['Functioning Day'])

#Average Bike Demand by Hour of Day figure
hourly_avg = df.groupby('Hour')['Rented Bike Count'].mean()

plt.figure(figsize=(10, 5))
plt.bar(hourly_avg.index, hourly_avg.values, color='purple', edgecolor='green')
plt.xlabel('Hour of Day')
plt.ylabel('Average Rented Bike Count')
plt.title('Average Bike Demand by Hour of Day')
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig('fig1_demand_by_hour.png', dpi=300)
plt.show()
print("Figure 1 saved.")

#Average Bike Demand by Season figure 2
season_avg = df.groupby('Seasons')['Rented Bike Count'].mean().reindex(['Spring', 'Summer', 'Autumn', 'Winter'])

plt.figure(figsize=(7, 5))
plt.bar(season_avg.index, season_avg.values, color='purple', edgecolor='green')
plt.xlabel('Season')
plt.ylabel('Average Rented Bike Count')
plt.title('Average Bike Demand by Season')
plt.tight_layout()
plt.savefig('fig2_demand_by_season.png', dpi=300)
plt.show()
print("Figure 2 saved.")