import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# # Graph 1 (Bar Chart)
stem_enrolled_df = pd.read_csv('datasets/top-percent-enrolled-per-country-20.csv')
print(list(stem_enrolled_df.columns))

# prep data
stem_enrolled_df = stem_enrolled_df[['Country', 'Year', 'Field', 'Value', 'EDUCATION_LEV', 'SEX', 'MOBILITY']]
stem_enrolled_df = stem_enrolled_df.drop(stem_enrolled_df[stem_enrolled_df['SEX'] != '_T'].index)
stem_enrolled_df = stem_enrolled_df.drop(stem_enrolled_df[stem_enrolled_df['MOBILITY'] != '_T'].index)

# check for unique
print(f'Unique Years: {pd.unique(stem_enrolled_df["Year"])}')
print(f'Unique SEX: {pd.unique(stem_enrolled_df["SEX"])}')
print(f'Unique MOBILITY: {pd.unique(stem_enrolled_df["MOBILITY"])}')
print(f'Unique Country: {pd.unique(stem_enrolled_df["Country"])}')

stem_enrolled_df = stem_enrolled_df.groupby(['Country', 'Field'], as_index=False)['Value'].sum()
stem_enrolled_df = stem_enrolled_df.drop(stem_enrolled_df[stem_enrolled_df['Value'] == 0].index)
stem_enrolled_df['Value'] = stem_enrolled_df['Value']/1000000   # in millions

print(f'Head: {stem_enrolled_df}')
print(f'Columns: {list(stem_enrolled_df.columns)}')
print(f'Number of Rows: {stem_enrolled_df.shape[0]}')

# graph 
# add data to a pivot table
pivot = stem_enrolled_df.pivot(index='Country', columns='Field', values='Value').fillna(0)

# plot the data
pivot.plot(kind='bar', figsize=(10, 6), color=['#231942', '#5e548e', '#9f86c0', '#be95c4'], zorder=3)

# customize the plot
plt.grid(True, which='both', linestyle='--', linewidth=0.7, zorder=0)
plt.title('Number of Students in Each STEM Related Field by Country')
plt.xlabel('Country')
plt.ylabel('Number of Students (in millions)')
plt.xticks(rotation=45)  # Rotate x-axis labels to 45 degrees for better readability
plt.legend(title='Fields')

# display the plot
plt.tight_layout()  # Adjust layout to make room for rotated x-axis labels
plt.show()




# Graph 2 (Line Chart)
stem_graduate_df = pd.read_csv('datasets/bsms-graduates-recent-us.csv')
print(list(stem_graduate_df.columns))

# prep data
stem_graduate_df = stem_graduate_df[['SEX', 'Year', 'Field', 'Value', 'EDUCATION_LEV']]
stem_graduate_df = stem_graduate_df.drop(stem_graduate_df[stem_graduate_df['SEX'] != '_T'].index)
stem_graduate_df['Value'] = stem_graduate_df['Value']/1000000   # in millions

# check for unique
print(f'Unique SEX: {pd.unique(stem_graduate_df["SEX"])}')

stem_graduate_df = stem_graduate_df.sort_values(by=['Year', 'Field'])
print(f'Head: {stem_graduate_df.head(10)}')

# separate df based on education level
stem_bs_graduate_df = stem_graduate_df[stem_graduate_df['EDUCATION_LEV'] == 'ISCED11_6']
stem_ms_graduate_df = stem_graduate_df[stem_graduate_df['EDUCATION_LEV'] == 'ISCED11_7']

# create dictionaries to store field and corresponding graduates
all_bs_degrees = []
for degree in list(pd.unique(stem_bs_graduate_df['Field'])):
    # print(degree)
    all_bs_degrees.append({f'BS {degree}': stem_bs_graduate_df.loc[stem_bs_graduate_df['Field'] == degree, 'Value'].tolist()})

all_ms_degrees = []
for degree in list(pd.unique(stem_ms_graduate_df['Field'])):
    # print(degree)
    all_ms_degrees.append({f'MS {degree}': stem_ms_graduate_df.loc[stem_ms_graduate_df['Field'] == degree, 'Value'].tolist()})

# order by year ascending
years = sorted(list(pd.unique(stem_graduate_df["Year"])))

# all bachelors fields for each layer
bs_fields = []
for item in all_bs_degrees:
    bs_fields.append(list(item.keys())[0])

# all values per each bachelors field
bs_values = []
for item in all_bs_degrees:
    bs_values.append(*list(item.values()))

# all masters fields for each layer
ms_fields = []
for item in all_ms_degrees:
    ms_fields.append(list(item.keys())[0])

# all values per each masters field
ms_values = []
for item in all_ms_degrees:
    ms_values.append(*list(item.values()))

print(f'bs fields : {bs_fields}\nbs values : {bs_values}')
print(f'ms fields : {ms_fields}\nms values : {ms_values}')

colors = ['#797d62','#9b9b7a','#d9ae94','#f1dca7','#ffcb69','#d08c60','#997b66','#ad9585']

plt.stackplot(years, *bs_values, *ms_values, labels=[*bs_fields, *ms_fields], colors=colors)
plt.xlabel('Year')
plt.ylabel('Number of Graduates (in millions)')
plt.title('Number of Recent US STEM Related Graduates Over Time')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2) # places legend below the plot in two columns
# Set x-axis ticks to whole numbers only
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

plt.tight_layout()
plt.show()
