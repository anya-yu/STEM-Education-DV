import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# # Graph 1 (Bar Chart)
# stem_enrolled_df = pd.read_csv('datasets/top-percent-enrolled-per-country-20.csv')
# print(list(stem_enrolled_df.columns))

# # Limit Data Set
# stem_enrolled_df = stem_enrolled_df[['Country', 'Year', 'Field', 'Value', 'EDUCATION_LEV', 'SEX', 'MOBILITY']]
# stem_enrolled_df = stem_enrolled_df.drop(stem_enrolled_df[stem_enrolled_df['SEX'] != '_T'].index)
# stem_enrolled_df = stem_enrolled_df.drop(stem_enrolled_df[stem_enrolled_df['MOBILITY'] != '_T'].index)

# # Check for Unique
# print(f'Unique Years: {pd.unique(stem_enrolled_df["Year"])}')
# print(f'Unique SEX: {pd.unique(stem_enrolled_df["SEX"])}')
# print(f'Unique MOBILITY: {pd.unique(stem_enrolled_df["MOBILITY"])}')
# print(f'Unique Country: {pd.unique(stem_enrolled_df["Country"])}')

# stem_enrolled_df = stem_enrolled_df.groupby(['Country', 'Field'], as_index=False)['Value'].sum()
# stem_enrolled_df = stem_enrolled_df.drop(stem_enrolled_df[stem_enrolled_df['Value'] == 0].index)
# stem_enrolled_df['Value'] = stem_enrolled_df['Value']/1000000   # in millions

# print(f'Head: {stem_enrolled_df}')
# print(f'Columns: {list(stem_enrolled_df.columns)}')
# print(f'Number of Rows: {stem_enrolled_df.shape[0]}')

# # Graph 
# # Step 3: Pivot the data
# pivot = stem_enrolled_df.pivot(index='Country', columns='Field', values='Value').fillna(0)

# # Step 4: Plot the data
# pivot.plot(kind='bar', figsize=(10, 6), color=['#231942', '#5e548e', '#9f86c0', '#be95c4'])

# # Customize the plot
# plt.title('Number of Students in Each STEM Field by Country')
# plt.xlabel('Country')
# plt.ylabel('Number of Students (in millions)')
# plt.xticks(rotation=45)  # Rotate x-axis labels to 45 degrees for better readability
# plt.legend(title='Field')

# # Display the plot
# plt.tight_layout()  # Adjust layout to make room for rotated x-axis labels
# plt.show()




# Graph 2 (Line Chart)
stem_graduate_df = pd.read_csv('datasets/bsms-graduates-recent-us.csv')
print(list(stem_graduate_df.columns))

# Limit Data Set
stem_graduate_df = stem_graduate_df[['SEX', 'Year', 'Field', 'Value', 'EDUCATION_LEV']]
stem_graduate_df = stem_graduate_df.drop(stem_graduate_df[stem_graduate_df['SEX'] != '_T'].index)

# # Check for Unique
print(f'Unique SEX: {pd.unique(stem_graduate_df["SEX"])}')


# stem_enrolled_df = stem_enrolled_df.drop(stem_enrolled_df[stem_enrolled_df['Value'] == 0].index)
# stem_enrolled_df['Value'] = stem_enrolled_df['Value']/1000000   # in millions


stem_graduate_df = stem_graduate_df.sort_values(by=['Year', 'Field'])
print(f'Head: {stem_graduate_df.head(10)}')

stem_bs_graduate_df = stem_graduate_df[stem_graduate_df['EDUCATION_LEV'] == 'ISCED11_6']
stem_ms_graduate_df = stem_graduate_df[stem_graduate_df['EDUCATION_LEV'] == 'ISCED11_7']

all_bs_degrees = []
for degree in list(pd.unique(stem_bs_graduate_df['Field'])):
    print(degree)
    all_bs_degrees.append({f'BS {degree}': stem_bs_graduate_df.loc[stem_bs_graduate_df['Field'] == degree, 'Value'].tolist()})

all_ms_degrees = []
for degree in list(pd.unique(stem_ms_graduate_df['Field'])):
    print(degree)
    all_ms_degrees.append({f'MS {degree}': stem_ms_graduate_df.loc[stem_ms_graduate_df['Field'] == degree, 'Value'].tolist()})

# Print
for item in all_bs_degrees:
    print(item)
for item in all_ms_degrees:
    print(item)

# # sorted(list(pd.unique(stem_enrolled_df["Year"])}))
years = sorted(list(pd.unique(stem_graduate_df["Year"])))

all_values = [*[all_bs_degrees[index][degree] for degree, index in enumerate(all_bs_degrees)], *[all_ms_degrees[index][degree] for degree, index in enumerate(all_ms_degrees)]]
print(all_values)

# plt.stackplot(years, bs_graduates, ms_graduates, labels=['BS Graduates', 'MS Graduates'])
# plt.xlabel('Year')
# plt.ylabel('Number of Graduates')
# plt.title('Graduates Over Time')
# plt.legend(loc='upper left')
# plt.show()


# # Graph
# # Step 3: Pivot the data
# pivot = stem_enrolled_df.pivot(index='Country', columns='Field', values='Value').fillna(0)

# # Step 4: Plot the data
# pivot.plot(kind='line', figsize=(10, 6), color=['#231942', '#5e548e', '#9f86c0', '#be95c4'])

# # Customize the plot
# plt.title('Number of Students in Each STEM Field by Country')
# plt.xlabel('Country')
# plt.ylabel('Number of Students (in millions)')
# plt.xticks(rotation=45)  # Rotate x-axis labels to 45 degrees for better readability
# plt.legend(title='Field')

# # Display the plot
# plt.tight_layout()  # Adjust layout to make room for rotated x-axis labels
# plt.show()