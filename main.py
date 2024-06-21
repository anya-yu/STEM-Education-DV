import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# # Graph 1 (Bar Chart)
# stem_enrolled_df = pd.read_csv('datasets/top-percent-enrolled-per-country-20.csv')
# print(list(stem_enrolled_df.columns))

# # prep data
# stem_enrolled_df = stem_enrolled_df[['Country', 'Year', 'Field', 'Value', 'EDUCATION_LEV', 'SEX', 'MOBILITY']]
# stem_enrolled_df = stem_enrolled_df.drop(stem_enrolled_df[stem_enrolled_df['SEX'] != '_T'].index)
# stem_enrolled_df = stem_enrolled_df.drop(stem_enrolled_df[stem_enrolled_df['MOBILITY'] != '_T'].index)

# # check for unique
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

# # graph 
# # add data to a pivot table
# pivot = stem_enrolled_df.pivot(index='Country', columns='Field', values='Value').fillna(0)

# # plot the data
# pivot.plot(kind='bar', figsize=(10, 6), color=['#231942', '#5e548e', '#9f86c0', '#be95c4'], zorder=3)

# # customize the plot
# plt.grid(True, which='both', linestyle='--', linewidth=0.7, zorder=0)
# plt.title('Number of Students in Each STEM Related Field by Country')
# plt.xlabel('Country')
# plt.ylabel('Number of Students (in millions)')
# plt.xticks(rotation=45)  # Rotate x-axis labels to 45 degrees for better readability
# plt.legend(title='Fields')

# # display the plot
# plt.tight_layout()  # Adjust layout to make room for rotated x-axis labels
# plt.show()




# # Graph 2 (Scatter Plot)
# stem_graduate_df = pd.read_csv('datasets/bsms-graduates-recent-us.csv')
# print(list(stem_graduate_df.columns))

# # prep data
# stem_graduate_df = stem_graduate_df[['SEX', 'Year', 'Field', 'Value', 'EDUCATION_LEV']]
# stem_graduate_df = stem_graduate_df.drop(stem_graduate_df[stem_graduate_df['SEX'] != '_T'].index)
# stem_graduate_df['Value'] = stem_graduate_df['Value']/1000000   # in millions

# # check for unique
# print(f'Unique SEX: {pd.unique(stem_graduate_df["SEX"])}')

# stem_graduate_df = stem_graduate_df.sort_values(by=['Year', 'Field'])
# print(f'Head: {stem_graduate_df.head(10)}')

# # separate df based on education level
# stem_bs_graduate_df = stem_graduate_df[stem_graduate_df['EDUCATION_LEV'] == 'ISCED11_6']
# stem_ms_graduate_df = stem_graduate_df[stem_graduate_df['EDUCATION_LEV'] == 'ISCED11_7']

# # create dictionaries to store field and corresponding graduates
# all_bs_degrees = []
# for degree in list(pd.unique(stem_bs_graduate_df['Field'])):
#     # print(degree)
#     all_bs_degrees.append({f'BS {degree}': stem_bs_graduate_df.loc[stem_bs_graduate_df['Field'] == degree, 'Value'].tolist()})

# all_ms_degrees = []
# for degree in list(pd.unique(stem_ms_graduate_df['Field'])):
#     # print(degree)
#     all_ms_degrees.append({f'MS {degree}': stem_ms_graduate_df.loc[stem_ms_graduate_df['Field'] == degree, 'Value'].tolist()})

# # order by year ascending
# years = sorted(list(pd.unique(stem_graduate_df["Year"])))

# # all bachelors fields for each layer
# bs_fields = []
# for item in all_bs_degrees:
#     bs_fields.append(list(item.keys())[0])

# # all values per each bachelors field
# bs_values = []
# for item in all_bs_degrees:
#     bs_values.append(*list(item.values()))

# # all masters fields for each layer
# ms_fields = []
# for item in all_ms_degrees:
#     ms_fields.append(list(item.keys())[0])

# # all values per each masters field
# ms_values = []
# for item in all_ms_degrees:
#     ms_values.append(*list(item.values()))

# print(f'bs fields : {bs_fields}\nbs values : {bs_values}')
# print(f'ms fields : {ms_fields}\nms values : {ms_values}')

# colors = ['#797d62','#9b9b7a','#d9ae94','#f1dca7','#ffcb69','#d08c60','#997b66','#ad9585']

# plt.stackplot(years, *bs_values, *ms_values, labels=[*bs_fields, *ms_fields], colors=colors)
# plt.xlabel('Year')
# plt.ylabel('Number of Graduates (in millions)')
# plt.title('Number of Recent US STEM Related Graduates Over Time')
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2) # places legend below the plot in two columns
# # Set x-axis ticks to whole numbers only
# plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# plt.tight_layout()
# plt.show()



# Graph 3 (Nested Donut Chart)
stem_nonstem_df = pd.read_csv('datasets/stem-v-nonstem-graduates.csv')
print(list(stem_nonstem_df.columns))

# prep data
stem_nonstem_df = stem_nonstem_df[['Year', 'Field', 'Value', 'Education level']]
stem_nonstem_df  = stem_nonstem_df.dropna()

# Rename the Fields
stem_list = [
    'Engineering, manufacturing and construction',
    'Information and Communication Technologies (ICTs)',
    'Natural sciences, mathematics and statistics',
    'Science, technology, engineering and mathematics']
stem_nonstem_df['Field'] = ["Stem" if field in stem_list else "Non-Stem" for field in stem_nonstem_df['Field']]

print(f'Changed field names: {stem_nonstem_df.head(10)}')
print(f'Unique education levels: {stem_nonstem_df['Field'].unique()}')

ed_lvl_map = {
 'Upper secondary vocational education': 'Upper Secondary',
 'Short-cycle tertiary education': 'Short Tertiary',
 'Bachelor’s or equivalent level': 'Bachelor’s',
 'Master’s or equivalent level': 'Master’s',
 'Doctoral or equivalent level': 'Doctoral',
 'Post-secondary non-tertiary vocational education': 'Non-Tertiary'
}
stem_nonstem_df['Education level'] = [ed_lvl_map[ed_lvl] for ed_lvl in stem_nonstem_df['Education level']]

print(f'Changed education level names: {stem_nonstem_df.head(10)}')

stem_nonstem_df = stem_nonstem_df.groupby(['Year', 'Field', 'Education level'], as_index=False)['Value'].sum()

print(f'Summed Countries: {stem_nonstem_df.head(10)}')

# sort in custom education level order
ed_lvl_order = ['Upper Secondary','Non-Tertiary', 'Short Tertiary', 'Bachelor’s', 'Master’s', 'Doctoral']
stem_nonstem_df['Education level'] = pd.Categorical(stem_nonstem_df['Education level'], categories=ed_lvl_order, ordered=True)

# sort in year and education level
stem_nonstem_df = stem_nonstem_df.sort_values(by=['Year', 'Field', 'Education level'])

print(f'Ordered Levels: {stem_nonstem_df.head(10)}')

# for year in pd.unique(stem_nonstem_df['Year']): # Each Year
#     for field in pd.unique(stem_nonstem_df['Field']): # Each Field

grouped_by_field = [stem_nonstem_df['Value'][i:i + 6] for i in range(0, len(stem_nonstem_df['Value']), 6)] # split list by grouping by field (stem v nonstem)
grouped_by_year = [grouped_by_field[i:i + 2] for i in range(0, len(grouped_by_field), 2)] # split list by grouping by year (2017-2021)

vals = np.array(grouped_by_year) # convert list to np.array
field_vals = np.array(grouped_by_field)
year_vals = np.array([stem_nonstem_df['Value'][i:i + 12] for i in range(0, len(stem_nonstem_df['Value']), 12)])


# Pie Chart
fig, ax = plt.subplots()

size = 0.2

#           0           1          2           3         4           5          6         7           8
colors = ['#264653', '#425e69', '#506a74', '#5d757e', '#6b8189', '#788c94', '#93a3a9', '#aebabf', '#c9d1d4',
        '#2A9D8F', '#45aa9d', '#53b0a4', '#60b6ab', '#7bc2b9', '#95cec7', '#b0dbd5', '#cae7e3', '#e5f3f1',
        '#E9C46A', '#eccc7d', '#eed087', '#efd390', '#f2dba3', '#f4e2b5', '#f7eac8', '#faf1da', '#fdf8ed',
        '#F4A261', '#f6ae75', '#f7b47f', '#f7ba89', '#f8c093', '#f9c69d', '#fad1b0', '#fcddc4', '#fde8d8',
        '#E76F51', '#ea8167', '#ec8a72', '#ed937d', '#ef9c88', '#f0a593', '#f3b7a8', '#f6c9be', '#f9dbd4'
]
#            0             1         3           4          6         7
omap = [    '#264653', '#425e69', '#5d757e', '#6b8189', '#93a3a9', '#aebabf',
            '#264653', '#425e69', '#5d757e', '#6b8189', '#93a3a9', '#aebabf',
            '#2A9D8F', '#45aa9d', '#60b6ab', '#7bc2b9', '#b0dbd5', '#cae7e3',
            '#2A9D8F', '#45aa9d', '#60b6ab', '#7bc2b9', '#b0dbd5', '#cae7e3',
            '#E9C46A', '#eccc7d', '#efd390', '#f2dba3', '#f7eac8', '#faf1da',
            '#E9C46A', '#eccc7d', '#efd390', '#f2dba3', '#f7eac8', '#faf1da',
            '#F4A261', '#f6ae75', '#f7ba89', '#f8c093', '#fad1b0', '#fcddc4',
            '#F4A261', '#f6ae75', '#f7ba89', '#f8c093', '#fad1b0', '#fcddc4',
            '#E76F51', '#ea8167', '#ed937d', '#ef9c88', '#f3b7a8', '#f6c9be',
            '#E76F51', '#ea8167', '#ed937d', '#ef9c88', '#f3b7a8', '#f6c9be'
]
out_labels = ['', '', '', '', '', '',
          '', '', '', '', '', '',
          '', '', '', '', '', '',
          '', '', '', '', '', '',
          'Upper Secondary', 'Non-Tertiary', 'Short Tertiary', 'Bachelor’s', 'Master’s', 'Doctoral',
          '', '', '', '', '', '',
          '', '', '', '', '', '',
          '', '', '', '', '', '',
          '', '', '', '', '', '',
          '', '', '', '', '', '',
]

middle_labels = ['', '', '', '', 'Non-STEM', 'STEM', '', '', '', '']

# outside layer
ax.pie(vals.flatten(), radius=1, colors=omap,
       wedgeprops=dict(width=size, edgecolor='w'), labels=out_labels, rotatelabels=False, labeldistance=1.05, textprops={'fontsize': 8})

ax.pie(field_vals.sum(axis=1), radius=1-size, colors=[colors[i] for i in [0, 6, 9, 15, 18, 24, 27, 33, 36, 42]],
       wedgeprops=dict(width=size, edgecolor='w'), labels=middle_labels, labeldistance=0.765, textprops={'fontsize': 8})

ax.pie(year_vals.sum(axis=1), radius=1-size*2, colors=[colors[i] for i in [0, 9, 18, 27, 36]],
       wedgeprops=dict(width=size, edgecolor='w'), labels=['2017', '2018', '2019', '2020', '2021'], labeldistance=0.45, rotatelabels=False, textprops={'fontsize': 10})

ax.set(aspect="equal", title='Pie plot with `ax.pie`')
plt.title('STEM Vs. Non-STEM Graduate Education Levels Within 5 Years')
plt.show()

# [[[6560369.0 # stem, 3768882.0 #nonstem] # 2017, BS
# ]
#  [37., 40.] # 2018
#  [29., 10.]] # 2019
# [[['Upper Secondary','Non-Tertiary', 'Short Tertiary', 'Bachelor’s', 'Master’s', 'Doctoral'],['Stem']],[['Non-Stem'],['Stem']],[2019],[2020],[2021]]

# [60., 32., 37., 40., 29., 10.] # flattened