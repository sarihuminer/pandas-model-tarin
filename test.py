import pandas as pd

excel_file = 'WikiRef-input.xlsx'
data = pd.read_excel(excel_file)
# movies = pd.read_excel(excel_file, engine='openpyxl')
# movies.head()

# data = {'y_Actual':    [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
#        'y_Predicted': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]
#      }

df = pd.DataFrame(data, columns=['isRefQK', 'title'])
print(df)
confusion_matrix = pd.crosstab(df['ref'], df['title'], rownames=['ref'], colnames=['title'])
print(confusion_matrix)





