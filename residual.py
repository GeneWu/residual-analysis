import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def cal_d(data, attribute_1, attribute_2):
	df1 = pd.crosstab(data[attribute_1],data[attribute_2], margins=True)

	# exp freq tab	
	chi2, p, dof, freq_tab = chi2_contingency(df1)
	# df2 = pd.DataFrame(freq_tab)

	# variance tab & adjusted residual table
	rd = len(freq_tab)
	cd = len(freq_tab[0])
	# var_table = np.zeros((rd, cd))
	d_table = np.zeros((rd, cd))
	for x in range(0, len(freq_tab)-1):
		for y in range(0, len(freq_tab[x])-1):
			var = (1-freq_tab[x,-1]/freq_tab[-1,-1])*(1-freq_tab[-1,y]/freq_tab[-1,-1])
			# var_table[x, y] = var
			e = freq_tab[x, y]
			o = df1.iloc[x, y]
			d = ((o - e)/np.sqrt(e))/np.sqrt(var)
			d_table[x, y] = d

	df2 = pd.DataFrame(d_table, columns=df1.columns, index=df1.index)

	return df2

np.random.seed(123)
N = 100
a1 = np.random.choice(['Y','N'], N)
a2 = np.random.choice([0., 1.], N)

# embed pattern (35% noise)
for x in range(0, len(a2)):
	if a1[x] == 'Y' and np.random.random() < 0.65:
		a2[x] = 1.

data = pd.DataFrame({'Loan_Status': a1,
                   'Credit_History':a2})
print('Input data (head 5 records):')
print(data.head(5))
print(data.describe())

d_table = cal_d(data, 'Loan_Status', 'Credit_History')
print('   ')
print('Adjusted residual table:')
print(d_table)
