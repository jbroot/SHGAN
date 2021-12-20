import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


nnType = np.asarray( ["MLP", "CNN"] * 3)
nnType = nnType[np.newaxis,:]
rmFeatOrder = np.asarray( ["Sensor"] * 2 + ["Signal"] * 2 + ["Time Differential"] * 2)
rmFeatOrder = rmFeatOrder[np.newaxis,:]

data = np.concatenate((nnType, rmFeatOrder), axis=0)

df = pd.DataFrame(data.T, columns=["Neural Networks", "Removed Feature"])

vals = [0.00337356, 0.00033109, 0.00107283, 7.92982464e-05, 0.00201232, 0.0005909]
df["Validation Categorical Crossentropy"] = vals

ax = sns.factorplot(x="Neural Networks", y="Validation Categorical Crossentropy", hue="Removed Feature",
                data=df, kind='bar')

plt.title("Feature Sensitivity")
# ax.set_titles("Feature Sensitivity")
#
plt.subplots_adjust(top=.9, bottom=.1)

plt.savefig("C:\\Users\\wind0\\Documents\\phd\\Research\\phdCode\\Sum21\\misc\\tstr\FeatureSensitivity.png")

plt.show()