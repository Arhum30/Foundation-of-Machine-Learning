#Muhammad Arhum Siddiqi, Muhammad Ahmed Khan Team 4

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# Excercise 1
print("Excercise 1")

# a)
mydata = {
    "Patient ID": ["000345", "000124", "001758", "000994", "001233", "001145", "000222"],
    "Age": [45, 60, 22, 38, 36, 77, 65],
    "Height [cm]": [167, 181, 158, 185, 164, 190, 180],
    "Weight [kg]": [67, 78, 57, 90, 72, 75, 110]
}
df = pd.DataFrame(mydata)

print(df)

# b)
df.loc[len(df)]=["001122", 51, 177, 81]

print("new row added")
print(df)

# c)
def norm(v):
    v_min = min(v)
    v_max = max(v)
    
    if v_max == v_min:
        return [0 for _ in v] 

        
    ret_arr = [(vm - v_min) / (v_max - v_min) for vm in v]    
    return ret_arr
    
# d)

print("Why is normalization important in Machine Learning?")
print("Normalization is important in Machine Learning as it scales features to a standard range,") 
print("improving model convergence and performance as well as preventing large-magnitude features")
print("from dominating small ones, this made sure there was always an even weight distribution.")

# e)

df["Age_norm"] = norm(df["Age"].tolist())
df["Height_norm"] = norm(df["Height [cm]"].tolist())
df["Weight_norm"] = norm(df["Weight [kg]"].tolist())

print(df)

# f)

df.to_csv("patients.csv", index=False)

# g)

average_age = df.Age.mean()
young_patients_df = df[df.Age < average_age]
print(average_age)
print(young_patients_df)

# Excercise 2
print("Excercise 2")

# a)

df2 = pd.read_csv("patients.csv")
df2.head()

# b)

heights = df2['Height [cm]']

plt.hist(heights, bins=range(heights.min()-1, heights.max()+1), edgecolor='black')
plt.title('Heights')
plt.xticks(range(heights.min()-1, heights.max()+1), rotation=45, fontsize=6)
plt.xlabel('Height')
plt.ylabel('Frequesncy')
plt.show()

# c)

# (Height vs Age)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)  
plt.scatter(df2['Age_norm'], df2['Height_norm'], color='blue')
plt.title('Height vs Age')
plt.xlabel('Age')
plt.ylabel('Height')

# (Weight vs Age)
plt.subplot(1, 2, 2)  
plt.scatter(df2['Age_norm'], df2['Weight_norm'], color='green')
plt.title('Weight vs Age')
plt.xlabel('Age')
plt.ylabel('Weight')


plt.show()

# d)


def p_corr(a, b):
    a = a.tolist()
    b = b.tolist() 
    a_mean = sum(a) /len(a)
    b_mean = sum(b) /len(a)

    numerator = sum((a[i] - a_mean) * (b[i] - b_mean) for i in range(len(a)))
    denominator = math.sqrt(sum((a[i] - a_mean) ** 2 for i in range(len(a))) * sum((b[i] - b_mean) ** 2 for i in range(len(a))))

    return numerator /denominator
    
p_corr_ageVheight = p_corr(df2["Age"], df2["Height [cm]"])
p_corr_ageVweight = p_corr(df2["Age"], df2["Weight [kg]"])

print("Correlation Age vs Height:", p_corr_ageVheight)
print("Age and height have a strong positive correlation (0.79), meaning older individuals tend to be taller.")

print("Correlation Age vs Weight:", p_corr_ageVweight)
print("Age and weight have a moderate positive correlation (0.50), indicating weight increases with age but with more variation.")

# Excercise 3
print("Excercise 3")

# a)

v1 = np.array([0.2, 0.1, 0.4, -0.4])
v2 = np.array([-0.1, -0.1, 0.8, 0.5])

# b)

def L1norm(v1, v2):
    return np.sum(np.abs(v1 - v2))
    
def L2norm(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))
    
def cosine_sim(v1, v2):
    num_prod = np.dot(v1, v2)
    den = (np.linalg.norm(v1) * np.linalg.norm(v2))
    return  num_prod/den
    
# c)
l1 = L1norm(v1, v2)
l2 = L2norm(v1, v2)
cosine = cosine_sim(v1, v2)

print("L1 Norm:", l1)
print("L2 Norm:", l2)
print("Cosine Similarity:", cosine)
