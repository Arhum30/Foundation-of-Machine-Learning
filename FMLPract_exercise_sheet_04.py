""" 
Muhammad Arhum Siddiqui, Muhammad Ahmed Khan
"""

# #%% Exercise 1
# import numpy as np
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import make_pipeline

# # (a)
# np.random.seed(42)
# M = 20
# x = np.random.uniform(0, 30, M)
# a, b, c = 0.25, -5, 0.2
# epsilon = np.random.normal(0, 4.5, M)
# y = a * x**2 + b * x + c + epsilon
# X = x.reshape(-1, 1)
# import matplotlib.pyplot as plt

# # Visualize the generated data and the true quadratic curve
# # x_curve = np.linspace(0, 30, 500)
# # y_curve = a * x_curve**2 + b * x_curve + c

# # plt.figure(figsize=(8, 5))
# # plt.scatter(x, y, label='Generated data')
# # plt.plot(x_curve, y_curve, label='True quadratic curve')
# # plt.xlabel('x')
# # plt.ylabel('y')
# # plt.title('Generated data and true function')
# # plt.legend()
# # plt.show()

# # (b)
# def PolynomialRegression(degree=2):
#     return make_pipeline(PolynomialFeatures(degree), LinearRegression())
# models = {
#     'Linear (deg=1)': PolynomialRegression(1),
#     'Quadratic (deg=2)': PolynomialRegression(2),
#     'High Degree (deg=10)': PolynomialRegression(10)
# }
# xfit = np.linspace(0, 30, 5000).reshape(-1, 1)
# predictions = {}
# for name, model in models.items():
#     model.fit(X, y)
#     predictions[name] = model.predict(xfit)


# # (c)
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6))
# plt.scatter(x, y, color='black', label='Data')
# for name, ypred in predictions.items():
#     plt.plot(xfit, ypred, label=name)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Polynomial Regression Models')
# plt.legend()
# plt.show()

# # (d)
# from sklearn.metrics import mean_squared_error

# for name, model in models.items():
#     y_pred_train = model.predict(X)
#     mse = mean_squared_error(y, y_pred_train)
#     print(f"{name} MSE: {mse:.2f}")

#%% Exercise 1 redone
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# (a)
np.random.seed(42)
M = 20
x = np.random.uniform(0, 30, M)
a, b, c = 0.25, -5, 0.2
epsilon = np.random.normal(0, 4.5, M)
y = a * x**2 + b * x + c + epsilon
X = x.reshape(-1, 1)

# (b)
def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression(**kwargs)
    )

xfit = np.linspace(0, 30, 5000).reshape(-1, 1)

linear_model = PolynomialRegression(1).fit(X, y)
yfit_linear = linear_model.predict(xfit)

quadratic_model = PolynomialRegression(2).fit(X, y)
yfit_quadratic = quadratic_model.predict(xfit)

high_degree_model = PolynomialRegression(10).fit(X, y)
yfit_high_degree = high_degree_model.predict(xfit)

# (c)
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='black', label='Data')
plt.plot(xfit, yfit_linear, label='Linear (deg=1)')
plt.plot(xfit, yfit_quadratic, label='Quadratic (deg=2)')
plt.plot(xfit, yfit_high_degree, label='High Degree (deg=10)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression Models')
plt.legend()
plt.show()

# (d)
mse_linear = mean_squared_error(y, linear_model.predict(X))
mse_quadratic = mean_squared_error(y, quadratic_model.predict(X))
mse_high_degree = mean_squared_error(y, high_degree_model.predict(X))

print(f"Linear (deg=1) MSE: {mse_linear:.2f}")
print(f"Quadratic (deg=2) MSE: {mse_quadratic:.2f}")
print(f"High Degree (deg=10) MSE: {mse_high_degree:.2f}")

print("""The linear model is underfitting because it is too simple to capture
       the curved quadratic relationship in the data. It can only fit a straight line,
       so it misses the main pattern and gives a relatively poor fit.
       The 10th-degree polynomial model is overfitting because it is too flexible for this small noise
       dataset. Instead of only learning the true quadratic trend, it also starts fitting random noise
       and fluctuations in the training points.""")
#%% Exercise 2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

# (a) theory in pdf
# (b) theory in pdf
# (c) theory in pdf

# (d)
y_hat1 = np.array([0.1, 0.8, 0.62, 0.3, 0.45, 0.2, 0.2, 0.9, 0.6, 0.2, 0.1, 0.8])
y_hat2 = np.array([0.2, 0.9, 0.45, 0.9, 0.1, 0.2, 0.55, 0.85, 0.15, 0.1, 0.3, 0.7])
y_true = np.array([0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1])

# (e)
fpr1, tpr1, _ = roc_curve(y_true, y_hat1)
fpr2, tpr2, _ = roc_curve(y_true, y_hat2)

roc_auc1 = auc(fpr1, tpr1)
roc_auc2 = auc(fpr2, tpr2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.plot(fpr1, tpr1, color='blue', lw=2, label=f'Model 1 (AUC = {roc_auc1:.2f})')
ax1.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Line of no discrimination
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve for Model 1')
ax1.legend(loc="lower right")

ax2.plot(fpr2, tpr2, color='green', lw=2, label=f'Model 2 (AUC = {roc_auc2:.2f})')
ax2.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Line of no discrimination
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve for Model 2')
ax2.legend(loc="lower right")

plt.tight_layout()
plt.show()

# (f)
y_pred1 = (y_hat1 >= 0.5).astype(int)
cm1 = confusion_matrix(y_true, y_pred1)
print(f'Confusion Matrix for Model 1:\n{cm1}')

y_pred2 = (y_hat2 >= 0.5).astype(int)
cm2 = confusion_matrix(y_true, y_pred2)
print(f'Confusion Matrix for Model 2:\n{cm2}')

sensitivity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
specificity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
f_score1 = 2 * (sensitivity1 * specificity1) / (sensitivity1 + specificity1)

sensitivity2 = cm2[1, 1] / (cm2[1, 0] + cm2[1, 1])
specificity2 = cm2[0, 0] / (cm2[0, 0] + cm2[0, 1])
f_score2 = 2 * (sensitivity2 * specificity2) / (sensitivity2 + specificity2)

print(f'Sensitivity for Model 1: {sensitivity1:.2f}')
print(f'Specificity for Model 1: {specificity1:.2f}')
print(f'F-Score for Model 1: {f_score1:.2f}')

print(f'Sensitivity for Model 2: {sensitivity2:.2f}')
print(f'Specificity for Model 2: {specificity2:.2f}')
print(f'F-Score for Model 2: {f_score2:.2f}')

# (g) based on results
print("\nComparative Analysis:")
if roc_auc1 > roc_auc2:
    print("Model 1 has a better AUC.")
else:
    print("Model 2 has a better AUC.")

if f_score1 > f_score2:
    print("Model 1 has a better F-Score.")
elif f_score1 < f_score2:
    print("Model 2 has a better F-Score.")
else:
    print("Both models have the same F-Score.")


# %%
