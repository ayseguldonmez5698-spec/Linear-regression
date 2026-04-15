import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression






X = np.random.rand(100, 1)
y=3+4*X+np.random.rand(100,1)
lin_reg=LinearRegression()
lin_reg.fit(X, y)






a1 = lin_reg.coef_[0][0]
a0 = lin_reg.intercept_[0]

print("eğim=", a1)
print("sabit=",a0)


plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="Data Points", alpha=0.6)

y_bulunan=a0+a1*X

plt.plot(X,y_bulunan,color="red", linewidth=2, label="Uydurulan eğri")

# %%
from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
import matplotlib.pyplot as plt


diabetes= load_diabetes()
diabetes_X,diabetes_y=load_diabetes(return_X_y=True)
diabetes_X=diabetes_X[: ,np.newaxis,2]

diabetes_X_train=diabetes_X[ : -20]
diabetes_X_test=diabetes_X[-20:]
diabetes_y_train=diabetes_y[:-20]
diabetes_y_test=diabetes_y[-20:]
lin_reg=LinearRegression()
lin_reg.fit(diabetes_X_train,diabetes_y_train)
diabetes_y_pred=lin_reg.predict(diabetes_X_test)
mse=mean_squared_error(diabetes_y_test, diabetes_y_pred)
print ("mse:",mse)
r2=r2_score(diabetes_y_test, diabetes_y_pred)
print("r2:", r2)


plt.figure(figsize=(10, 6))
# Gerçek verileri siyah noktalarla göster
plt.scatter(diabetes_X_test, diabetes_y_test, color="black", label="Gerçek Değerler")
# Tahmin edilen doğruyu mavi çizgiyle göster
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3, label="Regresyon Doğrusu")

plt.xlabel("BMI (Vücut Kitle Endeksi - Ölçeklenmiş)")
plt.ylabel("Hastalık İlerleme Skoru")
plt.title("Diyabet Veri Seti: BMI ve Hastalık İlişkisi")
plt.legend()
plt.show()



