# Customer-Churn-Prediction

# Tujuan Project
Pembuatan model untuk prediksi customer churn suatu ecommerce dari data telco sebuah perusahan.

# Background Project
Customer churn adalah kehilangan pelanggan dari suatu bisnis. Churn dihitung dari berapa banyak pelanggan meninggalkan bisnis Anda dalam waktu tertentu. Customer churn penting diketahui bisnis karena merupakan gambaran kesuksesan suatu bisnis dalam mempertahankan pelanggan.

# Library
library yang digunakan dalam project ini, yaitu:
1. pandas versi 1.4.2
2. numpy versi 1.21.5
3. scikit learn versi 1.1.2
4. feature engine packages versi 1.4.1

# Feature Engineering
Pada project dilakukan beberapa feature engineering seperti:
1. Feature selection mengunakan phik matriks
2. Handling outlier mengunakan metode capping
3. Feature scalling mengunakan metode standar scaller
4. Feature encoding mengunakan metode one hot encoder
5. Handling data imbalance mengunakan metode Smote over sampling

# Model
Model yang digunakan dalam project ini, Model Artificial Neural Network (ANN), dengan architecture model sebagai berikut:
input = tf.keras.Input(shape=(X_train_finalfix.shape[1],))
layers = tf.keras.layers.BatchNormalization()(input)
layers = tf.keras.layers.Dense(55, activation='relu', kernel_initializer='HeNormal',kernel_regularizer='l2')(layers)
layers = tf.keras.layers.BatchNormalization()(layers)
output = tf.keras.layers.Dense(out, activation='sigmoid')(layers)

model_func1 = tf.keras.Model(inputs=input, outputs=output)



