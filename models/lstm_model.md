Caricamento file .npz...
Totale sequenze: 17727
Preprocessing e Normalizzazione...
Forma finale X: (17727, 150, 51), y: (17727,)
Train: (11344, 150, 51), Val: (2837, 150, 51), Test: (3546, 150, 51)
Pesi calcolati (0=NonViolento, 1=Violento): {0: np.float64(1.0097917037564537), 1: np.float64(0.9903963680810197)}
Creazione modello Bidirectional LSTM...
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ masking (Masking)                    │ (None, 150, 51)             │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bidirectional (Bidirectional)        │ (None, 150, 128)            │          59,392 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 150, 128)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bidirectional_1 (Bidirectional)      │ (None, 64)                  │          41,216 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 64)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization                  │ (None, 64)                  │             256 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 32)                  │           2,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_2 (Dropout)                  │ (None, 32)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 1)                   │              33 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 102,977 (402.25 KB)
 Trainable params: 102,849 (401.75 KB)
 Non-trainable params: 128 (512.00 B)
Avvio training...
Epoch 1/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 61s 156ms/step - accuracy: 0.6344 - loss: 0.6501 - val_accuracy: 0.6630 - val_loss: 0.6067 - learning_rate: 0.0010
Epoch 2/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 54s 153ms/step - accuracy: 0.6823 - loss: 0.5883 - val_accuracy: 0.7088 - val_loss: 0.5633 - learning_rate: 0.0010
Epoch 3/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 54s 153ms/step - accuracy: 0.7193 - loss: 0.5466 - val_accuracy: 0.7303 - val_loss: 0.5464 - learning_rate: 0.0010
Epoch 4/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 54s 153ms/step - accuracy: 0.7414 - loss: 0.5122 - val_accuracy: 0.7427 - val_loss: 0.5137 - learning_rate: 0.0010
Epoch 5/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 54s 153ms/step - accuracy: 0.7558 - loss: 0.5012 - val_accuracy: 0.7511 - val_loss: 0.4993 - learning_rate: 0.0010
Epoch 6/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 55s 156ms/step - accuracy: 0.7722 - loss: 0.4746 - val_accuracy: 0.7642 - val_loss: 0.4936 - learning_rate: 0.0010
Epoch 7/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 57s 160ms/step - accuracy: 0.7831 - loss: 0.4573 - val_accuracy: 0.7649 - val_loss: 0.4912 - learning_rate: 0.0010
Epoch 8/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 55s 156ms/step - accuracy: 0.7898 - loss: 0.4401 - val_accuracy: 0.7853 - val_loss: 0.4842 - learning_rate: 0.0010
Epoch 9/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 55s 156ms/step - accuracy: 0.7932 - loss: 0.4344 - val_accuracy: 0.7698 - val_loss: 0.4683 - learning_rate: 0.0010
Epoch 10/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 56s 157ms/step - accuracy: 0.8011 - loss: 0.4265 - val_accuracy: 0.7582 - val_loss: 0.4806 - learning_rate: 0.0010
Epoch 11/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 56s 157ms/step - accuracy: 0.8049 - loss: 0.4131 - val_accuracy: 0.7346 - val_loss: 0.5166 - learning_rate: 0.0010
Epoch 12/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 56s 157ms/step - accuracy: 0.8109 - loss: 0.4094 - val_accuracy: 0.7956 - val_loss: 0.4461 - learning_rate: 0.0010
Epoch 13/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 56s 158ms/step - accuracy: 0.8116 - loss: 0.3994 - val_accuracy: 0.7765 - val_loss: 0.4615 - learning_rate: 0.0010
Epoch 14/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 56s 158ms/step - accuracy: 0.8219 - loss: 0.3864 - val_accuracy: 0.8030 - val_loss: 0.4330 - learning_rate: 0.0010
Epoch 15/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 56s 158ms/step - accuracy: 0.8204 - loss: 0.3827 - val_accuracy: 0.7769 - val_loss: 0.4599 - learning_rate: 0.0010
Epoch 16/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 56s 158ms/step - accuracy: 0.8173 - loss: 0.3868 - val_accuracy: 0.7927 - val_loss: 0.4466 - learning_rate: 0.0010
Epoch 17/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 0s 145ms/step - accuracy: 0.8144 - loss: 0.3977
Epoch 17: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
355/355 ━━━━━━━━━━━━━━━━━━━━ 56s 157ms/step - accuracy: 0.8010 - loss: 0.4210 - val_accuracy: 0.7776 - val_loss: 0.4661 - learning_rate: 0.0010
Epoch 18/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 56s 157ms/step - accuracy: 0.8361 - loss: 0.3605 - val_accuracy: 0.7899 - val_loss: 0.4680 - learning_rate: 5.0000e-04
Epoch 19/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 56s 159ms/step - accuracy: 0.8362 - loss: 0.3492 - val_accuracy: 0.7896 - val_loss: 0.4553 - learning_rate: 5.0000e-04
Epoch 20/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 0s 146ms/step - accuracy: 0.8499 - loss: 0.3362
Epoch 20: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
355/355 ━━━━━━━━━━━━━━━━━━━━ 56s 158ms/step - accuracy: 0.8434 - loss: 0.3398 - val_accuracy: 0.7966 - val_loss: 0.4342 - learning_rate: 5.0000e-04
Epoch 21/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 56s 159ms/step - accuracy: 0.8514 - loss: 0.3246 - val_accuracy: 0.7973 - val_loss: 0.4464 - learning_rate: 2.5000e-04
Epoch 22/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 54s 153ms/step - accuracy: 0.8523 - loss: 0.3158 - val_accuracy: 0.7998 - val_loss: 0.4589 - learning_rate: 2.5000e-04
Epoch 23/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 0s 141ms/step - accuracy: 0.8601 - loss: 0.3028
Epoch 23: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.
355/355 ━━━━━━━━━━━━━━━━━━━━ 54s 153ms/step - accuracy: 0.8558 - loss: 0.3079 - val_accuracy: 0.7991 - val_loss: 0.4677 - learning_rate: 2.5000e-04
Epoch 24/100
355/355 ━━━━━━━━━━━━━━━━━━━━ 53s 151ms/step - accuracy: 0.8596 - loss: 0.2969 - val_accuracy: 0.8037 - val_loss: 0.4665 - learning_rate: 1.2500e-04

--- Report sul Test Set (Binario) ---
111/111 ━━━━━━━━━━━━━━━━━━━━ 6s 52ms/step
              precision    recall  f1-score   support

Non Violento       0.86      0.72      0.78      1756
    Violento       0.76      0.89      0.82      1790

    accuracy                           0.80      3546
   macro avg       0.81      0.80      0.80      3546
weighted avg       0.81      0.80      0.80      3546


17727

14004
3540
183