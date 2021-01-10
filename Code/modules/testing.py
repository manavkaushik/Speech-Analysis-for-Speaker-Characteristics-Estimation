# Testing the final Model

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def test(model, feat_arr_test_male, feat_arr_test_female, label_male_a, label_male_h,label_female_a, label_female_h):
    
    pred_male = model.predict(np.array(feat_arr_test_male))
    pred_female = model.predict(np.array(feat_arr_test_female))

    # y_pred_a = []
    # y_pred_h = []
    y_pred_male_a = []
    y_pred_female_a = []
    y_pred_male_h = []
    y_pred_female_h = []

    # pred_male = scaler.inverse_transform(pred_male) 
    # pred_female = scaler.inverse_transform(pred_female)

    for i in pred_male:
        y_pred_male_a.append(i[1])
        y_pred_male_h.append(i[0])
        
    for i in pred_female:
        y_pred_female_a.append(i[1])
        y_pred_female_h.append(i[0])

    y_pred_male_a = np.array(y_pred_male_a)
    y_pred_female_a = np.array(y_pred_female_a)
    y_pred_male_h = np.array(y_pred_male_h)
    y_pred_female_h = np.array(y_pred_female_h)   

    # y_pred_a = np.array(y_pred_a)
    # y_pred_h = np.array(y_pred_h)
    # labels_test_h = np.array(labels_test_h).astype(float)
    # labels_test_a = np.array(labels_test_a).astype(float)


    # mae_h = mean_absolute_error((labels_test_h), y_pred_h)
    # mse_h = mean_squared_error(labels_test_h, y_pred_h)
    # mse_a = mean_squared_error(labels_test_a, y_pred_a)
    # mae_a = mean_absolute_error(labels_test_a, y_pred_a)
    male_mae_a = mean_absolute_error(label_male_a, y_pred_male_a.reshape(-1))
    male_mse_a = mean_squared_error(label_male_a, y_pred_male_a.reshape(-1))
    female_mae_a = mean_absolute_error(label_female_a, y_pred_female_a.reshape(-1))
    female_mse_a = mean_squared_error(label_female_a, y_pred_female_a.reshape(-1))
    male_mae_h = mean_absolute_error(label_male_h, y_pred_male_h.reshape(-1))
    male_mse_h = mean_squared_error(label_male_h, y_pred_male_h.reshape(-1))
    female_mae_h = mean_absolute_error(label_female_h, y_pred_female_h.reshape(-1))
    female_mse_h= mean_squared_error(label_female_h, y_pred_female_h.reshape(-1))

    print('Results on TEST SET')
    print('----------------------------------------------------------------')
    print()
    print('FOR HEIGHT:\n')

    print('RMSE for Height (Male): {}'.format(np.sqrt(male_mse_h)))
    print('RMSE for Height (Female): {}'.format(np.sqrt(female_mse_h)))
    print('MAE for Height (Male): {}'.format(male_mae_h))
    print('MAE for Height (Female): {}'.format(female_mae_h))
    # print('RMSE for AGE Model (Female) with LSTM with Attention: {}'.format(np.sqrt(female_mse_a)))
    #print('MAE for HEIGHT Model with LSTM with Attention: {}'.format(mae_h))
    print()
    print()
    print('FOR AGE:\n')

    print('RMSE for Age  (Male): {}'.format(np.sqrt(male_mse_a)))
    print('RMSE for Age (Female): {}'.format(np.sqrt(female_mse_a)))
    print('MAE for Age (Male): {}'.format(male_mae_a))
    print('MAE for Age (Female): {}'.format(female_mae_a))
    print()
    print('----------------------------------------------------------------')
