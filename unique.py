import pandas as pd

depth_test = pd.read_csv('/tmp/EcoDepth/svr_depth_summary_val_with_predictions.csv')
depth_test_crop = pd.read_csv('/tmp/EcoDepth/svr_depth_summary_val_with_predictions1.csv')

unique_rows = depth_test[~depth_test['annotation_id'].isin(depth_test_crop['annotation_id'])]

average_pred_Intradepth = depth_test_crop['pred_Intradepth'].mean()

unique_rows['pred_Intradepth'] = average_pred_Intradepth

result = pd.concat([depth_test_crop, unique_rows], ignore_index=True)

result.to_csv('depth_test_unique.csv', index=False)