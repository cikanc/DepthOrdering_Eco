import pandas as pd

depth_test_unique = pd.read_csv('depth_test_unique.csv')
# depth_test = pd.read_csv('/tmp/EcoDepth/svr_depth_summary_val_with_predictions.csv')
depth_test = pd.read_csv('./svr_depth_summary_val_with_predictions.csv')

merged_df = pd.merge(depth_test_unique, depth_test, on=['annotation_id', 'image_id', 'category_id'], suffixes=('_unique', '_original'))

result_df = merged_df[['annotation_id', 'image_id', 'category_id']]
result_df['pred_Interdepth'] = merged_df[['pred_Interdepth_unique', 'pred_Interdepth_original']].mean(axis=1)
result_df['pred_Intradepth'] = merged_df[['pred_Intradepth_unique', 'pred_Intradepth_unique']].mean(axis=1)

result_df.to_csv('result.csv', index=False)