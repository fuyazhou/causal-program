import argparse

# 1.创建解释器
parser = argparse.ArgumentParser()

parser.add_argument('--train_data_path', type=str, default="train_data", help="训练文件路径")
parser.add_argument('--inference_data_path', type=str, default="inference_data", help="推理文件路径")
parser.add_argument('--feature_columns', type=str, default="", help="所有的特征list")

parser.add_argument('--treatment_columns_category', type=str, default="", help="枚举值的treatment，list")
parser.add_argument('--treatment_columns_continuous', type=str, default="", help="连续值的treatment，dict")

parser.add_argument('--treatment_columns_common', type=str, default="", help="预先定义的treatment，list")
parser.add_argument('--outcome_column', type=str, default="", help="标签列名，list")
parser.add_argument('--userid_column', type=str, default="user_id", help="userid列名，list")
args = parser.parse_args()

train_data_path = args.train_data_path
inference_data_path = args.inference_data_path
feature_columns = eval(args.feature_columns)
treatment_columns_category = eval(args.treatment_columns_category)
treatment_columns_continuous = eval(args.treatment_columns_continuous)
treatment_columns_common = eval(args.treatment_columns_common)
outcome_column = eval(args.outcome_column)
userid_column = eval(args.userid_column)

print(1)
