import pandas as pd
import os

data_path = "./new_data"
feature_path = "./features"
merge_path = "./merge"

for user in os.listdir(data_path):
    # user = "gww"
    print(user)
    for condition in os.listdir(os.path.join(data_path, user)):
        print(condition)
        data_df = pd.read_csv(os.path.join(data_path, user, condition))
        feature_df = pd.read_csv(os.path.join(feature_path, user, condition))

        temp_labDelta = []
        temp_area = []
        # temp_eucDist = []
        temp_center_x = []
        temp_center_y = []
        data_idx = data_df["index"].to_list()
        for idx in data_idx:
            if len(feature_df[feature_df["index"] == idx]["labDelta"].to_list()) == 0:
                temp_labDelta.append(0)
                temp_area.append(0)
                temp_center_x.append(0)
                temp_center_y.append(0)
                # temp_eucDist.append(0)
            else:
                temp_labDelta.append(float(feature_df[feature_df["index"] == idx]["labDelta"]))
                temp_area.append(float(feature_df[feature_df["index"] == idx]["area"]))
                temp_center_x.append(float(feature_df[feature_df["index"] == idx]["center_x"]))
                temp_center_y.append(float(feature_df[feature_df["index"] == idx]["center_y"]))
                # temp_eucDist.append(float(feature_df[feature_df["index"] == idx]["eucDist"]))

        data_df["labDelta"] = temp_labDelta
        data_df["area"] = temp_area
        data_df["center_x"] = temp_center_x
        data_df["center_y"] = temp_center_x
        # data_df["eucDist"] = temp_eucDist

        fea = ["emd_ani_sal", "labDelta", "area", "center_x", "center_y"]
        # fea_MA = ["emd_ani_sal_MA", "labDelta_MA", "area_MA", "center_x_MA", "center_y_MA"]

        # data_df[fea_MA] = data_df[fea].rolling(10).mean()
        data_df.dropna(inplace=True)

        path = os.path.join(merge_path, user)
        if not os.path.exists(path):
            os.makedirs(path)
        data_df.to_csv(os.path.join(path, condition) + ".csv")
    # break