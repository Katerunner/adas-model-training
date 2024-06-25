from src.tsd_dfg.dataset_preparation import prepare_dataset

if __name__ == '__main__':
    prepare_dataset(data_folder='data',
                    dataset_name='dfg',
                    verbose=True,
                    master_dict_save=True,
                    master_dict_save_path="data/dfg/master.json",
                    config_yaml_save=True,
                    config_yaml_save_path="yolo_dataset_config/dfg_tsd.yaml",
                    max_workers=500)

