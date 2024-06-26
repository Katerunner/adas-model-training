import os

import copy
import json
import glob
import tqdm
import concurrent.futures

import numpy as np
import pandas as pd

from PIL import Image

YAML_HAT = """
path: dfg/
train: 'train/images'
val: 'val/images'

# class names
names: 
"""

DEFAULT_DATASET_PATH = "datasets\\dfg"


def explore_data(data_folder: str = 'data', dataset_name: str = 'dfg', verbose: bool = False):
    all_images_folder_path = os.path.join(data_folder, dataset_name, "images")
    all_images_paths = glob.glob(os.path.join(all_images_folder_path, "*.jpg"))

    with open(f"{data_folder}/{dataset_name}/train.json") as f:
        trn_json = json.load(f)

    with open(f"{data_folder}/{dataset_name}/test.json") as f:
        val_json = json.load(f)

    all_filenames = set(map(lambda x: x.split("\\")[-1], all_images_paths))
    trn_filenames = set([i['file_name'] for i in trn_json['images']])
    val_filenames = set([i['file_name'] for i in val_json['images']])

    metrics = {
        "num_images": len(all_images_paths),
        "num_names": len(all_filenames),
        "num_trn": len(trn_filenames),
        "num_val": len(val_filenames),
        "num_all": len((trn_filenames | val_filenames) & all_filenames)
    }

    if verbose:
        print("Number of images:", metrics["num_images"])
        print("Number of all names:", metrics["num_names"])
        print("Number of available train names:", metrics["num_trn"])
        print("Number of available val names:", metrics["num_val"])
        print("Total number of available names:", metrics["num_all"])

    return_assets = {
        "all_images_folder_path": all_images_folder_path,
        "all_images_paths": all_images_paths,
        "train_json": trn_json,
        "val_json": val_json,
        "filenames": {"trn": trn_filenames, "val": val_filenames, "all": all_filenames},
        "metrics": metrics
    }

    return return_assets


def prepare_set_annotations_dict(annotation_json: dict, include_segmentation: bool = False):
    annotations_dict = {}

    for ann in annotation_json:
        ann_no_image_id = copy.copy(ann)
        ann_no_image_id.pop('image_id')
        if not include_segmentation:
            ann_no_image_id.pop('segmentation', None)

        if ann['image_id'] not in annotations_dict:
            annotations_dict[ann['image_id']] = [ann_no_image_id]
        else:
            annotations_dict[ann['image_id']].append(ann_no_image_id)

    return annotations_dict


def _add_images_to_master_dict(image_data_list: list, set_type: str, master_dict: dict):
    for image_data in image_data_list:
        master_dict[set_type][image_data['id']] = image_data
        master_dict[set_type][image_data['id']]['annotations'] = None


def _update_annotations(annotations_dict: dict, set_type: str, master_dict: dict):
    for im_id, annotation in annotations_dict.items():
        if im_id in master_dict[set_type]:
            master_dict[set_type][im_id]['annotations'] = annotation


def _remove_images_without_annotations(set_type: str, master_dict: dict):
    im_ids_to_delete = [im_id for im_id, data in master_dict[set_type].items() if data['annotations'] is None]
    for im_id in im_ids_to_delete:
        del master_dict[set_type][im_id]


def prepare_master_dict(
        train_json: dict,
        val_json: dict,
        verbose: bool = False,
        save: bool = True,
        save_path: str = "data/dfg/master.json") -> dict:
    train_annotations = prepare_set_annotations_dict(train_json['annotations'])
    val_annotations = prepare_set_annotations_dict(val_json['annotations'])

    master_dict = {'train': {}, 'val': {}}

    _add_images_to_master_dict(train_json['images'], 'train', master_dict)
    _add_images_to_master_dict(val_json['images'], 'val', master_dict)

    _update_annotations(train_annotations, 'train', master_dict)
    _update_annotations(val_annotations, 'val', master_dict)

    _remove_images_without_annotations('train', master_dict)
    _remove_images_without_annotations('val', master_dict)

    if verbose:
        print("Number of train labels:", len(list(master_dict['train'].keys())))
        print("Number of val labels:", len(list(master_dict['val'].keys())))

    if save:
        with open(save_path, "w") as f:
            json.dump(master_dict, f)

    return master_dict


def load_class_config_with_effective_id(file_path: str, verbose: bool = False) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    effective_id = 0
    effective_ids = []

    for enabled in df['enabled']:
        if enabled == 1:
            effective_ids.append(effective_id)
            effective_id += 1
        else:
            effective_ids.append(None)

    df['effective_id'] = effective_ids

    if verbose:
        used_classes = df['enabled'].sum()
        disabled_classes = len(df) - used_classes
        print(f"Classes used: {used_classes}")
        print(f"Classes disabled: {disabled_classes}")

    return df


def generate_yolo_training_config_yaml(df: pd.DataFrame, save: bool = True, save_path: str = "dfg_tsd.yaml") -> str:
    yaml_text = YAML_HAT
    for _, row in df[df['enabled'] == 1].iterrows():
        yaml_text += f"  {int(row['effective_id'])}: {row['code']}\n"

    yaml_text = yaml_text.strip()

    if save:
        with open(save_path, "w") as f:
            f.write(yaml_text)

    return yaml_text.strip()


def prepare_label_text(image_info, class_config_df):
    text = ""
    for obj in image_info['annotations']:
        if obj.get('ignore', False):
            continue

        ci = obj['category_id']
        effective_id = class_config_df.loc[class_config_df['id'] == ci, 'effective_id'].values[0]

        if pd.isna(effective_id):
            continue

        effective_id = int(effective_id)
        cx = np.clip((obj['bbox'][0] + obj['bbox'][2] / 2) / image_info['width'], 0, 1)
        cy = np.clip((obj['bbox'][1] + obj['bbox'][3] / 2) / image_info['height'], 0, 1)
        bw = np.clip(obj['bbox'][2] / image_info['width'], 0, 1)
        bh = np.clip(obj['bbox'][3] / image_info['height'], 0, 1)

        text += f"{effective_id} {cx:.5f} {cy:.5f} {bw:.5f} {bh:.5f}\n"
    return text.strip()


def process_image_and_label(set_type, image_info, class_config_df, all_images_folder_path, datasets_path):
    image = Image.open(os.path.join(all_images_folder_path, image_info['file_name']))
    image_path = os.path.join(datasets_path, set_type, 'images', image_info['file_name'].replace(".jpg", ".png"))
    image.save(image_path)
    image.close()

    # Save label
    label = prepare_label_text(image_info, class_config_df)
    label_path = os.path.join(datasets_path, set_type, 'labels', image_info['file_name'].replace(".jpg", ".txt"))
    with open(label_path, "w") as f:
        f.write(label)


def process_set(master_dict, class_config_df, set_type, all_images_folder_path, datasets_path, max_workers=100):
    set_master = master_dict[set_type]
    futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for im_id in list(set_master.keys()):  # Adjust as needed
            image_info = set_master[im_id]
            futures.append(
                executor.submit(
                    process_image_and_label,
                    set_type,
                    image_info,
                    class_config_df,
                    all_images_folder_path,
                    datasets_path
                )
            )

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0):
            future.result()


def prepare_dataset(
        data_folder: str = 'data',
        dataset_name: str = 'dfg',
        verbose: bool = True,
        class_config_path: str = 'configurations/tsd_dfg_classes.csv',
        master_dict_save: bool = True,
        master_dict_save_path: str = "data/dfg/master.json",
        config_yaml_save: bool = True,
        config_yaml_save_path: str = "yolo_dataset_config/dfg_tsd.yaml",
        datasets_path: str = DEFAULT_DATASET_PATH,
        max_workers: int = 500):
    if verbose:
        print("Exploring dataset...")

    dataset_assets = explore_data(data_folder=data_folder, dataset_name=dataset_name, verbose=verbose)

    if verbose:
        print("Generating master json...")

    master_dict = prepare_master_dict(
        train_json=dataset_assets['train_json'],
        val_json=dataset_assets['val_json'],
        verbose=verbose,
        save=master_dict_save,
        save_path=master_dict_save_path,
    )

    if verbose:
        print("Generating training configuration...")

    class_config_df = load_class_config_with_effective_id(
        file_path=class_config_path,
        verbose=verbose)

    train_config_yaml_text = generate_yolo_training_config_yaml(
        df=class_config_df,
        save=config_yaml_save,
        save_path=config_yaml_save_path,
    )

    if verbose:
        print("Processing train set...")
    process_set(
        master_dict=master_dict,
        set_type='train',
        all_images_folder_path=dataset_assets['all_images_folder_path'],
        datasets_path=datasets_path,
        class_config_df=class_config_df,
        max_workers=max_workers)

    if verbose:
        print("Processing val set...")

    process_set(
        master_dict=master_dict,
        set_type='val',
        all_images_folder_path=dataset_assets['all_images_folder_path'],
        datasets_path=datasets_path,
        class_config_df=class_config_df,
        max_workers=max_workers)

    if verbose:
        print("Done!")

    return train_config_yaml_text
