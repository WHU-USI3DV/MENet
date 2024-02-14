import argparse
import os.path as osp

from data_converter import nuscenes_converter as nuscenes_converter
from data_converter import lyft_converter as lyft_converter
from data_converter.create_gt_database import create_groundtruth_database

def nuscenes_data_prep(
    root_path,
    info_prefix,
    version,
    dataset_name,
    out_dir,
    max_sweeps=10,
    load_augmented=None,
):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    if load_augmented is None:
        # otherwise, infos must have been created, we just skip.
        nuscenes_converter.create_nuscenes_infos(
            root_path, info_prefix, version=version, max_sweeps=max_sweeps
        )

        if version == "v1.0-test":
            info_test_path = osp.join(root_path, f"{info_prefix}_infos_test.pkl")
            nuscenes_converter.export_2d_annotation(root_path, info_test_path, version=version)
            return

        info_train_path = osp.join(root_path, f"{info_prefix}_infos_train.pkl")
        info_val_path = osp.join(root_path, f"{info_prefix}_infos_val.pkl")
        nuscenes_converter.export_2d_annotation(root_path, info_train_path, version=version)
        nuscenes_converter.export_2d_annotation(root_path, info_val_path, version=version)

    create_groundtruth_database(
        dataset_name,
        root_path,
        info_prefix,
        f"{out_dir}/{info_prefix}_infos_train.pkl",
        load_augmented=load_augmented,
    )

def lyft_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to Lyft dataset.

    Related data consists of '.pkl' files recording basic infos.
    Although the ground truth database and 2D annotations are not used in
    Lyft, it can also be generated like nuScenes.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Defaults to 10.
    """
    lyft_converter.create_lyft_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)

parser = argparse.ArgumentParser(description="Data converter arg parser")
parser.add_argument("dataset", metavar="kitti", help="name of the dataset")
parser.add_argument(
    "--root-path",
    type=str,
    default="./data/kitti",
    help="specify the root path of dataset",
)
parser.add_argument(
    "--version",
    type=str,
    default="v1.0",
    required=False,
    help="specify the dataset version, no need for kitti",
)
parser.add_argument(
    "--max-sweeps",
    type=int,
    default=10,
    required=False,
    help="specify sweeps of lidar per example",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default="./data/kitti",
    required="False",
    help="name of info pkl",
)
parser.add_argument("--extra-tag", type=str, default="kitti")
parser.add_argument("--painted", default=False, action="store_true")
parser.add_argument("--virtual", default=False, action="store_true")
parser.add_argument(
    "--workers", type=int, default=4, help="number of threads to be used"
)
args = parser.parse_args()

if __name__ == "__main__":
    load_augmented = None
    if args.virtual:
        if args.painted:
            load_augmented = "mvp"
        else:
            load_augmented = "pointpainting"

    if args.dataset == "nuscenes" and args.version != "v1.0-mini":
        train_version = f"{args.version}-trainval"
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            load_augmented=load_augmented,
        )
        test_version = f"{args.version}-test"
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            load_augmented=load_augmented,
        )
    elif args.dataset == "nuscenes" and args.version == "v1.0-mini":
        train_version = f"{args.version}"
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            load_augmented=load_augmented,
        )
    elif args.dataset == 'lyft':
        train_version = f'{args.version}-train'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            max_sweeps=args.max_sweeps)

        create_groundtruth_database(
            dataset_class_name="LyftDataset",
            data_path=args.root_path,
            info_prefix=args.extra_tag,
            info_path=f"{args.out_dir}/{args.extra_tag}_infos_train.pkl",
            load_augmented=load_augmented,
        )
