import os
import re
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset


def read_rgb(path):
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.tile(img[..., None], (1, 1, 3))
    return img[..., :3]


def read_pfm(path):
    with open(path, "rb") as f:
        header = f.readline().decode("utf-8").rstrip()
        if header not in {"PF", "Pf"}:
            raise ValueError(f"Not a PFM file: {path}")
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", f.readline().decode("utf-8"))
        if dim_match is None:
            raise ValueError(f"Malformed PFM header: {path}")
        width, height = map(int, dim_match.groups())
        scale = float(f.readline().decode("utf-8").rstrip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")
    shape = (height, width, 3) if header == "PF" else (height, width)
    data = np.reshape(data, shape)
    return np.flipud(data).astype(np.float32)


def resolve_split_root(dataset_root, split_dir_name="train"):
    candidates = [
        os.path.join(dataset_root, split_dir_name),
        os.path.join(dataset_root, split_dir_name.capitalize()),
        dataset_root,
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Cannot find split root under {dataset_root}")


def load_scene_split(split_file):
    with open(split_file, "r") as f:
        return yaml.safe_load(f)


def ensure_exists(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


@dataclass
class StereoSample:
    dataset: str
    setup: str
    scene: str
    left_file: str
    right_file: str
    gt_file: str | None
    left_name: str
    noc_file: str | None = None


class BaseStereoDataset(Dataset):
    def __init__(self, dataset_name, scale=1.0):
        self.dataset_name = dataset_name
        self.scale = float(scale)
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def load_disp(self, sample):
        raise NotImplementedError

    def load_noc(self, sample, gt):
        return None

    def apply_dataset_transforms(self, sample, img0, img1, gt, valid, noc):
        return img0, img1, gt, valid, noc

    def apply_global_scale(self, img0, img1, gt, valid, noc):
        if self.scale == 1.0:
            return img0, img1, gt, valid, noc
        new_w = max(1, int(round(img0.shape[1] * self.scale)))
        new_h = max(1, int(round(img0.shape[0] * self.scale)))
        img0 = cv2.resize(img0, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if gt is not None:
            gt = cv2.resize(gt, (new_w, new_h), interpolation=cv2.INTER_LINEAR) * self.scale
            valid = cv2.resize(valid.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 0
            if noc is not None:
                noc = cv2.resize(noc.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 0
        return img0, img1, gt, valid, noc

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img0 = read_rgb(sample.left_file)
        img1 = read_rgb(sample.right_file)
        gt = self.load_disp(sample)
        if gt is None:
            raise ValueError(f"{self.dataset_name} sample has no GT disparity: {sample.left_file}")
        gt = np.asarray(gt, dtype=np.float32)
        valid = np.isfinite(gt) & (gt > 0.0)
        noc = self.load_noc(sample, gt)
        if noc is not None:
            noc = np.asarray(noc).astype(bool)

        img0, img1, gt, valid, noc = self.apply_dataset_transforms(sample, img0, img1, gt, valid, noc)
        img0, img1, gt, valid, noc = self.apply_global_scale(img0, img1, gt, valid, noc)

        data = {
            "dataset": sample.dataset,
            "setup": sample.setup,
            "scene": sample.scene,
            "name": sample.left_name,
            "left_file": sample.left_file,
            "right_file": sample.right_file,
            "gt_file": sample.gt_file,
            "left": torch.from_numpy(img0).permute(2, 0, 1).float(),
            "right": torch.from_numpy(img1).permute(2, 0, 1).float(),
            "disp": torch.from_numpy(gt).float().unsqueeze(0),
            "valid": torch.from_numpy(valid.astype(np.float32)).unsqueeze(0),
        }
        if noc is not None:
            data["noc"] = torch.from_numpy(noc.astype(np.float32)).unsqueeze(0)
            data["noc_file"] = sample.noc_file
        return data


class BoosterStereoDataset(BaseStereoDataset):
    def __init__(
        self,
        booster_root,
        split_file,
        subset,
        setups,
        scale=1.0,
        balanced_input_scale=0.25,
        unbalanced_input_scale=1.0,
        match_unbalanced_left_to_right=True,
        split_dir_name="train",
    ):
        super().__init__(dataset_name="booster", scale=scale)
        self.booster_root = booster_root
        self.split_cfg = load_scene_split(split_file)
        self.setups = list(setups)
        self.balanced_input_scale = float(balanced_input_scale)
        self.unbalanced_input_scale = float(unbalanced_input_scale)
        self.match_unbalanced_left_to_right = bool(match_unbalanced_left_to_right)
        self.split_root = resolve_split_root(booster_root, split_dir_name)

        if subset not in {"train", "val"}:
            raise ValueError(f"Unsupported subset: {subset}")
        scene_names = self.split_cfg["train_scenes"] if subset == "train" else self.split_cfg["val_scenes"]
        self.samples = self._collect_samples(scene_names)

    def _collect_samples(self, scene_names):
        samples = []
        for setup in self.setups:
            for scene in scene_names:
                scene_dir = os.path.join(self.split_root, setup, scene)
                left_dir = os.path.join(scene_dir, "camera_00")
                right_dir = os.path.join(scene_dir, "camera_02" if setup == "balanced" else "camera_01")
                gt_file = ensure_exists(os.path.join(scene_dir, "disp_00.npy"), "Booster GT")
                noc_file = os.path.join(scene_dir, "mask_00.png")
                for left_file in sorted(Path(left_dir).glob("im*.png")):
                    right_file = os.path.join(right_dir, left_file.name)
                    ensure_exists(right_file, "Booster right image")
                    samples.append(
                        StereoSample(
                            dataset=self.dataset_name,
                            setup=setup,
                            scene=scene,
                            left_file=str(left_file),
                            right_file=right_file,
                            gt_file=gt_file,
                            left_name=left_file.name,
                            noc_file=noc_file if os.path.isfile(noc_file) else None,
                        )
                    )
        return samples

    def load_disp(self, sample):
        return np.load(sample.gt_file).astype(np.float32)

    def load_noc(self, sample, gt):
        if sample.noc_file is None:
            return None
        mask = imageio.imread(sample.noc_file)
        if mask.ndim == 3:
            mask = mask[..., 0]
        return mask == 255

    def apply_dataset_transforms(self, sample, img0, img1, gt, valid, noc):
        ori_w = img0.shape[1]
        if sample.setup == "unbalanced" and self.match_unbalanced_left_to_right and img1.shape[:2] != img0.shape[:2]:
            target_h, target_w = img1.shape[:2]
            img0 = cv2.resize(img0, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt, (target_w, target_h), interpolation=cv2.INTER_LINEAR) * (target_w / float(ori_w))
            valid = cv2.resize(valid.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0
            if noc is not None:
                noc = cv2.resize(noc.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0
        elif img1.shape[:2] != img0.shape[:2]:
            raise ValueError(f"Booster size mismatch: {img0.shape[:2]} vs {img1.shape[:2]}")

        setup_scale = self.balanced_input_scale if sample.setup == "balanced" else self.unbalanced_input_scale
        total_scale = self.scale * setup_scale
        if total_scale != 1.0:
            new_w = max(1, int(round(img0.shape[1] * total_scale)))
            new_h = max(1, int(round(img0.shape[0] * total_scale)))
            img0 = cv2.resize(img0, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            img1 = cv2.resize(img1, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt, (new_w, new_h), interpolation=cv2.INTER_LINEAR) * total_scale
            valid = cv2.resize(valid.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 0
            if noc is not None:
                noc = cv2.resize(noc.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 0
        return img0, img1, gt, valid, noc


class SceneFlowStereoDataset(BaseStereoDataset):
    def __init__(self, root, split="train", pass_name="finalpass", include_driving=True, include_monkaa=True, scale=1.0):
        super().__init__(dataset_name="sceneflow", scale=scale)
        self.root = ensure_exists(root, "SceneFlow root")
        self.split = split.lower()
        self.pass_name = pass_name.lower()
        if self.pass_name not in {"cleanpass", "finalpass"}:
            raise ValueError(f"Unsupported SceneFlow pass_name: {pass_name}")
        self.samples = self._collect_samples(include_driving=include_driving, include_monkaa=include_monkaa)

    def _pass_dir(self, stem):
        return f"{stem}_{self.pass_name}" if not stem.endswith(self.pass_name) else stem

    def _collect_samples(self, include_driving, include_monkaa):
        samples = []
        split_dir = "TRAIN" if self.split == "train" else "TEST"

        things_glob = os.path.join(self.root, f"frames_{self.pass_name}", split_dir, "*", "*", "left", "*.png")
        for left_file in sorted(glob(things_glob)):
            right_file = left_file.replace("/left/", "/right/")
            gt_file = left_file.replace(f"frames_{self.pass_name}", "disparity").replace(".png", ".pfm")
            scene = "/".join(Path(left_file).parts[-4:-1])
            samples.append(
                StereoSample("sceneflow", "default", scene, left_file, right_file, gt_file, os.path.basename(left_file))
            )

        if self.split == "train" and include_driving:
            driving_glob = os.path.join(self.root, f"driving_frames_{self.pass_name}", "*", "*", "*", "left", "*.png")
            for left_file in sorted(glob(driving_glob)):
                right_file = left_file.replace("/left/", "/right/")
                gt_file = left_file.replace(f"frames_{self.pass_name}", "disparity").replace(".png", ".pfm")
                scene = "/".join(Path(left_file).parts[-5:-1])
                samples.append(
                    StereoSample("sceneflow", "default", scene, left_file, right_file, gt_file, os.path.basename(left_file))
                )

        if self.split == "train" and include_monkaa:
            monkaa_glob = os.path.join(self.root, f"monkaa_frames_{self.pass_name}", "*", "left", "*.png")
            for left_file in sorted(glob(monkaa_glob)):
                right_file = left_file.replace("/left/", "/right/")
                gt_file = left_file.replace(f"frames_{self.pass_name}", "disparity").replace(".png", ".pfm")
                scene = "/".join(Path(left_file).parts[-3:-1])
                samples.append(
                    StereoSample("sceneflow", "default", scene, left_file, right_file, gt_file, os.path.basename(left_file))
                )

        return samples

    def load_disp(self, sample):
        disp = read_pfm(sample.gt_file)
        if disp.ndim == 3:
            disp = disp[..., 0]
        return np.ascontiguousarray(disp, dtype=np.float32)


class KITTIStereoDataset(BaseStereoDataset):
    def __init__(self, root, version="2015", split="train", scale=1.0):
        dataset_name = f"kitti{version}"
        super().__init__(dataset_name=dataset_name, scale=scale)
        self.root = ensure_exists(root, f"KITTI {version} root")
        self.version = str(version)
        self.split = split.lower()
        if self.version not in {"2012", "2015"}:
            raise ValueError(f"Unsupported KITTI version: {version}")
        self.samples = self._collect_samples()

    def _collect_samples(self):
        samples = []
        if self.version == "2015":
            split_dir = "training" if self.split == "train" else "testing"
            left_glob = os.path.join(self.root, split_dir, "image_2", "*_10.png")
            for left_file in sorted(glob(left_glob)):
                right_file = left_file.replace("/image_2/", "/image_3/")
                gt_file = left_file.replace("/image_2/", "/disp_occ_0/") if self.split == "train" else None
                noc_file = left_file.replace("/image_2/", "/disp_noc_0/") if self.split == "train" else None
                samples.append(
                    StereoSample(self.dataset_name, "default", split_dir, left_file, right_file, gt_file, os.path.basename(left_file), noc_file)
                )
        else:
            split_dir = "training" if self.split == "train" else "testing"
            left_glob = os.path.join(self.root, split_dir, "colored_0", "*_10.png")
            for left_file in sorted(glob(left_glob)):
                right_file = left_file.replace("/colored_0/", "/colored_1/")
                gt_file = left_file.replace("/colored_0/", "/disp_occ/") if self.split == "train" else None
                noc_file = left_file.replace("/colored_0/", "/disp_noc/") if self.split == "train" else None
                samples.append(
                    StereoSample(self.dataset_name, "default", split_dir, left_file, right_file, gt_file, os.path.basename(left_file), noc_file)
                )
        return samples

    def load_disp(self, sample):
        if sample.gt_file is None:
            return None
        disp = imageio.imread(sample.gt_file).astype(np.float32) / 256.0
        if disp.ndim == 3:
            disp = disp[..., 0]
        return disp

    def load_noc(self, sample, gt):
        if sample.noc_file is None or not os.path.isfile(sample.noc_file):
            return None
        noc = imageio.imread(sample.noc_file)
        if noc.ndim == 3:
            noc = noc[..., 0]
        return noc > 0


class MiddleburyStereoDataset(BaseStereoDataset):
    def __init__(self, root, split="train", resolution="H", scale=1.0):
        super().__init__(dataset_name="middlebury", scale=scale)
        self.root = ensure_exists(root, "Middlebury root")
        self.split = split.lower()
        self.resolution = str(resolution)
        self.samples = self._collect_samples()

    def _collect_samples(self):
        samples = []
        if self.resolution == "2014":
            scenes = sorted((Path(self.root) / "2014").glob("*"))
            for scene_dir in scenes:
                if self.split != "train":
                    continue
                for suffix in ["", "E", "L"]:
                    right_file = scene_dir / f"im1{suffix}.png"
                    if right_file.is_file():
                        samples.append(
                            StereoSample(
                                "middlebury",
                                "default",
                                scene_dir.name,
                                str(scene_dir / "im0.png"),
                                str(right_file),
                                str(scene_dir / "disp0.pfm"),
                                right_file.name.replace("im1", "im0", 1),
                                str(scene_dir / "mask0nocc.png") if (scene_dir / "mask0nocc.png").is_file() else None,
                            )
                        )
            return samples

        if self.resolution not in {"Q", "H", "F"}:
            raise ValueError(f"Unsupported Middlebury resolution: {self.resolution}")
        split_dir = f"training{self.resolution}" if self.split == "train" else f"test{self.resolution}"
        left_glob = os.path.join(self.root, "MiddEval3", split_dir, "*", "im0.png")
        for left_file in sorted(glob(left_glob)):
            scene_dir = os.path.dirname(left_file)
            samples.append(
                StereoSample(
                    "middlebury",
                    "default",
                    os.path.basename(scene_dir),
                    left_file,
                    os.path.join(scene_dir, "im1.png"),
                    os.path.join(scene_dir, "disp0GT.pfm") if self.split == "train" else None,
                    "im0.png",
                    os.path.join(scene_dir, "mask0nocc.png") if self.split == "train" else None,
                )
            )
        return samples

    def load_disp(self, sample):
        if sample.gt_file is None:
            return None
        disp = read_pfm(sample.gt_file)
        if disp.ndim == 3:
            disp = disp[..., 0]
        disp = np.ascontiguousarray(disp, dtype=np.float32)
        disp[~np.isfinite(disp)] = 0.0
        return disp

    def load_noc(self, sample, gt):
        if sample.noc_file is None or not os.path.isfile(sample.noc_file):
            return None
        noc = imageio.imread(sample.noc_file)
        if noc.ndim == 3:
            noc = noc[..., 0]
        return noc == 255


class ETH3DStereoDataset(BaseStereoDataset):
    def __init__(self, root, split="train", scale=1.0):
        super().__init__(dataset_name="eth3d", scale=scale)
        self.root = ensure_exists(root, "ETH3D root")
        self.split = split.lower()
        self.samples = self._collect_samples()

    def _collect_samples(self):
        samples = []
        left_glob = os.path.join(self.root, "*", "im0.png")
        for left_file in sorted(glob(left_glob)):
            scene_dir = os.path.dirname(left_file)
            gt_file = os.path.join(scene_dir, "disp0GT.pfm")
            has_gt = os.path.isfile(gt_file)
            if self.split == "train" and not has_gt:
                continue
            if self.split == "test" and has_gt:
                continue
            samples.append(
                StereoSample(
                    "eth3d",
                    "default",
                    os.path.basename(scene_dir),
                    left_file,
                    os.path.join(scene_dir, "im1.png"),
                    gt_file if has_gt else None,
                    "im0.png",
                    os.path.join(scene_dir, "mask0nocc.png") if has_gt else None,
                )
            )
        return samples

    def load_disp(self, sample):
        if sample.gt_file is None:
            return None
        disp = read_pfm(sample.gt_file)
        if disp.ndim == 3:
            disp = disp[..., 0]
        disp = np.ascontiguousarray(disp, dtype=np.float32)
        disp[~np.isfinite(disp)] = 0.0
        return disp

    def load_noc(self, sample, gt):
        if sample.noc_file is None or not os.path.isfile(sample.noc_file):
            return None
        noc = imageio.imread(sample.noc_file)
        if noc.ndim == 3:
            noc = noc[..., 0]
        return noc == 255


def single_batch_collate(batch):
    if len(batch) != 1:
        raise ValueError("Current unified loader assumes batch_size=1 for variable-size stereo inputs.")
    return batch[0]


def build_dataset(dataset_cfg, subset):
    dataset_name = dataset_cfg["dataset"].lower()

    if dataset_name == "booster":
        return BoosterStereoDataset(
            booster_root=dataset_cfg["booster_root"],
            split_file=dataset_cfg["split_file"],
            subset=subset,
            setups=dataset_cfg["train_setups"] if subset == "train" else dataset_cfg["val_setups"],
            scale=dataset_cfg.get("scale", 1.0),
            balanced_input_scale=dataset_cfg.get("balanced_input_scale", 0.25),
            unbalanced_input_scale=dataset_cfg.get("unbalanced_input_scale", 1.0),
            match_unbalanced_left_to_right=dataset_cfg.get("match_unbalanced_left_to_right", True),
            split_dir_name=dataset_cfg.get("split_dir_name", "train"),
        )

    if dataset_name == "sceneflow":
        return SceneFlowStereoDataset(
            root=dataset_cfg["root"],
            split=dataset_cfg.get("train_split", "train") if subset == "train" else dataset_cfg.get("val_split", "test"),
            pass_name=dataset_cfg.get("pass_name", "finalpass"),
            include_driving=dataset_cfg.get("include_driving", True),
            include_monkaa=dataset_cfg.get("include_monkaa", True),
            scale=dataset_cfg.get("scale", 1.0),
        )

    if dataset_name in {"kitti", "kitti2015", "kitti2012"}:
        version = dataset_cfg.get("version", "2015")
        if dataset_name == "kitti2012":
            version = "2012"
        elif dataset_name == "kitti2015":
            version = "2015"
        return KITTIStereoDataset(
            root=dataset_cfg["root"],
            version=version,
            split=dataset_cfg.get("train_split", "train") if subset == "train" else dataset_cfg.get("val_split", "test"),
            scale=dataset_cfg.get("scale", 1.0),
        )

    if dataset_name == "middlebury":
        return MiddleburyStereoDataset(
            root=dataset_cfg["root"],
            split=dataset_cfg.get("train_split", "train") if subset == "train" else dataset_cfg.get("val_split", "test"),
            resolution=dataset_cfg.get("resolution", "H"),
            scale=dataset_cfg.get("scale", 1.0),
        )

    if dataset_name == "eth3d":
        return ETH3DStereoDataset(
            root=dataset_cfg["root"],
            split=dataset_cfg.get("train_split", "train") if subset == "train" else dataset_cfg.get("val_split", "test"),
            scale=dataset_cfg.get("scale", 1.0),
        )

    raise NotImplementedError(f"Dataset '{dataset_name}' is not implemented in core/stereo_datasets.py")


def build_dataloader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=single_batch_collate,
    )
