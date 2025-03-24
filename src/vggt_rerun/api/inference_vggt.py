from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from jaxtyping import Float32, UInt8
from numpy import ndarray
from serde import from_dict
from simplecv.camera_parameters import (
    Extrinsics,
    Intrinsics,
    PinholeParameters,
)
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
from torch import Tensor
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from vggt_rerun.vggt_loading_utils import (
    VGGTPredictions,
    preprocess_images,
)

np.set_printoptions(suppress=True)

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
device = "cuda" if torch.cuda.is_available() else "cpu"


def create_blueprint(parent_log_path: Path, image_paths: list[Path]) -> rrb.Blueprint:
    view3d = rrb.Spatial3DView(
        origin=f"{parent_log_path}",
        contents=[
            "+ $origin/**",
            # don't include depths in the 3D view, as they can be very noisy
            *[f"- /{parent_log_path}/camera_{i}/pinhole/depth" for i in range(len(image_paths))],
        ],
    )
    view2d = rrb.Vertical(
        contents=[
            rrb.Horizontal(
                contents=[
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/camera_{i}/pinhole/",
                        contents=[
                            "+ $origin/**",
                        ],
                        name="Pinhole Content",
                    ),
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/camera_{i}/pinhole/confidence",
                        contents=[
                            "+ $origin/**",
                        ],
                        name="Confidence Map",
                    ),
                ]
            )
            # show at most 4 cameras
            for i in range(min(4, len(image_paths)))
        ]
    )

    blueprint = rrb.Blueprint(rrb.Horizontal(contents=[view3d, view2d], column_shares=[3, 1]), collapse_panels=True)
    return blueprint


@dataclass
class VGGTInferenceConfig:
    rr_config: RerunTyroConfig
    image_dir: Path
    confidence_threshold: float = 50.0


def run_inference(config: VGGTInferenceConfig) -> None:
    print("Running inference on images in", config.image_dir)

    start: float = timer()
    image_paths = []

    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        image_paths.extend(config.image_dir.glob(f"*{ext}"))
    image_paths: list[Path] = sorted(image_paths)
    assert len(image_paths) > 0, (
        f"No images found in {config.image_dir} in supported formats {SUPPORTED_IMAGE_EXTENSIONS}"
    )

    bgr_list: list[UInt8[ndarray, "H W 3"]] = [cv2.imread(str(image_path)) for image_path in image_paths]
    rgb_list: list[UInt8[ndarray, "H W 3"]] = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in bgr_list]

    # initialize rerun
    parent_log_path = Path("world")
    blueprint = create_blueprint(parent_log_path=parent_log_path, image_paths=image_paths)
    rr.send_blueprint(blueprint=blueprint)
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, static=True)
    # Apply the rotation to the root coordinate system
    rr.log(
        f"{parent_log_path}",
        rr.Transform3D(rotation=rr.RotationAxisAngle(axis=(0, 1, 0), radians=-np.pi / 4)),
        static=True,
    )

    img_tensors: Float32[Tensor, "num_img 3 H W"] = preprocess_images(rgb_list).to(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    print("Model loaded in", timer() - start, "seconds")

    # Run inference
    print("Running inference...")

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        # run model and convert to dataclass for type validaton + easy access
        predictions: dict = model(img_tensors)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], img_tensors.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions:
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].numpy(force=True)

    # Convert from dict to dataclass and performs runtime type validation for easy access
    pred_class: VGGTPredictions = from_dict(VGGTPredictions, predictions)
    pred_class = pred_class.remove_batch_dim_if_one()

    # Generate world points from depth map,this is usually more accurate than the world points from pose encoding
    depth_maps: Float32[ndarray, "num_cams H W 1"] = pred_class.depth
    world_points: Float32[ndarray, "num_cams H W 3"] = unproject_depth_map_to_point_map(
        depth_maps, pred_class.extrinsic, pred_class.intrinsic
    ).astype(np.float32)

    # Get colors from original images and reshape them to match points
    original_images: Float32[ndarray, "num_cams 3 H W"] = img_tensors.numpy(force=True)
    # Rearrange to match point shape expectation
    original_images: Float32[ndarray, "num_cams H W 3"] = rearrange(original_images, "num_cams C H W -> num_cams H W C")
    # Flatten both points and colors
    flattened_points: Float32[ndarray, "num_points 3"] = rearrange(world_points, "num_cams H W C -> (num_cams H W) C")
    flattened_colors: Float32[ndarray, "num_points 3"] = rearrange(
        original_images, "num_cams H W C -> (num_cams H W) C"
    )

    depth_confs: Float32[ndarray, "num_cams H W"] = pred_class.depth_conf
    conf: Float32[ndarray, "num_points"] = depth_confs.reshape(-1)  # noqa UP037

    # Convert percentage threshold to actual confidence value
    conf_threshold = 0.0 if config.confidence_threshold == 0.0 else np.percentile(conf, config.confidence_threshold)
    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    vertices_3d: Float32[ndarray, "num_points 3"] = flattened_points[conf_mask]
    colors_rgb: Float32[ndarray, "num_points 3"] = flattened_colors[conf_mask]

    rr.set_time_sequence("timeline", 0)

    rr.log(
        f"{parent_log_path}/point_cloud",
        rr.Points3D(vertices_3d, colors=colors_rgb),
    )
    for idx, (intri, extri, image, depth_map, depth_conf) in enumerate(
        zip(
            pred_class.intrinsic,
            pred_class.extrinsic,
            original_images,
            depth_maps,
            depth_confs,
            strict=True,
        )
    ):
        cam_name: str = f"camera_{idx}"
        cam_log_path: Path = parent_log_path / cam_name
        intri_param = Intrinsics(
            camera_conventions="RDF",
            fl_x=float(intri[0, 0]),
            fl_y=float(intri[1, 1]),
            cx=float(intri[0, 2]),
            cy=float(intri[1, 2]),
            width=image.shape[1],
            height=image.shape[0],
        )
        extri_param = Extrinsics(
            cam_R_world=extri[:, :3],
            cam_t_world=extri[:, 3],
        )
        pinhole_param = PinholeParameters(name=cam_name, intrinsics=intri_param, extrinsics=extri_param)
        conf_threshold = (
            0.0 if config.confidence_threshold == 0.0 else np.percentile(depth_conf, config.confidence_threshold)
        )
        conf_mask = (depth_conf >= conf_threshold) & (depth_conf > 1e-5)
        # filter depth map based on confidence
        depth_map = depth_map.squeeze()
        depth_map[~conf_mask] = 0.0

        rr.log(f"{cam_log_path}/pinhole/image", rr.Image(image))
        rr.log(f"{cam_log_path}/pinhole/confidence", rr.Image(conf_mask.astype(np.float32)))
        rr.log(f"{cam_log_path}/pinhole/depth", rr.DepthImage(depth_map, draw_order=1))
        log_pinhole(pinhole_param, cam_log_path=cam_log_path, image_plane_distance=0.1)

    # Clean up
    torch.cuda.empty_cache()
    print(f"Inference completed in {timer() - start:.2f} seconds")
