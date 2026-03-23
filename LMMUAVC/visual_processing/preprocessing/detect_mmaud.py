"""
MMAUD 2D Drone Detection Script
使用训练好的 YOLO11s 模型对 `Test_image` 目录中的图像执行无人机检测，
并复现 `visual_processing/yolov9/detect.py` 中的关键帧筛选与结果整理流程。
"""

import argparse
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils import LOGGER

DEFAULT_SOURCE = Path(__file__).resolve().parent / 'Test_image'


def _normalize_path(path_str: Optional[str], default_path: Path) -> Path:
    """将传入路径转换为绝对路径；若为空则返回默认路径。"""
    if not path_str:
        return default_path
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        try:
            path = path.resolve(strict=False)
        except TypeError:
            path = Path(str(path))
    return path


def _names_dict(names) -> Dict[int, str]:
    """确保类别名称以 {id: name} 字典形式返回。"""
    if isinstance(names, dict):
        return names
    if isinstance(names, list):
        return {idx: name for idx, name in enumerate(names)}
    raise TypeError("模型类别名称格式不支持。")


def _find_class_id(names: Dict[int, str], target_class: Optional[str]) -> Optional[int]:
    """根据类别名获取类别 id，找不到则返回 None。"""
    if target_class is None:
        return None
    target_class_lower = target_class.lower()
    for class_id, class_name in names.items():
        if class_name.lower() == target_class_lower:
            return class_id
    LOGGER.warning(f"未在模型标签中找到目标类别: {target_class}")
    return None


def _resolve_annotated_path(save_dir: Path, source_path: Path) -> Path:
    """
    在保存目录中查找与源文件同名（不依赖固定后缀）的标注结果。
    若找不到，则返回默认的同名路径。
    """
    candidate = save_dir / source_path.name
    if candidate.exists():
        return candidate

    matches = sorted(save_dir.glob(f'{source_path.stem}.*'))
    if matches:
        return matches[0]

    return candidate


def _select_keyframe_indices(
    detection_records: List[dict], kf_int: int, num_kf: int
) -> List[int]:
    """按帧号排序并每隔 kf_int 帧选取一个关键帧，限定数量不超过 num_kf（>0 时）。"""
    if not detection_records:
        return []
    if kf_int <= 0:
        LOGGER.warning("kf_int <= 0，关键帧选择将退化为保留全部检测帧。")
        kf_int = 0

    frame_to_indices = defaultdict(list)
    for idx, record in enumerate(detection_records):
        frame_to_indices[record['frame_idx']].append(idx)

    sorted_frames = sorted(frame_to_indices.keys())
    selected_indices: List[int] = []
    last_selected_frame = None

    for frame_idx in sorted_frames:
        # 同一帧可能检出多个目标，保留置信度最高的那个
        indices = frame_to_indices[frame_idx]
        best_idx = max(indices, key=lambda i: detection_records[i]['confidence'])

        if last_selected_frame is None or frame_idx - last_selected_frame >= kf_int:
            selected_indices.append(best_idx)
            last_selected_frame = frame_idx
            if num_kf > 0 and len(selected_indices) >= num_kf:
                break

    return selected_indices


def run_detection(
    weights: str = 'yolo11s.pt',
    source: Optional[str] = None,
    data: Optional[str] = None,
    imgsz: int = 640,
    conf_thres: float = 0.65,
    iou_thres: float = 0.45,
    max_det: int = 1000,
    device: str = '',
    save_txt: bool = False,
    save_conf: bool = False,
    save_crop: bool = False,
    project: str = 'runs/detect',
    name: str = 'mmaud_exp',
    exist_ok: bool = False,
    line_thickness: int = 3,
    hide_labels: bool = False,
    hide_conf: bool = False,
    half: bool = False,
    vid_stride: int = 1,
    num_kf: int = 0,
    kf_int: int = 12,
    target_class: str = 'Drone',
):
    """
    使用 YOLO11 模型执行无人机检测并选择关键帧。
    """
    resolved_source = _normalize_path(source, DEFAULT_SOURCE)
    resolved_project = Path(project).expanduser()
    resolved_weights = Path(weights).expanduser()

    LOGGER.info("============================================================")
    LOGGER.info("MMAUD 2D 无人机检测程序 (YOLO11)")
    LOGGER.info("============================================================")
    LOGGER.info(f"模型权重: {resolved_weights}")
    LOGGER.info(f"输入源: {resolved_source}")
    LOGGER.info(f"目标类别: {target_class}")
    LOGGER.info(f"置信度阈值: {conf_thres}")
    LOGGER.info(f"最大检测数: {max_det}")
    LOGGER.info(f"关键帧数量: {num_kf}")
    LOGGER.info(f"关键帧间隔: {kf_int}")
    LOGGER.info("============================================================")

    if device == '':
        device = '0' if torch.cuda.is_available() else 'cpu'

    LOGGER.info(f"使用设备: {device}")
    model = YOLO(str(resolved_weights))
    model_names = _names_dict(model.names)
    target_class_id = _find_class_id(model_names, target_class)

    predict_kwargs = dict(
        source=str(resolved_source),
        imgsz=imgsz,
        conf=conf_thres,
        iou=iou_thres,
        max_det=max_det,
        device=device,
        save=True,
        save_txt=save_txt,
        save_conf=save_conf,
        save_crop=save_crop,
        project=str(resolved_project),
        name=name,
        exist_ok=exist_ok,
        line_width=line_thickness,
        show_labels=not hide_labels,
        show_conf=not hide_conf,
        half=half,
        vid_stride=vid_stride,
    )

    if data:
        predict_kwargs['data'] = data
    if target_class_id is not None:
        predict_kwargs['classes'] = [target_class_id]

    results = list(model.predict(**predict_kwargs))

    if not results:
        LOGGER.warning("未获得任何检测结果。")
        return []

    detection_records: List[dict] = []
    for frame_idx, result in enumerate(results):
        boxes = getattr(result, 'boxes', None)
        if boxes is None or len(boxes) == 0:
            continue

        save_dir = Path(result.save_dir)
        source_path = Path(result.path)
        names_map = _names_dict(getattr(result, 'names', model_names))

        for box in boxes:
            cls_tensor = getattr(box, 'cls', None)
            conf_tensor = getattr(box, 'conf', None)
            bbox_xyxy = None
            if hasattr(box, 'xyxy'):
                xyxy_tensor = box.xyxy
                if hasattr(xyxy_tensor, 'cpu'):
                    xyxy_tensor = xyxy_tensor.cpu()
                bbox_values = xyxy_tensor.tolist()
                if isinstance(bbox_values, list):
                    if bbox_values and isinstance(bbox_values[0], list):
                        bbox_values = bbox_values[0]
                    if len(bbox_values) == 4:
                        bbox_xyxy = tuple(float(v) for v in bbox_values)

            if cls_tensor is None or conf_tensor is None:
                continue

            if hasattr(cls_tensor, 'item'):
                class_id = int(cls_tensor.item())
            else:
                class_id = int(cls_tensor)

            if target_class_id is not None and class_id != target_class_id:
                continue

            if hasattr(conf_tensor, 'item'):
                confidence = float(conf_tensor.item())
            else:
                confidence = float(conf_tensor)

            class_name = names_map.get(class_id, str(class_id))
            annotated_path = _resolve_annotated_path(save_dir, source_path)
            crop_path = save_dir / 'crops' / class_name / f'{source_path.stem}.jpg'

            detection_records.append(
                {
                    'confidence': confidence,
                    'frame_idx': frame_idx,
                    'source_path': source_path,
                    'save_dir': save_dir,
                    'annotated_path': annotated_path,
                    'crop_path': crop_path,
                    'class_name': class_name,
                    'bbox_xyxy': bbox_xyxy,
                }
            )

    LOGGER.info(f"检测到 {len(detection_records)} 个目标实例。")

    if detection_records:
        selected_indices = _select_keyframe_indices(detection_records, kf_int=kf_int, num_kf=num_kf)
        if selected_indices:
            base_save_dir = detection_records[selected_indices[0]]['save_dir']
            selected_dir = base_save_dir / 'selected'
            selected_dir_crop = selected_dir / 'crop'
            selected_dir.mkdir(parents=True, exist_ok=True)
            if save_crop:
                selected_dir_crop.mkdir(parents=True, exist_ok=True)

            LOGGER.info(f"选择了 {len(selected_indices)} 个关键帧。")
            for idx in selected_indices:
                record = detection_records[idx]
                annotated_path = record['annotated_path']
                if annotated_path.exists():
                    shutil.copy(annotated_path, selected_dir / annotated_path.name)
                    LOGGER.info(f"关键帧复制完成: {annotated_path.name}")
                else:
                    LOGGER.warning(f"未找到标注图像: {annotated_path}")

                if save_crop:
                    crop_output_path = selected_dir_crop / annotated_path.name
                    crop_copied = False
                    bbox_xyxy = record.get('bbox_xyxy')
                    source_path = record['source_path']
                    image_candidate = source_path if source_path.exists() else record['annotated_path']
                    if (
                        bbox_xyxy is not None
                        and image_candidate.exists()
                    ):
                        try:
                            with Image.open(image_candidate) as img:
                                width, height = img.size
                                x1, y1, x2, y2 = bbox_xyxy
                                x1 = max(0, min(width, int(round(x1))))
                                y1 = max(0, min(height, int(round(y1))))
                                x2 = max(0, min(width, int(round(x2))))
                                y2 = max(0, min(height, int(round(y2))))
                                if x2 > x1 and y2 > y1:
                                    img.crop((x1, y1, x2, y2)).save(crop_output_path)
                                    crop_copied = True
                                else:
                                    LOGGER.warning(
                                        f"裁剪坐标异常，跳过: {image_candidate.name} ({bbox_xyxy})"
                                    )
                        except Exception as exc:
                            LOGGER.warning(f"裁剪失败: {image_candidate} -> {exc}")
                    if not crop_copied:
                        crop_path = record['crop_path']
                        if crop_path.exists():
                            shutil.copy(crop_path, crop_output_path)
                            crop_copied = True
                        else:
                            LOGGER.warning(f"未找到裁剪图像: {crop_path}")

            LOGGER.info(f"关键帧及裁剪已保存至: {selected_dir}")
        else:
            LOGGER.info("未能满足关键帧间隔约束，未选择任何关键帧。")
    else:
        LOGGER.info("无检测结果或未启用关键帧选择。")

    LOGGER.info("检测流程结束。")
    return results


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='MMAUD 2D Drone Detection with YOLO11')

    parser.add_argument('--weights', type=str, default='yolo11s.pt', help='模型权重路径')
    parser.add_argument('--source', type=str, default=str(DEFAULT_SOURCE), help='输入图像/视频路径')
    parser.add_argument('--data', type=str, default='MMAUD_2D/data.yaml', help='数据集配置文件路径')
    parser.add_argument('--imgsz', '--img-size', type=int, default=640, help='推理图像尺寸')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IOU阈值')
    parser.add_argument('--max-det', type=int, default=1000, help='每张图像的最大检测数量')
    parser.add_argument('--device', default='', help='CUDA设备, 如 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--save-txt', action='store_true', help='保存结果到*.txt')
    parser.add_argument('--save-conf', action='store_true', help='在标签中保存置信度')
    parser.add_argument('--save-crop', action='store_true', help='保存裁剪的预测框')
    parser.add_argument('--project', default='runs/detect', help='保存结果的项目路径')
    parser.add_argument('--name', default='mmaud_exp', help='保存结果的名称')
    parser.add_argument('--exist-ok', action='store_true', help='允许使用已存在的项目/名称')
    parser.add_argument('--line-thickness', type=int, default=3, help='边界框线条粗细')
    parser.add_argument('--hide-labels', action='store_true', help='隐藏标签')
    parser.add_argument('--hide-conf', action='store_true', help='隐藏置信度')
    parser.add_argument('--half', action='store_true', help='使用FP16半精度推理')
    parser.add_argument('--vid-stride', type=int, default=1, help='视频帧率步长')
    parser.add_argument('--num-kf', type=int, default=5, help='选择的关键帧数量')
    parser.add_argument('--kf-int', type=int, default=12, help='关键帧之间的最小间隔（单位: 帧）')
    parser.add_argument('--target-class', type=str, default='Drone', help='筛选的目标类别名称')

    return parser.parse_args()


def main():
    """命令行入口。"""
    args = parse_args()
    run_detection(
        weights=args.weights,
        source=args.source,
        data=args.data,
        imgsz=args.imgsz,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_det=args.max_det,
        device=args.device,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        line_thickness=args.line_thickness,
        hide_labels=args.hide_labels,
        hide_conf=args.hide_conf,
        half=args.half,
        vid_stride=args.vid_stride,
        num_kf=args.num_kf,
        kf_int=args.kf_int,
        target_class=args.target_class,
    )


if __name__ == '__main__':
    main()
