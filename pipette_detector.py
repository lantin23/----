"""
检测模式：
1. circle:基于霍夫圆变换检测移液枪头圆形尾部
2. contour:基于轮廓检测移液枪头细长枪身
整齐度判断策略：
1. 枪头数量
2. 枪头圆心到最近中心槽的距离不超过容差
（3.）所有枪头方向一致
"""

import cv2
import numpy as np
from statistics import median
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Point = Tuple[float, float]

@dataclass
class DetectionResult:
    """
    检测结果数据类
    """
    center: List[Point] = field(default_factory=list)
    angles: List[float] = field(default_factory=list)
    radii: List[float] = field(default_factory=list)
    is_neat: bool = False
    reason: List[str] = field(default_factory=list)
    rows: List[List[Point]] = field(default_factory=list)
    cols: List[List[Point]] = field(default_factory=list)
    #每个点到最近槽线的偏差（单位：像素）
    slot_deviation: List[float] = field(default_factory=list)
    #散落枪头检测结果
    fallen_tips:List[Tuple[float, float, float, float]] = field(default_factory=list) # (cx, cy, w, h)

    @property
    def reason_text(self) -> str:
        if self.is_neat:
            return "整齐"
        return ";".join(self.reason) if self.reason else "未知原因"

class PipetteDetector:
    """移液枪头整齐度检测器"""

    def __init__(self):
        #参数
        self._min_tips: int = 4
        self._slot_aligment_tol: float = 12.0 #槽中心线对齐容差
        self._outlier_ratio_thresh: float = 0.15 #异常枪头对比阈值
        self._min_tips_per_slot: int = 2 #有效槽至少两个枪头
        self._check_uniform_spacing: bool = False
        self._space_ratio: float = 0.18
        self._min_space: float = 15.0

        #槽方向 "auto" | "horizontal" | "vertical"
        self._slot_direction: str = "auto"

        #检测模式
        self._detection_mode: str = "circle"

        #轮廓检测参数
        self._min_area: float = 80.0
        self._max_area: float = 5000.0
        self._angle_thresh: float = 6.0
        self._gaussian_kernel: int = 7
        self._canny_thresh1: int = 30
        self._min_ciecularity: float = 0.15
        self._min_rectangularity: float = 0.3

        #霍夫圆检测参数
        self._hough_dp: float = 1.0
        self._hough_min_dist: float = 25.0
        self._hough_param1: float = 80.0
        self._hough_param2: float = 20.0
        self._hough_min_radius: int = 10
        self._hough_max_radius: int = 40
        self._circle_merge_ratio: float = 0.5
        self._circle_min_radius: float = 8.0 
        self._circle_max_radius: float = 45.0
        self._intensity_diff_thresh: float = 15.0  #外圈-内圈最小灰度差

        #聚类容差
        self._cluster_tol: float = 20.0

        #散落枪头检测参数（细长）
        self._fallen_tip_min_aspect: float = 1.6       #细长比例阈值
        self._fallen_tip_max_aspect: float = 4.5       #最大细长比（过滤槽）
        self._fallen_tip_min_area: float = 300.0       #最小面积
        self._fallen_tip_max_area: float = 2000.0      #最大面积
        self._fallen_tip_min_fill: float = 0.2         #最小填充率
        self._fallen_tip_min_std: float = 25.0         #最小亮度标准
        self._fallen_tip_max_dist_from_array: float = 250.0 #最大距离(机架过滤)
        self._fallen_tip_max_count: int = 0            #最大散落枪头数量(不容忍)
        #完整枪头
        self._full_tip_min_area: float = 5000.0        #轮廓面积阈值
        self._full_tip_target_area: float = 9000.0     #完整枪头面积
        self._full_tip_max_dim: float = 260.0          #最大维度上线（排除槽、隔板）
        self._full_tip_min_dim: float = 160.0          #最小维度下线
        self._full_tip_min_aspect: float = 3.0         #完整枪头最小细长比
        self._full_tip_max_aspect: float = 5.0         #完整枪头最大细长比
        #斜挎槽上枪头改用霍夫线检测两条平行边
        self._diag_tip_min_len: float = 100.0              #线段最小长度
        self._diag_tip_min_angle: float = 25.0         #偏离槽方向的最小夹角
        self._diag_tip_max_angle: float = 75.0         #偏离槽方向的最大夹角
        self._diag_tip_cluster_dist: float = 50.0      #同一枪头两条边的中心类聚距离

        #检测结果
        self._result = DetectionResult()

    #参数getter/setter
    @property
    def detection_mode(self) -> str:
        return self._detection_mode

    @detection_mode.setter
    def detection_mode(self, value: str):
        if value not in("circle", "contour"):
            raise ValueError("检测模式必须是'circle' 或 'contour'")
        self._detection_mode = value

    @property
    def slot_direction(self) -> str:
        return self._slot_dirction

    @slot_direction.setter
    def slot_direction(self, value: str):
        if value not in("auto", "horizontal", "vertical"):
            raise ValueError("槽方向必须是'auto' | 'horizontal' | 'vertical'")
        self._slot_direction = value

    @property
    def min_area(self) -> float:
        return self._min_area

    @min_area.setter
    def min_area(self, value:float):
        if value <= 0:
            raise ValueError(f"最小面积必须大于0")
        if value >= self._max_area:
            raise ValueError(f"最小面积({value})不能大于等于最大面积({self._max_area})")
        self._min_area = value

    @property
    def max_area(self) -> float:
        return self._max_area

    @max_area.setter
    def max_area(self, value: float):
        if value <= 0:
            raise ValueError("最大面积必须大于0")
        if value <= self._min_area:
            raise ValueError(f"最大面积({value})不能小于等于最小面积({self._min_area})")
        self._max_area = value

    @property
    def angle_threshold(self) -> float:
        return self._angle_thresh

    @angle_threshold.setter
    def angle_threshold(self, value: float):
        if value <= 0:
            raise ValueError("角度阈值必须大于0")
        self._angle_thresh = value

    @property
    def space_ratio(self) -> float:
        return self._space_ratio

    @space_ratio.setter
    def space_ratio(self, value: float):
        if not (0 < value < 1):
            raise ValueError("间距偏差比例必须在(0, 1)范围内")
        self._space_ratio = value

    @property
    def min_tips(self) -> int:
        return self._min_tips

    @min_tips.setter
    def min_tips(self, value: int):
        if value <= 0:
            raise ValueError("最少枪头数量必须大于0")
        self._min_tips = value

    @property
    def slot_alignment_tolerance(self) -> float:
        return self._slot_alignment_tol

    @slot_alignment_tolerance.setter
    def slot_alignment_tolerance(self, value: float):
        if value <= 0:
            raise ValueError("槽对齐容差必须大于 0")
        self._slot_alignment_tol = value

    @property
    def outlier_ratio_threshold(self) -> float:
        return self._outlier_ratio_thresh

    @outlier_ratio_threshold.setter
    def outlier_ratio_threshold(self, value: float):
        if not (0 < value <= 1):
            raise ValueError("异常比例阈值必须在 (0, 1] 范围内")
        self._outlier_ratio_thresh = value

    @property
    def min_tips_per_slot(self) -> int:
        return self._min_tips_per_slot

    @min_tips_per_slot.setter
    def min_tips_per_slot(self, value: int):
        if value < 1:
            raise ValueError("每槽最少枪头数必须大于等于 1")
        self._min_tips_per_slot = value

    @property
    def check_uniform_spacing(self) -> bool:
        return self._check_uniform_spacing

    @check_uniform_spacing.setter
    def check_uniform_spacing(self, value: bool):
        self._check_uniform_spacing = bool(value)

    #圆形检测参数

    @property
    def hough_min_dist(self) -> float:
        return self._hough_min_dist

    @hough_min_dist.setter
    def hough_min_dist(self, value: float):
        if value <= 0:
            raise ValueError("圆心最小距离必须大于0")
        self._hough_min_dist = value

    @property
    def hough_param1(self) -> float:
        return self._hough_param1

    @hough_param1.setter
    def hough_param1(self, value: float):
        if value <= 0:
            raise ValueError("param1 必须大于0")

    @property
    def hough_param2(self) -> float:
        return self._hough_param2

    @hough_param2.setter
    def hough_param2(self, value: float):
        if value <= 0:
            raise ValueError("param2 必须大于0")
        self._hough_param2 = value

    @property
    def hough_min_radius(self) -> int:
        return self._hough_min_radius

    @hough_min_radius.setter
    def hough_min_radius(self, value: int):
        if value <= 0:
            raise ValueError("最小半径必须大于0")
        if value >= self._hough_max_radius:
            raise ValueError(f"最小半径({value})不能大于等于最大半径({self._hough_max_radius})")
        self._hough_min_radius = value

    @property 
    def hough_max_radius(self) -> int:
        return self._hough_max_radius

    @hough_max_radius.setter
    def hough_max_radius(self, value: int):
        if value <= 0:
            raise ValueError("最大半径必须大于 0")
        if value <= self._hough_min_radius:
            raise ValueError(f"最大半径({value})不能小于等于最小半径({self._hough_min_radius})")
        self._hough_max_radius = value 

    @property
    def circle_merge_ratio(self) -> float:
        return self._circle_merge_ratio

    @circle_merge_ratio.setter
    def circle_merge_ratio(self, value: float):
        if not (0 < value < 1):
            raise ValueError("合并比例必须在(0, 1]范围内")
        self._circle_merge_ratio = value

    @property
    def intensity_diff_threshold(self) -> float:
        return self._intensity_diff_thresh

    @intensity_diff_threshold.setter
    def intensity_diff_threshold(self, value: float):
        if value < 0:
            raise ValueError("强度差阈值必须大于等于 0")
        self._intensity_diff_thresh = value

    #散落枪头检测参数

    @property
    def fallen_tip_min_aspect(self) -> float:
        return self._fallen_tip_min_aspect

    @fallen_tip_min_aspect.setter
    def fallen_tip_min_aspect(self, value: float):
        if value < 1.0:
            raise ValueError("细长比例阈值必须 >= 1.0")
        self._fallen_tip_min_aspect = value 

    @property 
    def fallen_tip_min_area(self) -> float:
        return self._fallen_tip_min_area

    @fallen_tip_min_area.setter
    def fallen_tip_min_area(self, value: float):
        if value <= 0:
            raise ValueError("最小面积必须大于 0")
        if value >= self._fallen_tip_max_area:
            raise ValueError(f"最小面积({value})不能大于等于最大面积({self._fallen_tip_max_area})")
        self._fallen_tip_min_area = value 

    @property
    def fallen_tip_max_area(self) -> float:
        return self._fallen_tip_max_area

    @fallen_tip_max_area.setter
    def fallen_tip_max_area(self, value: float):
        if value <= 0:
            raise ValueError("最大面积必须大于 0")
        if value <= self._fallen_tip_min__area:
            raise ValueError(f"最大面积({value})不能小于等于最小面积({self._fallen_tip_min_area})")
        self._fallen_tip_max_area = value

    @property
    def fallen_tip_max_area(self) -> float:
        return self._fallen_tip_max_area

    @fallen_tip_max_area.setter
    def fallen_tip_max_count(self, value: int):
        if value < 0:
            raise ValueError("散落枪头容忍数必须 >= 0")
        self._fallen_tip_max_count = value

    @property
    def fallen_tip_min_fill(self) -> float:
        return self._fallen_tip_min_fill

    @fallen_tip_min_fill.setter
    def fallen_tip_min_fill(self, value: float):
        if not (0 < value < 1):
            raise ValueError("最小填充率必须在 (0, 1) 范围内")
        self._fallen_tip_min_fill = value

    @property
    def fallen_tip_min_std(self) -> float:
        return self._fallen_tip_min_std

    @fallen_tip_min_std.setter
    def fallen_tip_min_std(self, value: float):
        if value < 0:
            raise ValueError("最小亮度标准差必须 >= 0")
        self._fallen_tip_min_std = value

    #结果获取

    @property
    def result(self) -> DetectionResult:
        return self._result

    def is_neat(self) -> bool:
        return self._result.is_neat

    def get_reasons(self) -> List[str]:
        return self._result.reasons

    def get_centers(self) -> List[Point]:
        return self._result.centers

    def get_angles(self) -> List[float]:
        return self._result.angles

    def get_radii(self) -> List[float]:
        return self._result.radii

    #核心检测流程

    def _clear_data(self):
        self._result = DetectionResult()

    def detect(self, frame: np.ndarray) -> bool:
        """主检测函数，返回是否整齐"""
        self._clear_data()

        if frame is None or frame.size == 0:
            self._result.reasons.append("输入图像为空")
            return False

        try:
            if self._detection_mode == "circle":
                self._detect_circles(frame)
                # 散落枪头检测：基于轮廓找非圆形的细长形状
                self._detect_fallen_tips(frame)
            else:
                self._detect_contours(frame)
        except Exception as e:
            print(f"[检测错误] {e}")
            self._result.reasons.append(f"检测过程出错: {e}")
            return False

        self._result.is_neat = self._check_neatness()
        return self._result.is_neat

    #圆形检测模式

    def _detect_circles(self, frame: np.ndarray):
        """基于霍夫圆变换检测圆形枪头尾部"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=self._hough_dp,
            minDist=self._hough_min_dist,
            param1=self._hough_param1,
            param2=self._hough_param2,
            minRadius=self._hough_min_radius,
            maxRadius=self._hough_max_radius,
        )

        if circles is None or circles.size == 0:
            return

        raw = circles[0]

        # 半径预过滤
        filtered = [
            (float(c[0]), float(c[1]), float(c[2]))
            for c in raw
            if self._circle_min_radius <= c[2] <= self._circle_max_radius
        ]

        # 合并同心圆/重叠圆
        merged = self._merge_overlapping_circles(filtered)

        # 强度验证：枪头尾部应有中心暗孔 + 边缘亮环
        verified = [
            c for c in merged
            if self._verify_circle_intensity(gray, c)
        ]

        for cx, cy, r in verified:
            self._result.centers.append((cx, cy))
            self._result.radii.append(r)
            self._result.angles.append(0.0)

    def _merge_overlapping_circles(
        self, circles: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        """合并同心圆/重叠圆，一个簇保留一个最大外圆"""
        if not circles:
            return []
 
        sorted_idx = sorted(range(len(circles)), key=lambda i: circles[i][2], reverse=True)
        merged: List[Tuple[float, float, float]] = []
        used = set()

        for i in sorted_idx:
            if i in used:
                continue
            cx, cy, r = circles[i]
            cluster = [(cx, cy, r)]
            used.add(i)

            for j in sorted_idx:
                if j in used:
                    continue
                cx2, cy2, r2 = circles[j]
                dist = np.hypot(cx - cx2, cy - cy2)
                threshold = self._circle_merge_ratio * max(r, r2)
                if dist < threshold or dist < 12.0:
                    cluster.append((cx2, cy2, r2))
                    used.add(j)

            avg_x = float(np.mean([c[0] for c in cluster]))
            avg_y = float(np.mean([c[1] for c in cluster]))
            max_r = float(max([c[2] for c in cluster]))
            merged.append((avg_x, avg_y, max_r))

        return merged

    def _verify_circle_intensity(
        self, gray: np.ndarray, circle: Tuple[float, float, float]
    ) -> bool:
        """
        强度验证：真实枪头尾部截面为"中心暗孔 + 边缘亮环"。
        计算圆环外圈与内圈的平均灰度差，剔除背景/枪身等误检。
        """
        if self._intensity_diff_thresh <= 0:
            return True

        cx, cy, r = circle
        h, w = gray.shape

        # 生成圆环掩码
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)

        inner_mask = (dist >= max(r * 0.15, 2)) & (dist <= r * 0.55)
        outer_mask = (dist >= r * 0.75) & (dist <= r * 1.15)

        inner_pixels = gray[inner_mask]
        outer_pixels = gray[outer_mask]

        if len(inner_pixels) < 5 or len(outer_pixels) < 5:
            return True  # 像素不足，跳过验证

        inner_mean = float(np.mean(inner_pixels))
        outer_mean = float(np.mean(outer_pixels))

        # 外圈应明显亮于内圈（枪头尾部的塑料环 vs 中空孔）
        return (outer_mean - inner_mean) >= self._intensity_diff_thresh

    def _detect_fallen_tips(self, frame: np.ndarray):
        """
        散落枪头检测：在圆形检测之后，扫描整个图像寻找"非圆形"的枪头形状。
        当枪头从槽中掉落或横放时，从上方俯视会呈现为细长形状而非圆形。
        这些散落的枪头是"不整齐"的判断依据。

        检测策略:
        1. 用 Canny 边缘检测找出所有边缘轮廓
        2. 用 OTSU 阈值找出明亮区域
        3. 对两类轮廓都做形状分析：要求纵横比 > 阈值
        4. 灰度/纹理过滤：排除纯黑槽缝、均匀亮色块（颜色无关，不依赖槽/机架颜色）
        5. 槽方向过滤：排除沿槽方向的细长形状（多为槽边缘）
        6. 排除与已检测圆重叠的轮廓
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 边缘检测 (透明枪头的边缘在Canny上可见)
        edges = cv2.Canny(blur, 30, 90)

        # 形态学膨胀让细长特征更连贯
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        edges_dilated = cv2.dilate(edges, kernel_line, iterations=1)

        # OTSU 阈值找明亮区域
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        contours_e, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_b, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours = list(contours_e) + list(contours_b)

        # 已检测的圆形（用于去重）
        detected_circles = [
            (center[0], center[1], r)
            for center, r in zip(self._result.centers, self._result.radii)
        ]

        # 自动判断槽方向，过滤沿槽方向的细长形状
        slot_direction = self._determine_slot_direction() if detected_circles else "horizontal"

        fallen: List[Tuple[float, float, float, float]] = []
        seen: List[Tuple[float, float]] = []
        full_tip_candidates: List[Tuple[float, float, float, float, float, float, float]] = []
        self._last_full_tip_contours: List[np.ndarray] = []

        # 机架边距（机架外边框通常在图像最边缘，不依赖具体颜色）
        border_margin = max(80, int(min(h, w) * 0.06))

        for contour in all_contours:
            area = cv2.contourArea(contour)
            if area < self._fallen_tip_min_area or area > self._fallen_tip_max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            if bw <= 0 or bh <= 0:
                continue

            # 1. 位置过滤：排除图像最边缘（机架边框）
            if x < border_margin or y < border_margin:
                continue
            if x + bw > w - border_margin or y + bh > h - border_margin:
                continue

            # 2. 细长比过滤
            aspect = max(bw, bh) / min(bw, bh)
            if aspect < self._fallen_tip_min_aspect:
                continue
            if aspect > self._fallen_tip_max_aspect:
                continue

            # 3. 填充率过滤：真实物体的轮廓应该填满其边界框
            #    细长边缘线条的填充率很低（仅边缘），会被过滤
            bbox_area = bw * bh
            fill_ratio = area / bbox_area
            if fill_ratio < self._fallen_tip_min_fill:
                continue

            # 4. 槽方向过滤：极细长且沿槽方向的形状是槽边缘而非散落枪头
            #    只过滤极细长的情况（aspect > 4），而非任意沿槽方向
            #    这样水平散落的锥形枪头（aspect 2-3）不会被误判
            if slot_direction == "horizontal" and bw > bh * 4.0 and aspect > 5.0:
                continue
            if slot_direction == "vertical" and bh > bw * 4.0 and aspect > 5.0:
                continue

            cx = x + bw / 2
            cy = y + bh / 2

            # 5. 距圆阵列距离过滤（颜色无关）：
            #    散落/缺失的枪头必然位于枪头阵列内部或紧邻处；
            #    离所有已检圆心都很远的小块状候选，多为机架边框/背景纹理。
            if detected_circles:
                near_dist = min(
                    np.hypot(cx - ccx, cy - ccy) for ccx, ccy, _ in detected_circles
                )
                if (
                    near_dist > self._fallen_tip_max_dist_from_array
                    and max(bw, bh) < 70
                ):
                    continue

            # 6. 位置/方向过滤：真实散落枪头应在"槽中"（行间或列间），而不在行/列中心
            #    如果候选位置与已检测的圆心行/列重合，多半是槽边缘而非散落枪头
            if detected_circles:
                if slot_direction == "vertical":
                    # 竖槽：检查轮廓的 X 是否在某个圆心列上
                    for ccx, ccy, cr in detected_circles:
                        if abs(cx - ccx) < 30:  # X 重合
                            overlap_row = True
                            break
                    else:
                        overlap_row = False
                else:
                    # 横槽：检查轮廓的 Y 是否在某个圆心行上
                    for ccx, ccy, cr in detected_circles:
                        if abs(cy - ccy) < 30:  # Y 重合
                            overlap_row = True
                            break
                    else:
                        overlap_row = False
                # 如果与至少一个圆心行/列重合，且形状细长且沿槽方向 → 视为槽边缘
                # 但大块候选（轮廓面积 >= _full_tip_min_area）更可能是完整枪头，
                # 不应用此过滤，避免把横躺/竖躺在槽中的完整枪头误判为槽边缘。
                if overlap_row and area < self._full_tip_min_area:
                    if slot_direction == "vertical" and bh > bw * 1.3:
                        continue
                    if slot_direction == "horizontal" and bw > bh * 1.3:
                        continue

            # 6. 灰度/纹理过滤（颜色无关）：
            #    真实散落枪头是有明暗变化的塑料物体（锥形、开口、阴影），
            #    而槽缝/机架边缘是均匀的色块，靠灰度的均值与方差即可区分。
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            interior_pixels = frame[mask == 255]
            if len(interior_pixels) < 10:
                continue
            interior_gray = cv2.cvtColor(
                interior_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY
            ).flatten()
            mean_brightness = float(np.mean(interior_gray))
            std_brightness = float(np.std(interior_gray))

            # 纯黑区域 (槽缝) - 排除：平均亮度极低且变化小（均匀黑）
            if mean_brightness < 30 and std_brightness < 20:
                continue

            # 纯白/亮区域 (机架白条/亮色塑料条) - 排除
            if mean_brightness > 220:
                continue

            # 低变异区域 (机架塑料/纯色块) - 用 std 过滤
            # 完整透明枪头内部灰度变化较小，对其放宽阈值
            min_std_for_this = (
                12.0 if area >= self._full_tip_min_area else self._fallen_tip_min_std
            )
            if std_brightness < min_std_for_this:
                continue

            # 计算内部饱和度（仅用于下方网格过滤的"暗色候选"保留判断，
            
            interior_hsv = cv2.cvtColor(
                interior_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV
            ).reshape(-1, 3)
            mean_s = float(np.mean(interior_hsv[:, 1]))

            # 7. 垂直槽场景的网格规律性过滤：
            #    在竖槽图像中，整齐的列里每个圆心/相邻中点都是规则的。
            #    候选若正好落在这些规则网格位置（圆心或正常间距中点），
            #    多半是槽间阴影而非散落枪头。
            #    例外：颜色明显偏暗但有一定饱和度（塑料材质有色）的候选，
            #    更可能是缺失枪头/散落枪头，予以保留（用于捕捉竖槽缺失场景）。
            
            if detected_circles and slot_direction == "vertical":
                on_grid = self._is_on_regular_vertical_grid(cx, cy, detected_circles)
                is_dark_saturated = mean_brightness < 65 and mean_s > 45
                if on_grid and not is_dark_saturated:
                    continue

            # 8. 与已记录散落枪头去重
            dup = False
            for sx, sy in seen:
                if abs(cx - sx) < 40 and abs(cy - sy) < 40:
                    dup = True
                    break
            if dup:
                continue

            seen.append((cx, cy))
            fallen.append((cx, cy, bw, bh))

            # 记录可能是"完整枪头"的候选（可见完整长锥形枪头）
            if (
                area >= self._full_tip_min_area
                and self._full_tip_min_dim <= max(bw, bh) <= self._full_tip_max_dim
                and self._full_tip_min_aspect <= aspect <= self._full_tip_max_aspect
            ):
                full_tip_candidates.append((area, cx, cy, bw, bh, aspect, max(bw, bh)))
                self._last_full_tip_contours.append(contour)

        self._last_full_tip_candidates = full_tip_candidates

        # 后处理：识别"带圆头凸缘的真实枪头" vs "矩形空隙"
        self._last_fallen_all = list(fallen)
        # 后处理：识别"带圆头凸缘的真实枪头" vs "矩形空隙"
        # 真实散落枪头在槽缝投下的阴影带圆形凸缘（枪头顶部接口），与空槽位的矩形不同。
        # - 真实枪头：solidity < 0.92（凸包不紧密）+ 端部宽度比 > 1.15（圆头凸出主体）
        # - 矩形空隙：solidity ≥ 0.92 且端部宽度比 ≈ 1.0
        real_tip_candidates = []
        for cand, cnt in zip(full_tip_candidates, self._last_full_tip_contours):
            area, cx, cy, bw, bh, aspect, _ = cand
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1)

            # 端部宽度比：长轴 10 段分段
            x0, y0, _, _ = cv2.boundingRect(cnt)
            mask_local = np.zeros((bh + 4, bw + 4), dtype=np.uint8)
            cnt_shifted = cnt - np.array([x0 - 2, y0 - 2])
            cv2.drawContours(mask_local, [cnt_shifted], -1, 255, -1)
            if bw >= bh:
                seg_w = max(1, bw // 10)
                widths = []
                for i in range(10):
                    lo, hi = i * seg_w, (i + 1) * seg_w if i < 9 else bw
                    col = mask_local[:, lo:hi]
                    rows = np.where(col.any(axis=1))[0]
                    widths.append(rows.max() - rows.min() if len(rows) > 1 else 0)
            else:
                mask_local = cv2.rotate(mask_local, cv2.ROTATE_90_CLOCKWISE)
                h2, w2 = mask_local.shape
                seg_w = max(1, w2 // 10)
                widths = []
                for i in range(10):
                    lo, hi = i * seg_w, (i + 1) * seg_w if i < 9 else w2
                    col = mask_local[:, lo:hi]
                    rows = np.where(col.any(axis=1))[0]
                    widths.append(rows.max() - rows.min() if len(rows) > 1 else 0)

            max_w = max(widths) if widths else 0
            mid_w = sum(widths[3:7]) / 4 if len(widths) >= 7 else max_w
            end_max_ratio = max_w / max(mid_w, 1)

            # 真实枪头判别：solidity 显著低（圆头凸出导致凸包不紧密）
            # 且端部宽度比 > 1.15（圆头宽度 > 中段主体宽度）
            if solidity < 0.92 and end_max_ratio > 1.15:
                real_tip_candidates.append(cand)

            # ==================== 汇总散落枪头 ====================
        # 1) 完整枪头（带圆头凸缘，通过 solidity + 端宽比验证）—— 全部标记，不再只取 1 个
        real_tips: List[Tuple[float, float, float, float]] = [
            (cand[1], cand[2], cand[3], cand[4]) for cand in real_tip_candidates
        ]

        # 2) 对角线散落枪头（斜跨在槽上，霍夫线检测两条平行长边）
        diagonal_tips = self._detect_diagonal_tips(frame, slot_direction)

        # 3) 合并去重：完整枪头 + 对角线枪头
        merged_tips = self._merge_fallen_boxes(real_tips + diagonal_tips)

        if merged_tips:
            self._result.fallen_tips = merged_tips
        else:
            # 兜底：没有完整/对角线枪头，但仍可能有半透明反光碎片。
            # 真实散落枪头比暗色槽壁（mean≤77）亮得多，取 mean≥80 的碎片。
            bright_fragments: List[Tuple[float, float, float, float]] = []
            for fx, fy, fbw, fbh in fallen:
                x0 = max(0, int(fx - fbw / 2))
                y0 = max(0, int(fy - fbh / 2))
                x1 = min(gray.shape[1], int(fx + fbw / 2))
                y1 = min(gray.shape[0], int(fy + fbh / 2))
                if x1 <= x0 or y1 <= y0:
                    continue
                pixels = gray[y0:y1, x0:x1]
                if pixels.size > 0 and float(pixels.mean()) >= 80.0:
                    bright_fragments.append((fx, fy, fbw, fbh))
            self._result.fallen_tips = bright_fragments

    def _detect_diagonal_tips(
        self, frame: np.ndarray, slot_direction: str
    ) -> List[Tuple[float, float, float, float]]:
        """
        对角线散落枪头检测（霍夫线法）。

        当枪头从槽中掉落并斜跨在槽上时，从上方俯视呈现为一条偏离槽方向
        的细长斜线（约 25°~75°）。这类枪头的外接矩形接近正方形，无法用
        轮廓长宽比捕捉；但其两条长边在霍夫线变换中是两条近平行的长线段，
        据此可精确定位。

        步骤：
        1. Canny 边缘 + HoughLinesP 提取长线段
        2. 过滤出偏离槽方向 25°~75° 的对角线段
        3. 按"角度相近 + 中心邻近"聚类（同一枪头两条边聚为一簇）
        4. 每簇至少 2 条线，取其外接框作为枪头位置
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 90)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50,
            minLineLength=int(self._diag_tip_min_len), maxLineGap=20,
        )
        if lines is None or len(lines) == 0:
            return []

        lines = lines.reshape(-1, 4)
        slot_angle = 90.0 if slot_direction == "vertical" else 0.0

        # 1) 提取对角线段
        segments = []
        for x1, y1, x2, y2 in lines:
            length = np.hypot(x2 - x1, y2 - y1)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
            delta = abs(angle - slot_angle)
            delta = min(delta, 180.0 - delta)
            if (
                length >= self._diag_tip_min_len
                and self._diag_tip_min_angle <= delta <= self._diag_tip_max_angle
            ):
                segments.append(((x1 + x2) / 2, (y1 + y2) / 2, angle, x1, y1, x2, y2))

        if not segments:
            return []

        # 2) 并查集聚类：同一枪头两条边角度相近、中心邻近
        n = len(segments)
        parent = list(range(n))

        def _find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            cxi, cyi, angi = segments[i][0], segments[i][1], segments[i][2]
            for j in range(i + 1, n):
                cxj, cyj, angj = segments[j][0], segments[j][1], segments[j][2]
                if (
                    abs(angj - angi) < 20.0
                    and np.hypot(cxj - cxi, cyj - cyi) < self._diag_tip_cluster_dist
                ):
                    ri, rj = _find(i), _find(j)
                    if ri != rj:
                        parent[ri] = rj

        groups = {}
        for i in range(n):
            groups.setdefault(_find(i), []).append(segments[i])

        # 3) 每簇生成外接框
        border = max(80, int(min(h, w) * 0.06))
        boxes: List[Tuple[float, float, float, float]] = []
        for group in groups.values():
            if len(group) < 2:
                continue
            xs = [v for seg in group for v in (seg[3], seg[5])]
            ys = [v for seg in group for v in (seg[4], seg[6])]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            bw, bh = xmax - xmin, ymax - ymin
            if bw < 15 or bh < 15:
                continue
            if xmin < border or ymin < border or xmax > w - border or ymax > h - border:
                continue
            boxes.append((xmin + bw / 2, ymin + bh / 2, bw, bh))

        return boxes

    @staticmethod
    def _merge_fallen_boxes(
        boxes: List[Tuple[float, float, float, float]],
    ) -> List[Tuple[float, float, float, float]]:
        """合并重叠或邻近的散落枪头框，避免同一枪头被重复标注。"""
        merged: List[List[float]] = []
        for cx, cy, bw, bh in boxes:
            placed = False
            for m in merged:
                if (
                    abs(cx - m[0]) < (bw + m[2]) / 2
                    and abs(cy - m[1]) < (bh + m[3]) / 2
                ):
                    x1 = min(cx - bw / 2, m[0] - m[2] / 2)
                    y1 = min(cy - bh / 2, m[1] - m[3] / 2)
                    x2 = max(cx + bw / 2, m[0] + m[2] / 2)
                    y2 = max(cy + bh / 2, m[1] + m[3] / 2)
                    m[0], m[1], m[2], m[3] = (
                        (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1,
                    )
                    placed = True
                    break
            if not placed:
                merged.append([cx, cy, bw, bh])
        return [(m[0], m[1], m[2], m[3]) for m in merged]

    # ==================== 轮廓检测模式 ====================

    def _detect_contours(self, frame: np.ndarray):
        binary, edges = self._preprocess(frame)
        if edges is None:
            return

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if not self._is_good_contour(contour):
                continue

            center = self._calculate_center(contour)
            if center is None:
                continue

            angle = self._calculate_angle(contour)

            self._result.centers.append(center)
            self._result.angles.append(angle)

    def _preprocess(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self._gaussian_kernel, self._gaussian_kernel), 0)

        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        edges = cv2.Canny(binary, self._canny_thresh1, self._canny_thresh2)
        return binary, edges

    def _is_good_contour(self, contour: np.ndarray) -> bool:
        area = cv2.contourArea(contour)
        if area < self._min_area or area > self._max_area:
            return False

        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        if rect_area <= 0:
            return False
        rectangularity = area / rect_area
        if rectangularity < self._min_rectangularity:
            return False

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return False
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < self._min_circularity:
            return False

        return True

    def _calculate_center(self, contour: np.ndarray) -> Optional[Point]:
        m = cv2.moments(contour)
        if m["m00"] == 0:
            return None
        return (float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"]))

    def _calculate_angle(self, contour: np.ndarray) -> float:
        rect = cv2.minAreaRect(contour)
        angle = rect[2]
        if angle < -45:
            angle += 90.0
        return angle

    def _is_on_regular_vertical_grid(
        self, cx: float, cy: float, circles: List[Tuple[float, float, float]]
    ) -> bool:
        """
        判断候选点 (cx, cy) 是否落在竖槽图像的“规则网格”上。
        规则网格位置包括：圆心 Y，以及正常相邻间距（<=1.4 倍中位数）的中点 Y。
        若候选落在这些位置，则视为槽间阴影/正常间隙而非散落枪头。
        """
        # 找到候选所在的列（按 X 聚类）
        line = None
        best_line_dist = float("inf")
        for ccx, _, _ in circles:
            d = abs(cx - ccx)
            if d < best_line_dist:
                best_line_dist = d
                line = ccx
        if line is None or best_line_dist > self._cluster_tol:
            return False

        # 该列所有圆心的 Y 值
        ys = sorted([ccy for ccx, ccy, _ in circles if abs(ccx - line) < self._cluster_tol])
        if len(ys) < 2:
            return False

        # 计算该列相邻圆心的间距中位数
        gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        median_gap = float(np.median(gaps)) if gaps else 0.0
        if median_gap <= 0:
            return False

        tol = min(self._cluster_tol, median_gap * 0.25)

        # 检查候选 Y 是否接近某个圆心
        for y in ys:
            if abs(cy - y) < tol:
                return True

        # 检查候选 Y 是否接近“正常间距”的中点
        for i in range(len(ys) - 1):
            gap = ys[i + 1] - ys[i]
            if gap > median_gap * 1.4:
                # 异常大间距，不视为规则网格（可能是缺失枪头位置）
                continue
            mid = (ys[i] + ys[i + 1]) / 2.0
            if abs(cy - mid) < tol:
                return True

        return False

    # ==================== 整齐度判定 ====================

    def _check_neatness(self) -> bool:
        is_ok = True

        if len(self._result.centers) < self._min_tips:
            self._result.reasons.append(
                f"枪头数量不足(检测到 {len(self._result.centers)}, 需要 {self._min_tips})"
            )
            is_ok = False

        if self._detection_mode == "contour":
            if not self._check_angle_consistency():
                is_ok = False

        # 散落枪头：任何"非圆形"的可疑枪头都视为不整齐
        if len(self._result.fallen_tips) > self._fallen_tip_max_count:
            self._result.reasons.append(
                f"检测到 {len(self._result.fallen_tips)} 个散落/异形枪头"
            )
            is_ok = False

        if not self._check_slot_alignment():
            is_ok = False

        if not self._check_circle_density():
            is_ok = False

        if self._check_uniform_spacing:
            if not self._check_spacing_uniformity():
                is_ok = False

        if is_ok:
            self._result.reasons.clear()
            self._result.reasons.append("整齐")

        return is_ok

    def _check_angle_consistency(self) -> bool:
        if not self._result.angles:
            self._result.reasons.append("无角度数据")
            return False

        median_angle = median(self._result.angles)
        for i, angle in enumerate(self._result.angles):
            deviation = abs(angle - median_angle)
            if deviation > self._angle_thresh:
                self._result.reasons.append(
                    f"第{i + 1}个枪头角度偏差过大({deviation:.1f}°)"
                )
                return False
        return True

    def _check_slot_alignment(self) -> bool:
        """
        槽线对齐检查。
        将圆心按垂直于槽的方向聚类，为每个槽拟合一条直线（PCA）；
        计算每个圆心到最近槽线的垂直距离；
        若异常比例超过阈值，则判为不整齐。
        """
        if not self._result.centers:
            self._result.reasons.append("无中心点数据")
            return False

        direction = self._determine_slot_direction()

        # 拟合槽线
        fitted_lines = self._fit_slot_lines(direction)
        if not fitted_lines:
            self._result.reasons.append("枪头分布过于分散，无法形成有效槽线")
            return False

        deviations = []
        for point in self._result.centers:
            pt = np.array(point)
            dev = min(
                self._point_line_distance(pt, line_pt, line_dir)
                for line_pt, line_dir in fitted_lines
            )
            deviations.append(dev)

        self._result.slot_deviations = deviations

        outliers = [d for d in deviations if d > self._slot_alignment_tol]
        outlier_ratio = len(outliers) / len(deviations) if deviations else 0

        # 保存分行/分列结果用于可视化
        axis = "x" if direction == "vertical" else "y"
        self._result.rows = self._cluster_1d(
            self._result.centers, axis=axis, tolerance=self._cluster_tol
        )

        if outlier_ratio > self._outlier_ratio_thresh:
            self._result.reasons.append(
                f"有 {len(outliers)} 个枪头偏离槽中心线(偏差>{self._slot_alignment_tol:.0f}px, "
                f"占比 {outlier_ratio * 100:.1f}%)"
            )
            return False

        return True

    def _fit_slot_lines(
        self, direction: str
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        对每个有效槽（聚类簇）用 PCA 拟合一条直线，返回 [(直线上一点, 方向向量), ...]。
        可处理轻微旋转/透视导致的斜槽。
        """
        axis = "x" if direction == "vertical" else "y"
        clusters = self._cluster_1d(self._result.centers, axis=axis, tolerance=self._cluster_tol)

        lines: List[Tuple[np.ndarray, np.ndarray]] = []
        for c in clusters:
            if len(c) < self._min_tips_per_slot:
                continue
            pts = np.array(c, dtype=np.float64)
            mean = np.mean(pts, axis=0)
            if len(pts) == 1:
                # 退化为垂直/水平方向
                if direction == "vertical":
                    dir_vec = np.array([0.0, 1.0])
                else:
                    dir_vec = np.array([1.0, 0.0])
                lines.append((mean, dir_vec))
                continue

            centered = pts - mean
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            dir_vec = eigvecs[:, np.argmax(eigvals)]
            lines.append((mean, dir_vec))

        return lines

    @staticmethod
    def _point_line_distance(
        point: np.ndarray, line_pt: np.ndarray, line_dir: np.ndarray
    ) -> float:
        """点到直线的垂直距离"""
        norm = np.linalg.norm(line_dir)
        if norm < 1e-9:
            return float("inf")
        unit_dir = line_dir / norm
        vec = point - line_pt
        perp = vec - np.dot(vec, unit_dir) * unit_dir
        return float(np.linalg.norm(perp))

    def _check_circle_density(self) -> bool:
        """
        圆覆盖密度检查：若检测到的圆心凸包内，圆面积占比过低，
        说明机架大部分区域为空（大量枪头缺失/倾倒），判为不整齐。
        """
        if len(self._result.centers) < 10:
            # 点太少时不做密度判断，避免误伤局部拍摄/旋转图
            return True

        pts = np.array(self._result.centers, dtype=np.float32)
        hull = cv2.convexHull(pts)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            return True

        circle_area = sum(np.pi * r * r for r in self._result.radii)
        density = circle_area / hull_area
        min_density = 0.09

        if density < min_density:
            self._result.reasons.append(
                f"圆覆盖密度过低({density:.1%})，存在大量空位"
            )
            return False

        return True

    def _determine_slot_direction(self) -> str:
        """判断槽方向: vertical(竖槽) 或 horizontal(横槽)"""
        if self._slot_direction in ("horizontal", "vertical"):
            return self._slot_direction

        # auto: 分别对 X 和 Y 聚类，选择平均簇大小更大的方向
        x_clusters = self._cluster_1d(self._result.centers, axis="x", tolerance=self._cluster_tol)
        y_clusters = self._cluster_1d(self._result.centers, axis="y", tolerance=self._cluster_tol)

        nx = len(x_clusters)
        ny = len(y_clusters)
        avg_x = np.mean([len(c) for c in x_clusters]) if x_clusters else 0
        avg_y = np.mean([len(c) for c in y_clusters]) if y_clusters else 0

        x_valid = nx >= 2 and avg_x >= 2
        y_valid = ny >= 2 and avg_y >= 2

        if x_valid and y_valid:
            return "vertical" if avg_x >= avg_y else "horizontal"
        elif x_valid:
            return "vertical"
        elif y_valid:
            return "horizontal"
        else:
            # 默认根据图像方向
            return "vertical"

    def _get_slot_lines(self, direction: str) -> List[float]:
        """获取有效槽中心线坐标列表(过滤掉点数过少的噪声簇)"""
        axis = "x" if direction == "vertical" else "y"
        clusters = self._cluster_1d(self._result.centers, axis=axis, tolerance=self._cluster_tol)
        return [
            sum(p[0 if axis == "x" else 1] for p in c) / len(c)
            for c in clusters
            if len(c) >= self._min_tips_per_slot
        ]

    @staticmethod
    def _cluster_1d(points: List[Point], axis: str, tolerance: float) -> List[List[Point]]:
        """一维聚类"""
        if not points:
            return []

        idx = 1 if axis == "y" else 0
        sorted_points = sorted(points, key=lambda p: p[idx])

        clusters: List[List[Point]] = []
        current: List[Point] = [sorted_points[0]]

        for point in sorted_points[1:]:
            avg = sum(p[idx] for p in current) / len(current)
            if abs(point[idx] - avg) < tolerance:
                current.append(point)
            else:
                clusters.append(current)
                current = [point]

        if current:
            clusters.append(current)

        return clusters

    def _check_spacing_uniformity(self) -> bool:
        """严格间距均匀性检查(可选)"""
        if not self._result.centers:
            self._result.reasons.append("无中心点数据")
            return False

        direction = self._determine_slot_direction()
        axis = "y" if direction == "vertical" else "x"
        clusters = self._cluster_1d(self._result.centers, axis=axis, tolerance=self._cluster_tol)

        if not clusters:
            return True

        for row_idx, row in enumerate(clusters):
            if len(row) < 2:
                continue

            row_sorted = sorted(row, key=lambda p: p[0 if axis == "x" else 1])
            spaces = []
            for i in range(1, len(row_sorted)):
                d = row_sorted[i][0 if axis == "x" else 1] - row_sorted[i - 1][0 if axis == "x" else 1]
                if d > self._min_space:
                    spaces.append(d)

            if not spaces:
                continue

            ref_space = median(spaces)
            if ref_space <= 0:
                continue

            for space in spaces:
                error = abs(space - ref_space) / ref_space
                if error > self._space_ratio:
                    self._result.reasons.append(
                        f"第{row_idx + 1}行间距不均匀(误差 {error * 100:.1f}%)"
                    )
                    return False

        return True

    # ==================== 可视化 ====================

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """在帧上绘制检测结果，返回副本不修改原图"""
        if frame is None or frame.size == 0:
            return frame

        canvas = frame.copy()

        for i, center in enumerate(self._result.centers):
            cx, cy = int(center[0]), int(center[1])
            is_outlier = (
                i < len(self._result.slot_deviations)
                and self._result.slot_deviations[i] > self._slot_alignment_tol
            )

            color = (0, 0, 255) if is_outlier else (0, 255, 0)

            if self._detection_mode == "circle" and i < len(self._result.radii):
                r = int(self._result.radii[i])
                cv2.circle(canvas, (cx, cy), r, color, 2)
                cv2.circle(canvas, (cx, cy), max(r - 3, 3), (255, 0, 0), 1)
            else:
                cv2.ellipse(canvas, (cx, cy), (10, 6), self._result.angles[i], 0, 360, color, 2)
                angle_rad = self._result.angles[i] * np.pi / 180.0
                end_x = int(cx + 20 * np.cos(angle_rad))
                end_y = int(cy + 20 * np.sin(angle_rad))
                cv2.line(canvas, (cx, cy), (end_x, end_y), (0, 255, 255), 2)

            cv2.circle(canvas, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(canvas, str(i + 1), (cx - 8, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

        # 绘制散落枪头（用亮黄色矩形框标出）
        for idx, (cx, cy, w, h) in enumerate(self._result.fallen_tips, 1):
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(canvas, f"F{idx}", (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 结果横幅
        result_text = "TIDY" if self._result.is_neat else "MESSY"
        text_color = (0, 255, 0) if self._result.is_neat else (0, 0, 255)

        cv2.rectangle(canvas, (10, 10), (520, 90), (0, 0, 0), -1)
        cv2.rectangle(canvas, (10, 10), (520, 90), (255, 255, 255), 2)
        cv2.putText(canvas, result_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

        mode_text = (
            f"Mode: {self._detection_mode} | Tips: {len(self._result.centers)}"
            f" | Fallen: {len(self._result.fallen_tips)}"
        )
        cv2.putText(canvas, mode_text, (20, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return canvas






            

            

                


            



