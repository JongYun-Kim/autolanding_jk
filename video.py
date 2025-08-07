# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import cv2
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math


class VideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def init_with_frame(self, env, enabled=True, action=None):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        if action is None:
            action = np.zeros(3, dtype=np.float32)
        self.record_with_frame(env, action)

    def record(self, env):
        if self.enabled:
            #if hasattr(env, 'physics'):
            #    frame = env.physics.render(height=self.render_size,
            #                               width=self.render_size,
            #                               camera_id=0)
            frame = env.render()
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)

    def record_with_frame(self, env, action: np.ndarray):
        """
        좌측 : env.render() (전체 장면)
        우측 하단 : upsample(×4)된 env.rgb
        우측 상단 패널 (세로):
            ┌ 화살표  ┐   action[0](+y), action[1](-x)
            └ Z-bar ┘   action[2] (z-velocity)
        패널 안에 각 값(소수점 2자리)을 텍스트로 함께 표기.
        """

        if not getattr(self, "enabled", False):
            return

        # ─────────────────────────────────────── 원본 / 카메라 이미지 준비
        full_img: Image.Image = env.render()           # PIL(RGBA)
        w1, h1 = full_img.size

        cam_arr = np.squeeze(env.rgb)                  # (84, 84)
        cam_img = Image.fromarray(cam_arr).convert("RGBA")  # → PIL
        SCALE = 4
        cam_img = cam_img.resize(
            (cam_img.width * SCALE, cam_img.height * SCALE),
            resample=Image.NEAREST
        )
        w2, h2 = cam_img.size

        # ─────────────────────────────────────── 패널 높이 결정 (화살표·Zbar)
        panel_h = h2 // 2               # 위(화살표)·아래(Z) 각각 h2/4
        total_h = max(h1, panel_h + h2)
        total_w = w1 + w2

        # ─────────────────────────────────────── 캔버스 생성 & 붙이기
        canvas = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 255))
        canvas.paste(full_img, (0, total_h - h1))      # 좌측 하단 정렬
        cam_x, cam_y = w1, total_h - h2
        canvas.paste(cam_img, (cam_x, cam_y))          # 우측 하단

        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", 18)
        except IOError:
            font = ImageFont.load_default()

        # ──────────────────────────────── ① 화살표 패널 (panel_top ~ panel_mid)
        panel_top = cam_y - panel_h
        arrow_h = panel_h // 2
        arrow_center = (cam_x + w2 // 2, panel_top + arrow_h // 2)

        # 매핑 확인 :
        #   a(=action[0]) >0  ⇒  +y(위)  ⇒  이미지 dy = -a*scale
        #   b(=action[1]) >0  ⇒  -x(왼) ⇒  dx = -b*scale
        ARROW_LEN = min(w2, arrow_h) * 0.6     # 길이를 패널의 60 %로
        dx = -float(action[1]) * ARROW_LEN
        dy = -float(action[0]) * ARROW_LEN
        end_x = arrow_center[0] + dx
        end_y = arrow_center[1] + dy

        # 화살표 (머리 없음)
        draw.line([arrow_center, (end_x, end_y)], width=6, fill="white")

        # 값 라벨 - 화살표 패널 하단 왼쪽
        label_xy = f"X:{action[1]:+.2f}\nY:{action[0]:+.2f}"
        bbox_xy = draw.textbbox((0, 0), label_xy, font=font)
        lbl_x = cam_x + 6
        lbl_y = panel_top + arrow_h - (bbox_xy[3] - bbox_xy[1]) - 4
        draw.multiline_text((lbl_x, lbl_y), label_xy, font=font, fill="white")

        # ──────────────────────────────── ② Z-bar 패널 (panel_mid ~ panel_bot)
        bar_top = panel_top + arrow_h
        bar_h = panel_h - arrow_h
        BAR_W = w2 // 8
        bar_x0 = cam_x + (w2 - BAR_W) // 2
        bar_x1 = bar_x0 + BAR_W
        mid_y = bar_top + bar_h // 2

        # 외곽선
        draw.rectangle(
            [bar_x0, bar_top + 4, bar_x1, bar_top + bar_h - 4],
            outline="white",
            width=3
        )

        # 내부 채우기
        z_val = float(np.clip(action[2], -1, 1))
        fill_len = int((bar_h - 8) * abs(z_val) / 2 * 1.0)  # 1.0 배 확대
        if z_val >= 0:
            y0, y1 = mid_y - fill_len, mid_y
        else:
            y0, y1 = mid_y, mid_y + fill_len
        draw.rectangle([bar_x0, y0, bar_x1, y1], fill="white")

        # Z값 라벨 - 바 오른쪽 중앙
        z_label = f"Z:{z_val:+.2f}"
        bbox_z = draw.textbbox((0, 0), z_label, font=font)
        z_lbl_x = bar_x1 + 6
        z_lbl_y = mid_y - (bbox_z[3] - bbox_z[1]) // 2
        draw.text((z_lbl_x, z_lbl_y), z_label, font=font, fill="white")

        # ─────────────────────────────────────── 프레임 저장
        self.frames.append(canvas)


class TrainVideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / 'train_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            frame = cv2.resize(obs[-3:].transpose(1, 2, 0),
                               dsize=(self.render_size, self.render_size),
                               interpolation=cv2.INTER_CUBIC)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
