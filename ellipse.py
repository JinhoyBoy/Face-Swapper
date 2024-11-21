import numpy as np
import math
import cv2

def draw_ellipse(img, center, axes, angle, start_angle, end_angle, color):
    
    h, k = center
    a, b = axes
    angle = math.radians(angle)  # 회전 각도를 라디안으로 변환

    # 시작 각도와 끝 각도를 라디안으로 변환
    start_angle = math.radians(start_angle)
    end_angle = math.radians(end_angle)

    # 각도에 따라 점을 계산
    for theta in np.linspace(start_angle, end_angle, 1000):  # 1000개의 점으로 타원 근사
        # 기본 타원 점 계산 (회전 전)
        x = a * math.cos(theta)
        y = b * math.sin(theta)

        # 회전 변환 적용
        x_rot = x * math.cos(angle) - y * math.sin(angle)
        y_rot = x * math.sin(angle) + y * math.cos(angle)

        # 이미지에 중심 좌표를 더해서 실제 좌표 계산
        px = int(h + x_rot)
        py = int(k + y_rot)

        # 이미지의 해당 좌표에 색상 적용
        if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:  # 이미지 범위 내 점만 그림
            img[py, px] = color


# 타원 그리기
#draw_ellipse(image, center, axes, angle, start_angle, end_angle, color)

# 이미지 출력
#cv2.imshow("Ellipse", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
