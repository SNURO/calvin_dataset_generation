import cv2
import numpy as np

# 빈 이미지 생성 (흰색 배경)
img = np.ones((500, 500, 3), dtype=np.uint8) * 255

# 이미지에 텍스트 추가
cv2.putText(img, 'OpenCV Test', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

# 이미지 표시
cv2.imshow('Test Window', img)

# 키 입력 대기
print("Press any key to close the window...")
cv2.waitKey(1)

# 모든 창 닫기
cv2.destroyAllWindows()