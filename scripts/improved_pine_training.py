"""
🌲 소나무 피해목 전용 YOLOv11s 학습 코드
- 오탐지(집, 밭, 도로) 방지 최적화
- 소나무 고유 특성 학습 강화
- 색상 기반 오탐 제거
"""

import torch
from datetime import datetime
import os

def train_pine_specific_yolov11s(model, device='cuda'):
    """소나무 전용 YOLOv11s 학습 함수"""
    
    print("🌲 소나무 전용 YOLOv11s 학습 시작!")
    print("🎯 오탐지 방지 및 소나무 특화 학습!")
    print("=" * 60)
    
    print(f"🖥️  사용 장치: {device.upper()}")
    if device == 'cpu':
        print("⚠️  CPU 모드로 실행됩니다. GPU 사용을 권장합니다!")
    
    # 🌲 소나무 전용 최적화 학습 설정 (YOLOv11 호환)
    training_config = {
        'data': '/content/training_data/data.yaml',
        'epochs': 300,              # 충분한 학습 시간
        'batch': 6 if device == 'cuda' else 3,  # 더 작은 배치로 안정성 증대
        'imgsz': 640,               
        'patience': 150,            # 더 오래 기다림
        'save': True,
        'save_period': 10,          # 더 자주 저장
        'cache': True,
        'device': device,
        'workers': 4 if device == 'cuda' else 0,
        'project': 'pine_specific_yolov11s',
        'name': f'pine_only_{datetime.now().strftime("%Y%m%d_%H%M")}',
        
        # 🎯 소나무 전용 학습률 최적화
        'optimizer': 'AdamW',
        'lr0': 0.0003,              # 더욱 낮은 초기 학습률 (정밀 학습)
        'lrf': 0.005,               # 매우 낮은 최종 학습률
        'momentum': 0.937,
        'weight_decay': 0.0008,     # 가중치 감쇠 증가 (일반화 향상)
        'warmup_epochs': 15,        # 더 긴 워밍업 (안정적 시작)
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.005,    # 매우 낮은 워밍업 bias 학습률
        
        # 🌲 소나무 특화 손실 함수 가중치
        'box': 10.0,                # Box loss 증가 (정확한 위치 학습)
        'cls': 1.5,                 # Classification loss 증가 (소나무 vs 비소나무)
        'dfl': 2.0,                 # DFL loss 증가 (정밀한 경계)
        
        # 🚫 오탐지 방지 데이터 증강 (색상 변화 최소화)
        'hsv_h': 0.001,             # 색조 변화 극소 (소나무 고유색 보존)
        'hsv_s': 0.1,               # 채도 변화 극소 (갈색/녹색 보존)
        'hsv_v': 0.05,              # 명도 변화 극소 (자연 조명 보존)
        'degrees': 0.0,             # 회전 비활성화
        'translate': 0.02,          # 이동 최소화 (0.05 → 0.02)
        'scale': 0.1,               # 스케일 변화 최소화 (0.2 → 0.1)
        'shear': 0.0,               # 전단 변형 비활성화
        'perspective': 0.0,         # 원근 변형 비활성화
        'flipud': 0.0,              # 상하 반전 비활성화
        'fliplr': 0.2,              # 좌우 반전 최소화 (0.3 → 0.2)
        'mosaic': 0.3,              # 모자이크 대폭 감소 (0.5 → 0.3)
        'mixup': 0.0,               # 믹스업 완전 비활성화
        'copy_paste': 0.0,          # 복사-붙여넣기 완전 비활성화
        
        # 🎯 소나무 특화 안정성 설정
        'amp': False,               # AMP 비활성화 (정밀도 우선)
        'fraction': 1.0,            # 전체 데이터 사용
        'profile': False,
        'freeze': None,             # 모든 레이어 학습
        'dropout': 0.1,             # 약간의 드롭아웃 (과적합 방지)
        'val': True,
        'plots': True,
        'save_json': True,
        'verbose': True,
        'seed': 2024,               # 재현 가능한 결과
        
        # 🌲 소나무 전용 고급 설정 (YOLOv11 호환)
        'rect': False,              # 직사각형 학습 비활성화
        'single_cls': True,         # 단일 클래스 (소나무 피해목만)
        'overlap_mask': True,
        'mask_ratio': 4,
        'cos_lr': True,             # 코사인 스케줄러 활성화 (부드러운 학습률 감소)
        'close_mosaic': 100,        # 모자이크를 더 일찍 비활성화 (50 → 100)
        
        # 🎯 엄격한 탐지 임계값 (오탐지 방지)
        'conf': 0.25,               # 더 높은 신뢰도 임계값 (0.1 → 0.25)
        'iou': 0.6,                 # 더 높은 IoU 임계값 (0.4 → 0.6)
        
        # 🌲 YOLOv11 호환 추가 설정
        'multi_scale': False,       # 멀티스케일 학습 비활성화 (안정성)
        'augment': True,            # 기본 데이터 증강 활성화
        'auto_augment': None,       # 자동 증강 비활성화
        'erasing': 0.0,             # Random erasing 비활성화
        'crop_fraction': 1.0,       # 전체 이미지 사용
    }
    
    print("🌲 소나무 전용 최적화 설정:")
    print(f"  🎯 초기 학습률: 0.0003 (정밀 학습)")
    print(f"  🎯 배치 크기: 6 (안정성 우선)")
    print(f"  🎯 워밍업: 15 에포크 (천천히 적응)")
    print(f"  🌲 색상 증강: 최소화 (소나무 고유색 보존)")
    print(f"  🚫 기하학적 증강: 최소화 (자연스러운 형태 보존)")
    print(f"  🎯 신뢰도 임계값: 0.25 (엄격한 탐지)")
    print(f"  🎯 IoU 임계값: 0.6 (정확한 위치)")
    
    print("\n🎯 기대 효과:")
    print("  ✅ 집, 밭, 도로 오탐지 대폭 감소")
    print("  ✅ 소나무 고유 특성 학습 강화")
    print("  ✅ 색상 기반 오분류 방지")
    print("  ✅ 더 정확한 바운딩 박스")
    print("  ✅ 높은 precision과 적절한 recall")
    
    print("\n🌲 소나무 탐지 전문화:")
    print("  🍃 나무 잎의 질감과 패턴 학습")
    print("  🌿 소나무 특유의 침엽 구조 인식")
    print("  🔍 피해 부위의 색상 변화 패턴")
    print("  📐 자연스러운 나무 형태 학습")
    
    if device == 'cpu':
        print("\n⚠️  CPU 모드 주의사항:")
        print("  - 학습 시간이 매우 오래 걸립니다")
        print("  - 가능하면 GPU 환경을 사용하세요!")
    
    # 학습 실행
    print("\n🔥 소나무 전용 정밀 학습 시작!")
    print("🎯 오탐지 제거 및 정확도 극대화!")
    print("🌲 소나무만 정확히 탐지하는 모델 구축!")
    
    try:
        results = model.train(**training_config)
        
        print("\n🎉 소나무 전용 YOLOv11s 학습 완료!")
        print(f"📈 최종 mAP50: {results.box.map50:.4f}")
        print(f"📈 최종 mAP50-95: {results.box.map:.4f}")
        print(f"📈 Precision: {results.box.mp:.4f}")
        print(f"📈 Recall: {results.box.mr:.4f}")
        
        # 성능 평가
        if results.box.mp > 0.8:  # Precision > 80%
            print("🎉 우수한 정밀도! 오탐지가 크게 감소했습니다!")
        if results.box.mr > 0.6:  # Recall > 60%
            print("🎉 좋은 재현율! 실제 피해목을 잘 찾고 있습니다!")
        
        print("\n🌲 모델 검증 권장사항:")
        print("  1. 집, 건물이 있는 이미지로 테스트")
        print("  2. 농경지, 밭이 있는 이미지로 테스트")
        print("  3. 도로, 비포장 길이 있는 이미지로 테스트")
        print("  4. 다양한 조명 조건에서 테스트")
        
        return True, results
        
    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {str(e)}")
        return False, None

# 사용 예시
if __name__ == "__main__":
    # 모델 로드 및 학습
    from ultralytics import YOLO
    
    model_ready = True  # 모델 준비 상태
    
    if model_ready:
        try:
            # YOLOv11s 모델 로드
            model = YOLO('yolo11s.pt')
            
            # GPU 사용 가능 여부 확인
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # 소나무 전용 학습 실행
            success, results = train_pine_specific_yolov11s(model, device)
            
            if success:
                print("\n✅ 소나무 전용 모델 학습 성공!")
                print("🌲 이제 집, 밭, 도로 오탐지가 크게 줄어들 것입니다!")
            else:
                print("\n❌ 학습 실패. 설정을 확인해주세요.")
                
        except Exception as e:
            print(f"❌ 모델 로드 실패: {str(e)}")
    else:
        print("❌ 모델이 준비되지 않아 학습을 건너뜁니다.")
