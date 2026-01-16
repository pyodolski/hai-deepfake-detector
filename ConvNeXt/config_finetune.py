"""
CPU Fine-tuning 설정 - model_4 기반 추가 학습
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FineTuneConfig:
    # Paths - 로컬 데이터셋
    image_real_dir: str = "../dataset/image/Train/Real"
    image_fake_dir: str = "../dataset/image/Train/Fake"
    video_real_dir: str = "../dataset/video/real"
    video_fake_dir: str = "../dataset/video/fake"
    
    # 기존 모델 로드
    pretrained_model: str = "./checkpoints/model_6_finetune2/best_model.pt"
    
    output_dir: str = "./outputs_finetune3"
    checkpoint_dir: str = "./checkpoints/model_7_finetune3"
    
    # Model - model_4와 동일
    model_name: str = "convnext_small"
    pretrained: bool = False  # 기존 모델 로드할 것
    num_classes: int = 1
    
    # Training - CPU 최적화
    epochs: int = 3  # 짧게
    batch_size: int = 4  # CPU라서 작게
    num_workers: int = 0  # CPU는 0
    
    # Optimizer - Fine-tuning용
    optimizer: str = "adamw"
    learning_rate: float = 1e-5  # 매우 낮은 학습률 (fine-tuning)
    weight_decay: float = 1e-4
    
    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 0  # warmup 없음
    
    # Loss
    loss_fn: str = "bce_with_logits"
    label_smoothing: float = 0.1
    
    # Mixed Precision - CPU는 False
    use_amp: bool = False
    gradient_clip: float = 1.0
    
    # Data - CPU라서 작은 데이터셋
    image_size: int = 224
    face_detection: bool = False  # CPU라서 끔
    num_frames_per_video: int = 4  # 적게
    use_video: bool = False  # 이미지만 사용
    
    # Augmentation - 최소화
    random_resized_crop: bool = True
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0
    rotation_limit: int = 0
    jpeg_quality_range: tuple = (70, 100)
    gaussian_blur_prob: float = 0.0
    gaussian_noise_prob: float = 0.0
    color_jitter: bool = False
    brightness_contrast_prob: float = 0.0
    
    # Validation
    val_split: float = 0.1
    
    # Data Sampling - CPU라서 매우 작게 (다른 샘플 사용)
    max_samples_per_class: int = 1000  # 클래스당 1000개
    sample_offset: int = 1000  # 이전과 다른 샘플 (1000~2000번째)
    
    # Seed
    seed: int = 42
    
    # Device
    device: str = "cpu"
    
    # Early Stopping
    patience: int = 2
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
