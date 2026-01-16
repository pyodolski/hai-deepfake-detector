"""
최종 모델로 추론 - Model 6 (Fine-tuned)
"""
from inference import inference
from pathlib import Path

# 설정 - Model 6 사용
MODEL_PATH = "./checkpoints/model_6_finetune2/best_model.pt"
TEST_DIR = "./open/test_data"
OUTPUT_DIR = "./submissions/convNeXt_6_finetune2"
OUTPUT_CSV = f"{OUTPUT_DIR}/submission.csv"

MODEL_NAME = "convnext_small"
IMAGE_SIZE = 224
BATCH_SIZE = 32
USE_FACE_DETECTION = True
NUM_FRAMES = 8
DEVICE = "cuda"

if __name__ == "__main__":
    # 출력 폴더 생성
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Model 6 (Fine-tuned) Inference")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Model Type: {MODEL_NAME}")
    print(f"Frames per video: {NUM_FRAMES}")
    print(f"Face Detection: {USE_FACE_DETECTION}")
    print(f"Output: {OUTPUT_CSV}")
    print("=" * 60)
    
    inference(
        model_path=MODEL_PATH,
        test_dir=TEST_DIR,
        output_csv=OUTPUT_CSV,
        model_name=MODEL_NAME,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        use_face_detection=USE_FACE_DETECTION,
        num_frames=NUM_FRAMES,
        device=DEVICE
    )
    
    print("\n" + "=" * 60)
    print("✓ Inference Complete!")
    print(f"Submit this file: {OUTPUT_CSV}")
    print("=" * 60)
