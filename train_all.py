import os
import subprocess
import sys

def run_script(script_path):
    print(f"\n🚀 Running {os.path.basename(script_path)} ...\n")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("⚠️ ERRORS/WARNINGS:\n", result.stderr)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))  # detects src folder automatically
    src_dir = os.path.join(base_dir, "src")
    model_dir = os.path.join(base_dir, "model")
    data_dir = os.path.join(base_dir, "data")

    os.makedirs(model_dir, exist_ok=True)

    dna_script = os.path.join(src_dir, "train_dna.py")
    mri_script = os.path.join(src_dir, "train_mri.py")

    # Run DNA training
    if os.path.exists(dna_script):
        os.environ["CSV_PATH"] = os.path.join(data_dir, "dna_sequences.csv")
        os.environ["SAVE_PATH"] = os.path.join(model_dir, "dna_model.pth")
        run_script(dna_script)
    else:
        print("❌ train_dna.py not found in src/")

    # Run MRI training
    if os.path.exists(mri_script):
        os.environ["DATA_DIR"] = os.path.join(data_dir, "mri_images")
        os.environ["SAVE_PATH"] = os.path.join(model_dir, "mri_model.pth")
        run_script(mri_script)
    else:
        print("❌ train_mri.py not found in src/")

    print("\n✅ Training completed! Models saved in:", model_dir)
