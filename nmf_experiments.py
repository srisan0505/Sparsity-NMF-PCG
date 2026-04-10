# Program Name: nmf_experiments.py
# Version: 4.0 
# Description: Generates and saves the Diagnostic Heatmap (Figure 3) 
#              using the Sparsity-Constrained NMF method.

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.decomposition import NMF
import glob
from google.colab import drive

# ==========================================
# 1. CONFIGURATION
# ==========================================
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})

CONF = {
    'SR': 2000,
    'N_FFT': 1024,
    'HOP_LEN': 256,
    'N_ATOMS': 8,
    'SPARSITY': 0.1,      # Matches the paper and cv script
    'FREQ_LIMIT': 1000,
    'DURATION': 5.0,
    'BASE_PATH': '/content/drive/MyDrive/PhysioNet2016',
    'RANDOM_STATE': 42
}

# ==========================================
# 2. GENERATE HEATMAP
# ==========================================
def generate_diagnostic_heatmap(file_path):
    print(f"Loading and processing: {os.path.basename(file_path)}")
    y, sr = librosa.load(file_path, sr=CONF['SR'], duration=CONF['DURATION']) 
    
    # STFT Spectrogram
    S_complex = librosa.stft(y, n_fft=CONF['N_FFT'], hop_length=CONF['HOP_LEN'])
    S_mag = np.abs(S_complex)
    
    # Filter to 0-1000 Hz
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=CONF['N_FFT'])
    valid_bins = freq_bins <= CONF['FREQ_LIMIT']
    V = S_mag[valid_bins, :]

    # Max-Min Normalization (identical to the CV script)
    V_max = np.max(V) + 1e-9
    V_norm = V / V_max 

    # NMF Decomposition
    print("Running NMF decomposition...")
    model = NMF(n_components=CONF['N_ATOMS'], init='nndsvda', 
                solver='mu', beta_loss='kullback-leibler', 
                alpha_H=CONF['SPARSITY'], l1_ratio=1.0, 
                random_state=CONF['RANDOM_STATE'], max_iter=500)
    
    W = model.fit_transform(V_norm) 
    H = model.components_      
    
    # --- Plotting Fig 3 ---
    plt.figure(figsize=(8, 4))
    
    # Subplot A: Original Noisy Spectrogram
    plt.subplot(1, 2, 1)
    librosa.display.specshow(librosa.amplitude_to_db(V_norm, ref=np.max), 
                             y_axis='linear', x_axis='time', sr=CONF['SR'], 
                             hop_length=CONF['HOP_LEN'])
    plt.title('(a) Input Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    # Subplot B: Isolated Atom (We pick the one with the highest energy)
    plt.subplot(1, 2, 2)
    atom_energies = [np.sum(H[k,:]) for k in range(CONF['N_ATOMS'])]
    best_atom_idx = np.argmax(atom_energies)
    
    Atom_Map = np.outer(W[:, best_atom_idx], H[best_atom_idx, :]) 
    librosa.display.specshow(librosa.amplitude_to_db(Atom_Map, ref=np.max), 
                             y_axis='linear', x_axis='time', sr=CONF['SR'], 
                             hop_length=CONF['HOP_LEN'])
    plt.title(f'(b) Diagnostic Heatmap (Atom #{best_atom_idx})')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    
    # Save the figure properly
    save_path = 'Fig3_Heatmap.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nSuccess! Saved as '{save_path}'")
    
    # Also attempt to save directly to Drive for convenience
    try:
        drive_path = os.path.join(CONF['BASE_PATH'], save_path)
        import shutil
        shutil.copy(save_path, drive_path)
        print(f"Also copied to: {drive_path}")
    except:
        pass

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    drive.mount('/content/drive', force_remount=True)

    # Find an abnormal file to visualize (Abnormal files have clear murmurs/issues)
    ref_files = glob.glob(os.path.join(CONF['BASE_PATH'], '**', 'REFERENCE.csv'), recursive=True)
    target_file = None
    
    for ref in ref_files:
        folder = os.path.dirname(ref)
        try:
            import pandas as pd
            df = pd.read_csv(ref, names=['file', 'label'], header=None)
            # Find the first abnormal (label 1) file
            abnormal_row = df[df['label'] == 1].iloc[0]
            target_file = os.path.join(folder, f"{abnormal_row['file']}.wav")
            break
        except:
            continue
            
    if target_file and os.path.exists(target_file):
        generate_diagnostic_heatmap(target_file)
    else:
        print("Could not find a target .wav file. Please check CONF['BASE_PATH'].")