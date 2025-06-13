import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.nifti_explorer import NIfTIExplorer
from src.data.dicom_explorer import DICOMExplorer
from src.data.unified_loader import UnifiedDataLoader
import matplotlib.pyplot as plt
import pandas as pd
import json

# Create output directories if they don't exist
outputs_dir = project_root / 'outputs'
outputs_dir.mkdir(exist_ok=True)

# 1. Analyze NIfTI files
print("=== NIfTI Analysis ===")
nifti_explorer = NIfTIExplorer(str(project_root / 'data/raw/nifti'))
nifti_results = nifti_explorer.analyze_dataset()
nifti_explorer.save_analysis(str(outputs_dir / 'nifti_analysis.json'))

# Summarize NIfTI dataset
nifti_summary = nifti_explorer.summarize_dataset()
print(f"\nNIfTI Dataset Summary:")
print(f"Total files: {nifti_summary['total_files']}")
print(f"Total size: {nifti_summary['total_size_gb']:.2f} GB")
print(f"Unique shapes: {len(nifti_summary['shapes'])}")
print(f"Unique spacings: {len(nifti_summary['spacings'])}")

# 2. Analyze DICOM files
print("\n=== DICOM Analysis ===")
dicom_explorer = DICOMExplorer(str(project_root / 'data/raw/dicom'))
dicom_results = dicom_explorer.analyze_all_series()

# Save DICOM analysis
with open(outputs_dir / 'dicom_analysis.json', 'w') as f:
    json.dump(dicom_results, f, indent=2)

# 3. Convert DICOM to NIfTI for easier processing
print("\n=== Converting DICOM to NIfTI ===")
dicom_explorer.convert_to_nifti(str(project_root / 'data/processed/dicom_converted'))

# 4. Test unified loader
print("\n=== Testing Unified Loader ===")
loader = UnifiedDataLoader()

# Load sample NIfTI
nifti_sample = list((project_root / 'data/raw/nifti').glob('*.nii*'))[0]
nifti_volume = loader.load(nifti_sample)
print(f"NIfTI loaded: shape={nifti_volume.shape}, spacing={nifti_volume.spacing}")

# Load sample DICOM
dicom_sample = list((project_root / 'data/raw/dicom').iterdir())[0]
if dicom_sample.is_dir():
    dicom_volume = loader.load(dicom_sample)
    print(f"DICOM loaded: shape={dicom_volume.shape}, spacing={dicom_volume.spacing}")

# 5. Visualize intensity distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# NIfTI intensity distribution
nifti_intensities = []
for result in nifti_results.values():
    if 'intensity' in result:
        nifti_intensities.append([
            result['intensity']['min'],
            result['intensity']['mean'],
            result['intensity']['max']
        ])

if nifti_intensities:
    nifti_df = pd.DataFrame(nifti_intensities, columns=['Min', 'Mean', 'Max'])
    nifti_df.boxplot(ax=axes[0, 0])
    axes[0, 0].set_title('NIfTI Intensity Distribution')
    axes[0, 0].set_ylabel('HU Value')

# Shape distribution
shapes = list(nifti_summary['shapes'].keys())
counts = list(nifti_summary['shapes'].values())
axes[0, 1].bar(range(len(shapes)), counts)
axes[0, 1].set_xticks(range(len(shapes)))
axes[0, 1].set_xticklabels([s[:20] for s in shapes], rotation=45)
axes[0, 1].set_title('Volume Shapes Distribution')
axes[0, 1].set_ylabel('Count')

# Spacing distribution
spacings = list(nifti_summary['spacings'].keys())
spacing_counts = list(nifti_summary['spacings'].values())
axes[1, 0].bar(range(len(spacings)), spacing_counts)
axes[1, 0].set_xticks(range(len(spacings)))
axes[1, 0].set_xticklabels([s[:20] for s in spacings], rotation=45)
axes[1, 0].set_title('Voxel Spacing Distribution')
axes[1, 0].set_ylabel('Count')

# Orientation distribution
orientations = list(nifti_summary['orientations'].keys())
orient_counts = list(nifti_summary['orientations'].values())
axes[1, 1].pie(orient_counts, labels=orientations, autopct='%1.1f%%')
axes[1, 1].set_title('Orientation Distribution')

plt.tight_layout()
plt.savefig(str(outputs_dir / 'data_analysis_summary.png'), dpi=300)
plt.show()

print("\nData exploration complete! Check outputs folder for detailed analysis.")