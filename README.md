# IonQ-Encode
Practical Guidelines for Quantum Data Encodings

## Overview
**IonQ-Encode** is a research project that benchmarks various quantum data encoding methods for machine learning tasks on the **IonQ trapped-ion hardware**.  
Our primary focus is to evaluate the trade-offs between accuracy, resource efficiency, and noise robustness across different encoding strategies.  

This project aims to provide **practical guidelines** for choosing the most suitable encoding methods when working with mid-scale datasets on near-term quantum devices.  

---

## Research Objectives
1. **Benchmark Encodings**  
   - Angle Encoding (AE)  
   - Data Re-Uploading (DRU)  
   - Amplitude Encoding (exact & approximate)  
   - Hybrid Embeddings (e.g., Angle + Amplitude)  
   - Kernel Feature Maps (ZZFeatureMap, IQP)  
   - Quantum Kitchen Sinks (QKS)

2. **Evaluate Trade-offs**  
   - Accuracy and generalization performance  
   - Resource costs: gate count, depth, shots, runtime  
   - Robustness under realistic IonQ noise models  

3. **Provide IonQ-Specific Guidelines**  
   - Optimal encoding per dataset size (PCA-16, PCA-32, PCA-64)  
   - Recommended circuit depth and shot count  
   - Practical error mitigation strategies (ZNE, readout calibration, etc.)  

---

## Dataset & Tasks
- **Dataset**: Fashion-MNIST (downsampled to 8×8, PCA-32/64 variants)  
- **Tasks**:  
  - **T2**: Binary classification (T-shirt vs Pullover)  
  - **T4**: 4-class classification (T-shirt, Pullover, Coat, Shirt)  
  - **T10**: Reduced 10-class classification  
