# GeoGravityLab Architecture

```mermaid
graph TD

A[User]

A --> B[Interactive Geological Model Builder]

B --> C[Geological Model]

C --> D[Density Grid Generator]

D --> E[Forward Gravity Model]

E --> F[Sensitivity Matrix]

F --> G[Synthetic Gravity]

G --> H[Noise Simulation]

H --> I[FFT]

I --> J[Low Pass Filter]

J --> K[Tikhonov]

J --> L[Depth Weighted]

J --> M[IRLS]

K --> N[Visualization]

L --> N

M --> N

N --> O[Scientific Figures]