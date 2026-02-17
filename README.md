# LEO TDOA Simulation

LEO Satellite TDOA Localization Simulation - Simulates signal detection, direction finding, and TDOA-based localization from 500km LEO satellites detecting a ground-based emitter.

## Key Parameters

- Frequency: 14.25 GHz (Ku-band)
- Bandwidth: 62.5 MHz
- Modulation: OFDM
- Emitter EIRP: 62 dBm
- Antenna: 34x44 phased array
- Satellite altitude: 500 km

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python leo_tdoa_simulation.py
```

## Features

- TDOA-based localization simulation
- Link budget analysis
- GDOP calculation
- Monte Carlo analysis
- Interactive GUI with parameter controls
