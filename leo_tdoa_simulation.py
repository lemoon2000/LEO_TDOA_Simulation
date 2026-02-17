"""
LEO Satellite TDOA Localization Simulation
============================================
Simulates signal detection, direction finding, and TDOA-based localization
from 500km LEO satellites detecting a ground-based emitter.

Key Parameters:
- Frequency: 14.25 GHz (Ku-band)
- Bandwidth: 62.5 MHz
- Modulation: OFDM
- Emitter EIRP: 62 dBm
- Antenna: 34x44 phased array
- Satellite altitude: 500 km
"""

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

from antenna_pattern import PhasedArrayAntenna

# Configure matplotlib for Chinese font display
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Physical Constants
C = 299792458.0  # Speed of light (m/s)
EARTH_RADIUS = 6371e3  # Earth radius (m)
K = 1.380649e-23  # Boltzmann constant (J/K)
T0 = 290.0  # Reference temperature (K)


@dataclass
class SimulationParameters:
    frequency: float = 14.25e9  # Hz
    bandwidth: float = 62.5e6  # Hz
    emitter_eirp: float = 62.0  # dBm
    antenna_nx: int = 34  # Phased array elements in X
    antenna_ny: int = 44  # Phased array elements in Y
    satellite_altitude: float = 500e3  # m
    num_satellites: int = 4
    satellite_spacing: float = 300e3  # m (along-track spacing)
    required_accuracy: float = 100.0  # m
    noise_figure: float = 3.0  # dB
    satellite_antenna_gain: float = 30.0  # dBi (satellite receive antenna)
    integration_time: float = 1e-3  # s
    num_averages: int = 10
    
    @property
    def wavelength(self) -> float:
        return C / self.frequency
    
    @property
    def wavelength_m(self) -> float:
        return self.wavelength


@dataclass
class Satellite:
    id: int
    position: np.ndarray  # [x, y, z] in local coordinates (km)
    velocity: np.ndarray  # [vx, vy, vz] (km/s)
    
    @property
    def altitude(self) -> float:
        return self.position[2]


@dataclass
class Emitter:
    position: np.ndarray  # [x, y, z] in local coordinates (km)
    eirp_dbm: float
    antenna: Optional[PhasedArrayAntenna] = None
    pointing_direction: Optional[np.ndarray] = None


class TDOALocalizationSimulator:
    def __init__(self, params: SimulationParameters):
        self.params = params
        self.satellites: List[Satellite] = []
        self.emitter: Optional[Emitter] = None
        self.estimated_position: Optional[np.ndarray] = None
        self.position_error: Optional[float] = None
        self.tdoa_measurements: Optional[np.ndarray] = None
        self.snr_values: Optional[List[float]] = None
        self.rx_power_values: Optional[List[float]] = None
        self.off_axis_angles: Optional[List[float]] = None
        self.gdop: Optional[float] = None
        
        self.emitter_antenna = PhasedArrayAntenna(
            nx=params.antenna_nx,
            ny=params.antenna_ny,
            frequency=params.frequency
        )
        
    def setup_scenario(self, emitter_offset_x: float = 0.0, emitter_offset_y: float = 0.0):
        self._create_satellites()
        self._create_emitter(emitter_offset_x, emitter_offset_y)
        
    def _create_satellites(self):
        self.satellites = []
        altitude = self.params.satellite_altitude / 1e3  # Convert to km
        spacing = self.params.satellite_spacing / 1e3    # Convert to km
        num_sats = self.params.num_satellites
        
        R = EARTH_RADIUS / 1e3  # Earth radius in km
        
        sat1_pos = np.array([0.0, 0.0, altitude])
        sat1_vel = np.array([0.0, 0.0, 0.0])
        self.satellites.append(Satellite(id=0, position=sat1_pos, velocity=sat1_vel))
        
        if num_sats >= 4:
            for i in range(3):
                angle = i * 2 * np.pi / 3
                
                alpha = spacing / (2 * (R + altitude))
                
                x = spacing * np.sqrt(1 - alpha**2) * np.cos(angle)
                y = spacing * np.sqrt(1 - alpha**2) * np.sin(angle)
                z = altitude - spacing**2 / (2 * (R + altitude))
                
                position = np.array([x, y, z])
                velocity = np.array([0.0, 0.0, 0.0])
                
                self.satellites.append(Satellite(id=i+1, position=position, velocity=velocity))
        else:
            for i in range(1, num_sats):
                angle = (i - 1) * 2 * np.pi / (num_sats - 1)
                
                alpha = spacing / (2 * (R + altitude))
                x = spacing * np.sqrt(1 - alpha**2) * np.cos(angle)
                y = spacing * np.sqrt(1 - alpha**2) * np.sin(angle)
                z = altitude - spacing**2 / (2 * (R + altitude))
                
                position = np.array([x, y, z])
                velocity = np.array([0.0, 0.0, 0.0])
                
                self.satellites.append(Satellite(id=i, position=position, velocity=velocity))
    
    def _create_emitter(self, offset_x: float = 0.0, offset_y: float = 0.0):
        position = np.array([offset_x, offset_y, 0.0])
        self.emitter = Emitter(
            position=position,
            eirp_dbm=self.params.emitter_eirp,
            antenna=self.emitter_antenna,
            pointing_direction=np.array([0.0, 0.0, 1.0])
        )
    
    def calculate_link_budget(self, sat: Satellite) -> Tuple[float, float, float]:
        if self.emitter is None:
            raise ValueError("Emitter not set")
        
        emitter_pos = self.emitter.position * 1e3
        sat_pos = sat.position * 1e3
        
        distance = np.linalg.norm(sat_pos - emitter_pos)
        
        fspl_db = 20 * np.log10(distance) + 20 * np.log10(self.params.frequency) + 20 * np.log10(4 * np.pi / C)
        
        if self.emitter.pointing_direction is not None:
            sat_direction = (sat_pos - emitter_pos) / distance
            
            az_deg, el_deg = self._calculate_az_el(self.emitter.pointing_direction, sat_direction)
            
            antenna_gain_loss = self._get_antenna_gain_loss(az_deg, el_deg)
        else:
            antenna_gain_loss = 0
        
        received_power_dbm = self.emitter.eirp_dbm - fspl_db - antenna_gain_loss + self.params.satellite_antenna_gain
        
        noise_power_dbm = 10 * np.log10(K * T0 * self.params.bandwidth * 10**(self.params.noise_figure / 10) * 1000)
        
        snr_db = received_power_dbm - noise_power_dbm
        
        return received_power_dbm, snr_db, distance
    
    def _calculate_az_el(self, pointing: np.ndarray, direction: np.ndarray) -> Tuple[float, float]:
        az_deg = np.degrees(np.arctan2(direction[0], direction[2]))
        el_deg = np.degrees(np.arctan2(direction[1], direction[2]))
        
        return az_deg, el_deg
    
    def _get_antenna_gain_loss(self, az_deg: float, el_deg: float) -> float:
        main_lobe_gain = self.emitter_antenna.calculate_gain_db(0, 0, 0, 0)
        gain_at_angle = self.emitter_antenna.calculate_gain_db(el_deg, az_deg, 0, 0)
        
        gain_loss = main_lobe_gain - gain_at_angle
        return max(0, gain_loss)
    
    def calculate_tdoa_precision(self, snr_db: float) -> float:
        snr_linear = 10 ** (snr_db / 10)
        
        theoretical_time_rmse = 1 / (2 * np.pi * self.params.bandwidth * np.sqrt(snr_linear * self.params.num_averages))
        
        clock_sync_error = 5e-9  # 5ns clock synchronization error
        atmospheric_error = 2e-9  # 2ns atmospheric delay uncertainty
        multipath_error = 3e-9  # 3ns multipath error
        measurement_noise = 5e-9  # 5ns additional measurement noise
        
        total_time_rmse = np.sqrt(theoretical_time_rmse**2 + clock_sync_error**2 + 
                                   atmospheric_error**2 + multipath_error**2 + measurement_noise**2)
        
        return total_time_rmse
    
    def simulate_tdoa_measurements(self) -> np.ndarray:
        if self.emitter is None or len(self.satellites) < 2:
            raise ValueError("Scenario not properly set up")
        
        emitter_pos = self.emitter.position * 1e3  # Convert km to m
        true_distances = [np.linalg.norm(sat.position * 1e3 - emitter_pos) for sat in self.satellites]
        true_tdoas = [(true_distances[i] - true_distances[0]) / C for i in range(len(self.satellites))]
        
        self.snr_values = []
        self.rx_power_values = []
        self.off_axis_angles = []
        time_errors = []
        
        pointing = np.array([0.0, 0.0, 1.0])
        
        for sat in self.satellites:
            rx_power, snr_db, _ = self.calculate_link_budget(sat)
            self.rx_power_values.append(rx_power)
            self.snr_values.append(snr_db)
            
            sat_pos = sat.position * 1e3
            direction = (sat_pos - emitter_pos) / np.linalg.norm(sat_pos - emitter_pos)
            off_axis = np.degrees(np.arccos(np.clip(np.dot(pointing, direction), -1, 1)))
            self.off_axis_angles.append(off_axis)
            
            time_rmse = self.calculate_tdoa_precision(snr_db)
            time_errors.append(time_rmse)
        
        measured_tdoas = np.array(true_tdoas, dtype=float)
        for i in range(1, len(measured_tdoas)):
            combined_error = np.sqrt(time_errors[0]**2 + time_errors[i]**2)
            measured_tdoas[i] += np.random.normal(0, combined_error)
        
        self.tdoa_measurements = measured_tdoas
        return measured_tdoas
    
    def tdoa_localization(self, tdoa_measurements: np.ndarray) -> np.ndarray:
        if len(self.satellites) < 3:
            raise ValueError("Need at least 3 satellites for TDOA localization")
        
        sat_positions = np.array([sat.position for sat in self.satellites])
        
        return self._tdoa_numerical_optimization(sat_positions, tdoa_measurements)
    
    def _tdoa_chan_algorithm(self, stations: np.ndarray, tdoa_measurements: np.ndarray) -> np.ndarray:
        return self._tdoa_numerical_optimization(stations, tdoa_measurements)
    
    def _tdoa_numerical_optimization(self, stations: np.ndarray, tdoa_measurements: np.ndarray) -> np.ndarray:
        stations_m = stations * 1e3
        
        def cost_function(target_pos):
            distances = np.array([np.linalg.norm(sat_pos - target_pos) for sat_pos in stations_m])
            predicted_tdoas = (distances - distances[0]) / C
            return np.sum((predicted_tdoas[1:] - tdoa_measurements[1:]) ** 2)
        
        from scipy.optimize import differential_evolution
        
        avg_sat_pos = np.mean(stations_m, axis=0)

        bounds = [(-1000e3, 1000e3), (-1000e3, 1000e3), (-10e3, 10e3)]
        
        result = differential_evolution(
            cost_function,
            bounds,
            maxiter=500,
            tol=1e-10,
            seed=42
        )
        
        position_km = result.x / 1e3
        self.estimated_position = position_km
        return position_km
    
    def _initial_guess(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0])
    
    def calculate_gdop(self) -> float:
        if len(self.satellites) < 3:
            return float('inf')
        
        sat_positions = np.array([sat.position for sat in self.satellites])
        
        if self.estimated_position is None:
            if self.emitter is None:
                return float('inf')
            target_pos = self.emitter.position
        else:
            target_pos = self.estimated_position
        
        H = []
        for i in range(1, len(sat_positions)):
            diff_i = target_pos - sat_positions[i]
            dist_i = np.linalg.norm(diff_i)
            unit_vec_i = diff_i / dist_i
            
            diff_0 = target_pos - sat_positions[0]
            dist_0 = np.linalg.norm(diff_0)
            unit_vec_0 = diff_0 / dist_0
            
            row = unit_vec_i - unit_vec_0
            H.append(row)
        
        H = np.array(H)
        
        try:
            H_pseudo_inv = np.linalg.pinv(H.T @ H)
            gdop = np.sqrt(np.trace(H_pseudo_inv))
        except Exception:
            gdop = float('inf')
        
        self.gdop = gdop
        return gdop
    
    def calculate_required_satellite_spacing(self) -> float:
        required_accuracy = self.params.required_accuracy
        
        time_precision = self._estimate_time_precision()
        
        range_precision = time_precision * C
        
        time_res = 1 / self.params.bandwidth
        range_res = time_res * C
        
        gdop_estimate = 2.0
        
        min_baseline = (required_accuracy * C) / (gdop_estimate * time_precision * C)
        
        return min_baseline
    
    def _estimate_time_precision(self) -> float:
        avg_snr = 20  # dB typical
        snr_linear = 10 ** (avg_snr / 10)
        return 1 / (2 * np.pi * self.params.bandwidth * np.sqrt(snr_linear * self.params.num_averages))
    
    def run_simulation(self, emitter_offset_x: float = 0.0, emitter_offset_y: float = 0.0) -> dict:
        self.setup_scenario(emitter_offset_x, emitter_offset_y)
        
        tdoa_meas = self.simulate_tdoa_measurements()
        
        estimated_pos = self.tdoa_localization(tdoa_meas)
        
        gdop = self.calculate_gdop()
        
        true_pos = self.emitter.position
        self.position_error = np.linalg.norm((estimated_pos - true_pos) * 1e3)  # Convert to meters
        
        required_spacing = self.calculate_required_satellite_spacing()
        
        return {
            'true_position': true_pos,
            'estimated_position': estimated_pos,
            'position_error': self.position_error,
            'gdop': gdop,
            'snr_values': self.snr_values,
            'rx_power_values': self.rx_power_values,
            'off_axis_angles': self.off_axis_angles,
            'tdoa_measurements': tdoa_meas,
            'required_spacing': required_spacing,
            'satellites': self.satellites
        }


class MonteCarloAnalysis:
    def __init__(self, params: SimulationParameters):
        self.params = params
        
    def run_analysis(self, num_runs: int = 20, spacing_range: Tuple[float, float] = (50e3, 300e3), 
                     progress_callback: Optional[Callable] = None) -> dict:
        spacings = np.linspace(spacing_range[0], spacing_range[1], 10)
        mean_errors = []
        std_errors = []
        gdop_values = []
        
        total_steps = len(spacings)
        
        for idx, spacing in enumerate(spacings):
            if progress_callback:
                progress_callback(idx + 1, total_steps, f"分析间距 {spacing/1e3:.0f} km...")
            
            self.params.satellite_spacing = spacing
            errors = []
            gdops = []
            
            for _ in range(num_runs):
                sim = TDOALocalizationSimulator(self.params)
                try:
                    result = sim.run_simulation()
                    errors.append(result['position_error'])
                    gdops.append(result['gdop'])
                except:
                    continue
            
            if errors:
                mean_errors.append(np.mean(errors))
                std_errors.append(np.std(errors))
                gdop_values.append(np.mean(gdops))
            else:
                mean_errors.append(float('inf'))
                std_errors.append(0)
                gdop_values.append(float('inf'))
        
        return {
            'spacings': spacings,
            'mean_errors': np.array(mean_errors),
            'std_errors': np.array(std_errors),
            'gdop_values': np.array(gdop_values)
        }
    
    def find_minimum_spacing(self, target_accuracy: float = 100.0, num_runs: int = 10,
                             progress_callback: Optional[Callable] = None) -> float:
        time_precision = 8e-9
        
        total_steps = 199
        
        for idx, spacing_km in enumerate(range(10, 2000, 10)):
            if progress_callback:
                progress_callback(idx + 1, total_steps, f"搜索最小间距... 当前: {spacing_km} km")
            
            self.params.satellite_spacing = spacing_km * 1e3
            sim = TDOALocalizationSimulator(self.params)
            sim.setup_scenario(0, 0)
            
            sat_positions = np.array([sat.position for sat in sim.satellites])
            target_pos = np.array([0.0, 0.0, 0.0])
            
            H = []
            for i in range(1, len(sat_positions)):
                diff = sat_positions[i] - target_pos
                dist = np.linalg.norm(diff)
                unit_vec = diff / dist
                ref_diff = sat_positions[0] - target_pos
                ref_dist = np.linalg.norm(ref_diff)
                ref_unit = ref_diff / ref_dist
                
                row = unit_vec - ref_unit
                H.append(row)
            
            H = np.array(H)
            
            try:
                H_pseudo_inv = np.linalg.pinv(H.T @ H)
                gdop = np.sqrt(np.trace(H_pseudo_inv))
            except Exception:
                gdop = float('inf')
            
            estimated_error = gdop * time_precision * C
            
            if estimated_error <= target_accuracy:
                return spacing_km * 1e3
        
        return 2000e3


class LEOSimulationGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LEO卫星TDOA定位仿真系统")
        self.root.geometry("1600x900")
        
        self.params = SimulationParameters()
        self.simulator = TDOALocalizationSimulator(self.params)
        self.monte_carlo = MonteCarloAnalysis(self.params)
        
        self.last_result = None
        
        self._setup_ui()
        self._run_initial_simulation()
    
    def _setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        left_panel = ttk.Frame(main_frame, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        left_panel.pack_propagate(False)
        
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self._create_parameter_panel(left_panel)
        self._create_visualization_panel(right_panel)
        self._create_results_panel(left_panel)
    
    def _create_parameter_panel(self, parent):
        params_frame = ttk.LabelFrame(parent, text="仿真参数设置", padding=10)
        params_frame.pack(fill=tk.X, pady=5)
        
        row = 0
        
        ttk.Label(params_frame, text="信号参数").grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(5,2))
        row += 1
        
        ttk.Label(params_frame, text="频率 (GHz):").grid(row=row, column=0, sticky=tk.W)
        self.freq_var = tk.StringVar(value="14.25")
        ttk.Entry(params_frame, textvariable=self.freq_var, width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(params_frame, text="带宽 (MHz):").grid(row=row, column=0, sticky=tk.W)
        self.bw_var = tk.StringVar(value="62.5")
        ttk.Entry(params_frame, textvariable=self.bw_var, width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(params_frame, text="发射EIRP (dBm):").grid(row=row, column=0, sticky=tk.W)
        self.eirp_var = tk.StringVar(value="62")
        ttk.Entry(params_frame, textvariable=self.eirp_var, width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(params_frame, text="天线阵元 Nx:").grid(row=row, column=0, sticky=tk.W)
        self.ant_nx_var = tk.StringVar(value="34")
        ttk.Entry(params_frame, textvariable=self.ant_nx_var, width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(params_frame, text="天线阵元 Ny:").grid(row=row, column=0, sticky=tk.W)
        self.ant_ny_var = tk.StringVar(value="44")
        ttk.Entry(params_frame, textvariable=self.ant_ny_var, width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Separator(params_frame, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=10)
        row += 1
        
        ttk.Label(params_frame, text="卫星参数").grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(5,2))
        row += 1
        
        ttk.Label(params_frame, text="卫星高度 (km):").grid(row=row, column=0, sticky=tk.W)
        self.alt_var = tk.StringVar(value="500")
        ttk.Entry(params_frame, textvariable=self.alt_var, width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(params_frame, text="卫星数量:").grid(row=row, column=0, sticky=tk.W)
        self.num_sat_var = tk.StringVar(value="4")
        ttk.Entry(params_frame, textvariable=self.num_sat_var, width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(params_frame, text="卫星间距 (km):").grid(row=row, column=0, sticky=tk.W)
        self.spacing_var = tk.StringVar(value="300")
        ttk.Entry(params_frame, textvariable=self.spacing_var, width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(params_frame, text="接收天线增益 (dBi):").grid(row=row, column=0, sticky=tk.W)
        self.ant_gain_var = tk.StringVar(value="30")
        ttk.Entry(params_frame, textvariable=self.ant_gain_var, width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(params_frame, text="噪声系数 (dB):").grid(row=row, column=0, sticky=tk.W)
        self.nf_var = tk.StringVar(value="3")
        ttk.Entry(params_frame, textvariable=self.nf_var, width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Separator(params_frame, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=10)
        row += 1
        
        ttk.Label(params_frame, text="发射源位置(相对原点偏移)").grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(5,2))
        row += 1
        
        ttk.Label(params_frame, text="X偏移 (km):").grid(row=row, column=0, sticky=tk.W)
        self.lat_var = tk.StringVar(value="0.0")
        ttk.Entry(params_frame, textvariable=self.lat_var, width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(params_frame, text="Y偏移 (km):").grid(row=row, column=0, sticky=tk.W)
        self.lon_var = tk.StringVar(value="0.0")
        ttk.Entry(params_frame, textvariable=self.lon_var, width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Separator(params_frame, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=10)
        row += 1
        
        ttk.Label(params_frame, text="定位要求").grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(5,2))
        row += 1
        
        ttk.Label(params_frame, text="目标精度 (m):").grid(row=row, column=0, sticky=tk.W)
        self.accuracy_var = tk.StringVar(value="100")
        ttk.Entry(params_frame, textvariable=self.accuracy_var, width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        button_frame = ttk.Frame(params_frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=15)
        
        ttk.Button(button_frame, text="运行仿真", command=self._run_simulation).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="蒙特卡洛分析", command=self._run_monte_carlo).pack(side=tk.LEFT, padx=5)
    
    def _create_results_panel(self, parent):
        results_frame = ttk.LabelFrame(parent, text="仿真结果", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.results_text = tk.Text(results_frame, height=20, width=40, font=('Consolas', 9))
        self.results_text.pack(fill=tk.BOTH, expand=True)
    
    def _create_visualization_panel(self, parent):
        self.fig = Figure(figsize=(12, 8), dpi=100)
        
        self.ax_3d = self.fig.add_subplot(221)
        self.ax_2d = self.fig.add_subplot(222)
        self.ax_error = self.fig.add_subplot(223)
        self.ax_gdop = self.fig.add_subplot(224)
        
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _update_parameters(self):
        try:
            self.params.frequency = float(self.freq_var.get()) * 1e9
            self.params.bandwidth = float(self.bw_var.get()) * 1e6
            self.params.emitter_eirp = float(self.eirp_var.get())
            self.params.antenna_nx = int(self.ant_nx_var.get())
            self.params.antenna_ny = int(self.ant_ny_var.get())
            self.params.satellite_altitude = float(self.alt_var.get()) * 1e3
            self.params.num_satellites = int(self.num_sat_var.get())
            self.params.satellite_spacing = float(self.spacing_var.get()) * 1e3
            self.params.satellite_antenna_gain = float(self.ant_gain_var.get())
            self.params.noise_figure = float(self.nf_var.get())
            self.params.required_accuracy = float(self.accuracy_var.get())
            return True
        except ValueError as e:
            messagebox.showerror("参数错误", f"请检查输入参数: {e}")
            return False
    
    def _run_simulation(self):
        if not self._update_parameters():
            return
        
        try:
            self.simulator = TDOALocalizationSimulator(self.params)
            emitter_offset_x = float(self.lat_var.get())
            emitter_offset_y = float(self.lon_var.get())
            
            self.last_result = self.simulator.run_simulation(emitter_offset_x, emitter_offset_y)
            
            self._update_visualization()
            self._update_results()
            
        except Exception as e:
            messagebox.showerror("仿真错误", f"仿真过程中出错: {e}")
    
    def _run_monte_carlo(self):
        if not self._update_parameters():
            return
        
        progress_window = tk.Toplevel(self.root)
        progress_window.title("蒙特卡洛分析进度")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        ttk.Label(progress_window, text="正在进行蒙特卡洛分析...", font=('Arial', 10)).pack(pady=10)
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100, length=350)
        progress_bar.pack(pady=5)
        
        status_label = ttk.Label(progress_window, text="初始化...")
        status_label.pack(pady=5)
        
        self.root.update()
        
        def update_progress(current, total, message):
            progress_var.set((current / total) * 100)
            status_label.config(text=message)
            progress_window.update()
        
        try:
            min_spacing = self.monte_carlo.find_minimum_spacing(
                target_accuracy=self.params.required_accuracy,
                num_runs=10,
                progress_callback=update_progress
            )
            
            mc_result = self.monte_carlo.run_analysis(
                num_runs=10,
                progress_callback=update_progress
            )
            
            progress_window.destroy()
            
            self._update_monte_carlo_visualization(mc_result, min_spacing)
            self._update_monte_carlo_results(mc_result, min_spacing)
            
        except Exception as e:
            progress_window.destroy()
            messagebox.showerror("分析错误", f"蒙特卡洛分析出错: {e}")
    
    def _show_progress(self, message: str):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"{message}\n请稍候...\n")
        self.root.update()
    
    def _update_visualization(self):
        for ax in [self.ax_3d, self.ax_2d, self.ax_error, self.ax_gdop]:
            ax.clear()
        
        self._plot_3d_scenario()
        self._plot_2d_footprint()
        self._plot_error_ellipse()
        self._plot_snr_profile()
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _plot_3d_scenario(self):
        ax = self.ax_3d
        
        ax.axhline(y=0, color='brown', linestyle='-', linewidth=2, label='地面')
        
        for i, sat in enumerate(self.last_result['satellites']):
            pos = sat.position
            ax.scatter(pos[0], pos[2], s=100, c='red', marker='^', label=f'卫星{i+1}' if i == 0 else '')
            ax.annotate(f'S{i+1}', (pos[0], pos[2]), textcoords="offset points", xytext=(5,5), fontsize=8)
        
        true_pos = self.last_result['true_position']
        ax.scatter(true_pos[0], true_pos[2], s=80, c='green', marker='o', label='真实位置')
        
        est_pos = self.last_result['estimated_position']
        ax.scatter(est_pos[0], est_pos[2], s=80, c='orange', marker='x', label='估计位置')
        
        ax.plot([true_pos[0], est_pos[0]], [true_pos[2], est_pos[2]], 'k--', alpha=0.5)
        
        for sat in self.last_result['satellites']:
            pos = sat.position
            ax.plot([pos[0], true_pos[0]], [pos[2], true_pos[2]], 'r-', alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('高度Z (km)')
        ax.set_title('XZ平面投影(侧视图)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_2d_footprint(self):
        ax = self.ax_2d
        
        for i, sat in enumerate(self.last_result['satellites']):
            pos = sat.position
            ax.scatter(pos[0], pos[1], s=100, c='red', marker='^', label=f'卫星{i+1}' if i == 0 else '')
            ax.annotate(f'S{i+1}', (pos[0], pos[1]), textcoords="offset points", xytext=(5,5), fontsize=8)
        
        true_pos = self.last_result['true_position']
        ax.scatter(true_pos[0], true_pos[1], s=80, c='green', marker='o', label='真实位置')
        
        est_pos = self.last_result['estimated_position']
        ax.scatter(est_pos[0], est_pos[1], s=80, c='orange', marker='x', label='估计位置')
        
        ax.plot([true_pos[0], est_pos[0]], [true_pos[1], est_pos[1]], 'k--', alpha=0.5)
        
        circle = plt.Circle((est_pos[0], est_pos[1]), self.params.required_accuracy/1e3, fill=False, color='red', linestyle='--', label='精度要求')
        ax.add_patch(circle)
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_title('XY平面投影(俯视图)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_error_ellipse(self):
        ax = self.ax_error
        
        error = self.last_result['position_error']
        
        categories = ['定位误差', '目标精度']
        values = [error, self.params.required_accuracy]
        colors = ['blue' if error <= self.params.required_accuracy else 'red', 'green']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                   f'{val:.1f}m', ha='center', fontsize=10)
        
        ax.set_ylabel('距离 (m)')
        ax.set_title(f'定位精度分析 (GDOP: {self.last_result["gdop"]:.2f})')
        ax.axhline(y=self.params.required_accuracy, color='green', linestyle='--', alpha=0.5)
    
    def _plot_snr_profile(self):
        ax = self.ax_gdop
        
        satellites = range(1, len(self.last_result['snr_values']) + 1)
        snr_values = self.last_result['snr_values']
        
        bars = ax.bar(satellites, snr_values, color='steelblue', alpha=0.7)
        
        for bar, snr in zip(bars, snr_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{snr:.1f}dB', ha='center', fontsize=9)
        
        ax.set_xlabel('卫星编号')
        ax.set_ylabel('SNR (dB)')
        ax.set_title('各卫星接收信噪比')
        ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='检测门限')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _update_monte_carlo_visualization(self, mc_result: dict, min_spacing: float):
        for ax in [self.ax_3d, self.ax_2d, self.ax_error, self.ax_gdop]:
            ax.clear()
        
        self.ax_3d.plot(mc_result['spacings']/1e3, mc_result['mean_errors'], 'b-o', linewidth=2)
        self.ax_3d.fill_between(mc_result['spacings']/1e3, 
                                mc_result['mean_errors'] - mc_result['std_errors'],
                                mc_result['mean_errors'] + mc_result['std_errors'],
                                alpha=0.3)
        self.ax_3d.axhline(y=self.params.required_accuracy, color='r', linestyle='--', label='目标精度')
        self.ax_3d.axvline(x=min_spacing/1e3, color='g', linestyle=':', label=f'最小间距: {min_spacing/1e3:.0f}km')
        self.ax_3d.set_xlabel('卫星间距 (km)')
        self.ax_3d.set_ylabel('定位误差 (m)')
        self.ax_3d.set_title('定位误差 vs 卫星间距')
        self.ax_3d.legend()
        self.ax_3d.grid(True, alpha=0.3)
        
        self.ax_2d.plot(mc_result['spacings']/1e3, mc_result['gdop_values'], 'g-s', linewidth=2)
        self.ax_2d.set_xlabel('卫星间距 (km)')
        self.ax_2d.set_ylabel('GDOP')
        self.ax_2d.set_title('几何精度因子 vs 卫星间距')
        self.ax_2d.grid(True, alpha=0.3)
        
        self.ax_error.hist(mc_result['mean_errors'], bins=15, color='steelblue', alpha=0.7, edgecolor='black')
        self.ax_error.axvline(x=self.params.required_accuracy, color='r', linestyle='--', label='目标精度')
        self.ax_error.set_xlabel('定位误差 (m)')
        self.ax_error.set_ylabel('频次')
        self.ax_error.set_title('定位误差分布')
        self.ax_error.legend()
        
        self.ax_gdop.text(0.5, 0.7, f'最小卫星间距要求:\n{min_spacing/1e3:.1f} km', 
                         transform=self.ax_gdop.transAxes, fontsize=14,
                         ha='center', va='center',
                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        self.ax_gdop.text(0.5, 0.3, f'目标精度: {self.params.required_accuracy} m\n'
                                    f'带宽: {self.params.bandwidth/1e6:.1f} MHz\n'
                                    f'卫星高度: {self.params.satellite_altitude/1e3:.0f} km',
                         transform=self.ax_gdop.transAxes, fontsize=11,
                         ha='center', va='center',
                         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        self.ax_gdop.set_title('分析结论')
        self.ax_gdop.axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _update_results(self):
        self.results_text.delete(1.0, tk.END)
        
        result = self.last_result
        
        text = "═══════════════════════════════════════\n"
        text += "         TDOA定位仿真结果\n"
        text += "═══════════════════════════════════════\n\n"
        
        text += "【信号参数】\n"
        text += f"  频率: {self.params.frequency/1e9:.2f} GHz\n"
        text += f"  带宽: {self.params.bandwidth/1e6:.1f} MHz\n"
        text += f"  波长: {self.params.wavelength*100:.2f} cm\n\n"
        
        text += "【发射源参数】\n"
        text += f"  EIRP: {self.params.emitter_eirp} dBm\n"
        text += f"  天线阵元: {self.params.antenna_nx}x{self.params.antenna_ny}\n"
        if self.simulator.emitter is not None:
            emitter_pos = self.simulator.emitter.position
            text += f"  位置: ({emitter_pos[0]:.2f}, {emitter_pos[1]:.2f}, {emitter_pos[2]:.2f}) km\n"
        text += f"  相对原点偏移: ({self.lat_var.get()}, {self.lon_var.get()}) km\n\n"
        
        text += "【卫星参数】\n"
        text += f"  数量: {self.params.num_satellites}\n"
        text += f"  高度: {self.params.satellite_altitude/1e3:.0f} km\n"
        text += f"  间距: {self.params.satellite_spacing/1e3:.0f} km\n"
        text += f"  天线增益: {self.params.satellite_antenna_gain} dBi\n\n"
        
        text += "【定位结果】\n"
        text += f"  定位误差: {result['position_error']:.2f} m\n"
        text += f"  目标精度: {self.params.required_accuracy} m\n"
        text += f"  GDOP: {result['gdop']:.3f}\n"
        
        if result['position_error'] <= self.params.required_accuracy:
            text += "  ✓ 满足精度要求\n"
        else:
            text += "  ✗ 未满足精度要求\n"
        
        text += "\n【接收信号功率】\n"
        for i, (rx_power, snr, off_axis) in enumerate(zip(result['rx_power_values'], result['snr_values'], result['off_axis_angles'])):
            text += f"  卫星{i+1}: {rx_power:.1f} dBm (SNR: {snr:.1f} dB, 离轴角: {off_axis:.2f}°)\n"
        
        text += "\n【TDOA测量值】\n"
        for i, tdoa in enumerate(result['tdoa_measurements']):
            text += f"  TDOA_{i}: {tdoa*1e6:.3f} μs\n"
        
        text += "\n═══════════════════════════════════════\n"
        
        self.results_text.insert(tk.END, text)
    
    def _update_monte_carlo_results(self, mc_result: dict, min_spacing: float):
        self.results_text.delete(1.0, tk.END)
        
        text = "═══════════════════════════════════════\n"
        text += "       蒙特卡洛分析结果\n"
        text += "═══════════════════════════════════════\n\n"
        
        text += "【分析参数】\n"
        text += f"  目标精度: {self.params.required_accuracy} m\n"
        text += f"  卫星数量: {self.params.num_satellites}\n"
        text += f"  卫星高度: {self.params.satellite_altitude/1e3:.0f} km\n"
        text += f"  信号带宽: {self.params.bandwidth/1e6:.1f} MHz\n\n"
        
        text += "【关键结论】\n"
        text += f"  最小卫星间距: {min_spacing/1e3:.1f} km\n"
        text += f"  最小间距对应精度: {np.min(mc_result['mean_errors']):.1f} m\n\n"
        
        text += "【不同间距下的定位误差】\n"
        text += "  间距(km)  误差均值(m)  标准差(m)  GDOP\n"
        text += "  ─────────────────────────────────────\n"
        for i in range(len(mc_result['spacings'])):
            text += f"  {mc_result['spacings'][i]/1e3:8.0f}  "
            text += f"{mc_result['mean_errors'][i]:10.1f}  "
            text += f"{mc_result['std_errors'][i]:8.1f}  "
            text += f"{mc_result['gdop_values'][i]:6.2f}\n"
        
        text += "\n【理论分析】\n"
        time_res = 1 / self.params.bandwidth
        range_res = time_res * C
        text += f"  时间分辨率: {time_res*1e9:.2f} ns\n"
        text += f"  距离分辨率: {range_res:.2f} m\n"
        
        text += "\n【建议】\n"
        if min_spacing < self.params.satellite_spacing:
            text += f"  当前间距 {self.params.satellite_spacing/1e3:.0f} km 满足要求\n"
        else:
            text += f"  建议增加卫星间距至 {min_spacing/1e3:.0f} km\n"
        
        text += "\n═══════════════════════════════════════\n"
        
        self.results_text.insert(tk.END, text)
    
    def _run_initial_simulation(self):
        self._run_simulation()


def main():
    root = tk.Tk()
    app = LEOSimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
