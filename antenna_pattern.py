import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class PhasedArrayAntenna:
    def __init__(self, nx: int = 34, ny: int = 44, frequency: float = 14.25e9):
        self.nx = nx
        self.ny = ny
        self.frequency = frequency
        self.wavelength = 299792458.0 / frequency
        
    def array_factor_1d(self, n: int, psi: np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            numerator = np.sin(n * psi / 2)
            denominator = n * np.sin(psi / 2)
            af = np.where(np.abs(denominator) < 1e-10, 1.0, numerator / denominator)
        return np.abs(af)
    
    def calculate_pattern(self, theta_deg: float, phi_deg: float = 0,
                          scan_theta_deg: float = 0, scan_phi_deg: float = 0) -> float:
        theta = np.radians(theta_deg)
        phi = np.radians(phi_deg)
        scan_theta = np.radians(scan_theta_deg)
        scan_phi = np.radians(scan_phi_deg)
        
        psi_x = np.pi * (np.sin(phi) - np.sin(scan_phi))
        psi_y = np.pi * (np.sin(theta) - np.sin(scan_theta))
        
        af_x = self.array_factor_1d(self.nx, psi_x)
        af_y = self.array_factor_1d(self.ny, psi_y)
        
        af = af_x * af_y
        
        return float(af)
    
    def calculate_gain_db(self, theta_deg: float, phi_deg: float = 0,
                          scan_theta_deg: float = 0, scan_phi_deg: float = 0) -> float:
        pattern = self.calculate_pattern(theta_deg, phi_deg, scan_theta_deg, scan_phi_deg)
        max_pattern = self.calculate_pattern(scan_theta_deg, scan_phi_deg, scan_theta_deg, scan_phi_deg)
        
        if max_pattern > 0 and pattern > 0:
            pattern_norm = pattern / max_pattern
        else:
            return -100
        
        D0 = 4 * np.pi * self.nx * self.ny * (0.5) ** 2
        D0_db = 10 * np.log10(D0)
        
        efficiency = 0.7
        gain_db = D0_db + 10 * np.log10(efficiency) + 20 * np.log10(pattern_norm)
        
        return float(gain_db)
    
    def get_gain_at_angle(self, theta_deg: float, phi_deg: float = 0,
                          scan_theta_deg: float = 0, scan_phi_deg: float = 0) -> float:
        return self.calculate_gain_db(theta_deg, phi_deg, scan_theta_deg, scan_phi_deg)
    
    def get_main_lobe_gain(self) -> float:
        return self.calculate_gain_db(0, 0)


class AntennaPatternGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("相控阵天线方向图仿真")
        self.root.geometry("1400x900")
        
        self.antenna = PhasedArrayAntenna()
        
        self._setup_ui()
        self._plot_patterns()
    
    def _update_title(self):
        self.root.title(f"{self.antenna.nx}x{self.antenna.ny}相控阵天线方向图仿真")
    
    def _setup_ui(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        ttk.Label(control_frame, text="阵元列数 Nx:").pack(side=tk.LEFT, padx=5)
        self.nx_var = tk.StringVar(value="44")
        ttk.Entry(control_frame, textvariable=self.nx_var, width=6).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="阵元行数 Ny:").pack(side=tk.LEFT, padx=5)
        self.ny_var = tk.StringVar(value="34")
        ttk.Entry(control_frame, textvariable=self.ny_var, width=6).pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Label(control_frame, text="水平角 Az (度):").pack(side=tk.LEFT, padx=5)
        self.az_var = tk.StringVar(value="0")
        ttk.Entry(control_frame, textvariable=self.az_var, width=8).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="俯仰角 El (度):").pack(side=tk.LEFT, padx=5)
        self.el_var = tk.StringVar(value="0")
        ttk.Entry(control_frame, textvariable=self.el_var, width=8).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="更新方向图", command=self._update_patterns).pack(side=tk.LEFT, padx=20)
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        right_frame = ttk.LabelFrame(main_frame, text="方向图参数", padding=10, width=420)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        self.info_text = tk.Text(right_frame, font=('Consolas', 9), wrap=tk.NONE)
        self.info_text.pack(fill=tk.BOTH, expand=True)
    
    def _plot_patterns(self):
        self.fig.clear()
        
        try:
            nx = int(self.nx_var.get())
            ny = int(self.ny_var.get())
        except ValueError:
            nx, ny = 34, 44
        
        self.antenna = PhasedArrayAntenna(nx=nx, ny=ny)
        self._update_title()
        
        az_deg = float(self.az_var.get())
        el_deg = float(self.el_var.get())
        
        ax1 = self.fig.add_subplot(221)
        self._plot_azimuth_pattern(ax1, az_deg, el_deg)
        
        ax2 = self.fig.add_subplot(222)
        self._plot_elevation_pattern(ax2, az_deg, el_deg)
        
        ax3 = self.fig.add_subplot(223)
        self._plot_2d_pattern(ax3, az_deg, el_deg)
        
        ax4 = self.fig.add_subplot(224)
        self._plot_3d_pattern(ax4, az_deg, el_deg)
        
        self._update_info(az_deg, el_deg)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _plot_azimuth_pattern(self, ax, az_deg: float, el_deg: float):
        az_range = np.linspace(-60, 60, 241)
        
        gains = []
        for az in az_range:
            gains.append(self.antenna.calculate_gain_db(el_deg, az, el_deg, az_deg))
        gain_db = np.array(gains)
        gain_db = np.clip(gain_db, -50, 50)
        
        ax.plot(az_range, gain_db, 'b-', linewidth=1.5)
        ax.axvline(x=az_deg, color='r', linestyle='--', alpha=0.7, label=f'扫描位置: {az_deg}°')
        ax.set_xlabel('水平角 Az (度)')
        ax.set_ylabel('增益 (dB)')
        ax.set_title(f'水平面方向图 ({self.antenna.nx}阵元边)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-50, 40)
        ax.legend(loc='upper right')
    
    def _plot_elevation_pattern(self, ax, az_deg: float, el_deg: float):
        el_range = np.linspace(-60, 60, 241)
        
        gains = []
        for el in el_range:
            gains.append(self.antenna.calculate_gain_db(el, az_deg, el_deg, az_deg))
        gain_db = np.array(gains)
        gain_db = np.clip(gain_db, -50, 50)
        
        ax.plot(el_range, gain_db, 'b-', linewidth=1.5)
        ax.axvline(x=el_deg, color='r', linestyle='--', alpha=0.7, label=f'扫描位置: {el_deg}°')
        ax.set_xlabel('俯仰角 El (度)')
        ax.set_ylabel('增益 (dB)')
        ax.set_title(f'俯仰面方向图 ({self.antenna.ny}阵元边)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-50, 40)
        ax.legend(loc='upper right')
    
    def _plot_2d_pattern(self, ax, az_deg: float, el_deg: float):
        az_range = np.linspace(-60, 60, 121)
        el_range = np.linspace(-60, 60, 121)
        
        gains = np.zeros((len(el_range), len(az_range)))
        
        for i, el in enumerate(el_range):
            for j, az in enumerate(az_range):
                gains[i, j] = self.antenna.calculate_gain_db(el, az, el_deg, az_deg)
        
        gain_db = np.clip(gains, -50, 40)
        
        im = ax.pcolormesh(az_range, el_range, gain_db, shading='auto', cmap='jet', vmin=-50, vmax=40)
        ax.plot(az_deg, el_deg, 'w*', markersize=15, label='扫描位置')
        ax.set_xlabel('水平角 Az (度)')
        ax.set_ylabel('俯仰角 El (度)')
        ax.set_title('2D方向图')
        ax.legend(loc='upper right')
        
        cbar = self.fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('增益 (dB)')
    
    def _plot_3d_pattern(self, ax, az_deg: float, el_deg: float):
        ax = self.fig.add_subplot(224, projection='3d')
        
        az_range = np.linspace(-60, 60, 61)
        el_range = np.linspace(-60, 60, 61)
        
        AZ, EL = np.meshgrid(az_range, el_range)
        gains = np.zeros_like(AZ)
        
        for i in range(len(el_range)):
            for j in range(len(az_range)):
                gains[i, j] = self.antenna.calculate_gain_db(el_range[i], az_range[j], el_deg, az_deg)
        
        gains = np.clip(gains, -30, 40)
        
        ax.plot_surface(AZ, EL, gains, cmap='jet', alpha=0.8)
        ax.set_xlabel('水平角 Az (度)')
        ax.set_ylabel('俯仰角 El (度)')
        ax.set_zlabel('增益 (dB)')
        ax.set_title('3D方向图')
        ax.view_init(elev=25, azim=-60)
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    def _update_info(self, az_deg: float, el_deg: float):
        self.info_text.delete(1.0, tk.END)
        
        main_lobe_gain = self.antenna.calculate_gain_db(el_deg, az_deg, el_deg, az_deg)
        
        info = "=" * 40 + "\n"
        info += "         天线参数\n"
        info += "=" * 40 + "\n"
        info += f"阵元配置: {self.antenna.nx} x {self.antenna.ny}\n"
        info += f"扫描角度: Az={az_deg:.1f}°, El={el_deg:.1f}°\n"
        info += f"主瓣增益: {main_lobe_gain:.2f} dB\n\n"
        
        info += "=" * 40 + "\n"
        info += "      水平面方向图旁瓣分析\n"
        info += "=" * 40 + "\n"
        
        az_range = np.linspace(-60, 60, 481)
        gains_az = []
        for az in az_range:
            gains_az.append(self.antenna.calculate_gain_db(el_deg, az, el_deg, az_deg))
        gains_az = np.array(gains_az)
        
        peaks_az = self._find_peaks(gains_az, az_range)
        
        for i, (angle, gain) in enumerate(peaks_az):
            relative_db = gain - peaks_az[0][1]
            if i == 0:
                info += f"[主瓣] Az={angle:+.1f}°, 增益={gain:.2f}dB\n"
            else:
                info += f" 旁瓣{i}: Az={angle:+.1f}°, 增益={gain:.2f}dB, 相对主瓣={relative_db:.2f}dB\n"
        
        info += "\n" + "=" * 40 + "\n"
        info += "      俯仰面方向图旁瓣分析\n"
        info += "=" * 40 + "\n"
        
        el_range = np.linspace(-60, 60, 481)
        gains_el = []
        for el in el_range:
            gains_el.append(self.antenna.calculate_gain_db(el, az_deg, el_deg, az_deg))
        gains_el = np.array(gains_el)
        
        peaks_el = self._find_peaks(gains_el, el_range)
        
        for i, (angle, gain) in enumerate(peaks_el):
            relative_db = gain - peaks_el[0][1]
            if i == 0:
                info += f"[主瓣] El={angle:+.1f}°, 增益={gain:.2f}dB\n"
            else:
                info += f" 旁瓣{i}: El={angle:+.1f}°, 增益={gain:.2f}dB, 相对主瓣={relative_db:.2f}dB\n"
        
        info += "\n" + "=" * 40 + "\n"
        info += "        波束宽度分析\n"
        info += "=" * 40 + "\n"
        
        az_3db = self._find_3db_width(az_range, gains_az, peaks_az[0][1])
        el_3db = self._find_3db_width(el_range, gains_el, peaks_el[0][1])
        
        info += f"水平面3dB波束宽度: {az_3db:.2f}°\n"
        info += f"俯仰面3dB波束宽度: {el_3db:.2f}°\n"
        
        self.info_text.insert(tk.END, info)
    
    def _find_3db_width(self, angles: np.ndarray, gains: np.ndarray, peak_gain: float) -> float:
        target_gain = peak_gain - 3
        above_3db = gains >= target_gain
        
        if np.sum(above_3db) == 0:
            return 0.0
        
        indices = np.where(above_3db)[0]
        width = angles[indices[-1]] - angles[indices[0]]
        return abs(width)
    
    def _find_peaks(self, gains: np.ndarray, angles: np.ndarray) -> List[Tuple[float, float]]:
        peaks = []
        for i in range(1, len(gains) - 1):
            if gains[i] > gains[i-1] and gains[i] > gains[i+1]:
                peaks.append((angles[i], gains[i]))
        
        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks
    
    def _update_patterns(self):
        self._plot_patterns()


def calculate_antenna_gain_for_satellite(antenna: PhasedArrayAntenna,
                                          sat_position: np.ndarray,
                                          emitter_position: np.ndarray,
                                          emitter_pointing: np.ndarray) -> float:
    direction = sat_position - emitter_position
    distance = np.linalg.norm(direction)
    direction_unit = direction / distance
    
    cos_angle = np.dot(emitter_pointing, direction_unit)
    cos_angle = np.clip(cos_angle, -1, 1)
    off_boresight_angle = np.degrees(np.arccos(cos_angle))
    
    phi_deg = 0
    theta_deg = off_boresight_angle
    
    gain_db = antenna.get_gain_at_angle(theta_deg, phi_deg, 0, 0)
    
    return gain_db


if __name__ == "__main__":
    root = tk.Tk()
    app = AntennaPatternGUI(root)
    root.mainloop()
