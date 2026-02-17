"""
Phased Array Antenna Pattern Module
====================================
Models the radiation pattern of a rectangular phased array antenna.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class PhasedArrayAntenna:
    nx: int
    ny: int
    frequency: float
    element_spacing: float = 0.5
    
    @property
    def wavelength(self) -> float:
        return 299792458.0 / self.frequency
    
    @property
    def element_spacing_m(self) -> float:
        return self.element_spacing * self.wavelength
    
    def calculate_gain_db(self, theta_deg: float, phi_deg: float, 
                          scan_theta: float = 0, scan_phi: float = 0) -> float:
        theta = np.radians(theta_deg)
        phi = np.radians(phi_deg)
        scan_theta_rad = np.radians(scan_theta)
        scan_phi_rad = np.radians(scan_phi)
        
        k = 2 * np.pi / self.wavelength
        dx = self.element_spacing_m
        dy = self.element_spacing_m
        
        element_gain = 10 * np.log10(1.5)
        
        if self.nx > 1:
            psi_x = k * dx * (np.sin(theta) * np.cos(phi) - 
                             np.sin(scan_theta_rad) * np.cos(scan_phi_rad))
            array_factor_x = np.sin(self.nx * psi_x / 2) / (self.nx * np.sin(psi_x / 2 + 1e-10))
            array_factor_x = np.abs(array_factor_x)
        else:
            array_factor_x = 1.0
        
        if self.ny > 1:
            psi_y = k * dy * (np.sin(theta) * np.sin(phi) - 
                             np.sin(scan_theta_rad) * np.sin(scan_phi_rad))
            array_factor_y = np.sin(self.ny * psi_y / 2) / (self.ny * np.sin(psi_y / 2 + 1e-10))
            array_factor_y = np.abs(array_factor_y)
        else:
            array_factor_y = 1.0
        
        array_gain = 20 * np.log10(array_factor_x * array_factor_y + 1e-10)
        
        total_gain = element_gain + array_gain
        
        return total_gain
    
    def get_3db_beamwidth(self) -> tuple:
        approx_bw_deg = 51 / (self.nx * self.element_spacing)
        return approx_bw_deg, approx_bw_deg
    
    def get_directivity_db(self) -> float:
        return 10 * np.log10(4 * np.pi * self.nx * self.ny * (self.element_spacing ** 2))
