"""
Generate simulated sensor data for anomaly detection training.

This script generates CSV files with 128 samples each, organized in
normal_operation and anomaly_operation directories, following the formulas
from the data generation plan.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
import logging

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataGenerator:
    """Generator for simulated industrial sensor data."""
    
    def __init__(
        self,
        n_equipment: int = 100,
        n_time_steps: int = 40,
        samples_per_file: int = 128,
        random_seed: int = 42,
        output_dir_normal: str = "datasets/normal_operation",
        output_dir_anomaly: str = "datasets/anomaly_operation"
    ):
        """
        Initialize data generator.
        
        Args:
            n_equipment: Number of equipment units
            n_time_steps: Number of time steps per equipment
            samples_per_file: Number of samples per CSV file
            random_seed: Random seed for reproducibility
            output_dir_normal: Directory for normal operation samples
            output_dir_anomaly: Directory for anomaly operation samples
        """
        self.n_equipment = n_equipment
        self.n_time_steps = n_time_steps
        self.samples_per_file = samples_per_file
        self.random_seed = random_seed
        self.output_dir_normal = Path(output_dir_normal)
        self.output_dir_anomaly = Path(output_dir_anomaly)
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Create output directories
        self.output_dir_normal.mkdir(parents=True, exist_ok=True)
        self.output_dir_anomaly.mkdir(parents=True, exist_ok=True)
        
        # Counters
        self.normal_sample_count = 0
        self.anomaly_sample_count = 0
        
        # Buffer for accumulating samples
        self.normal_buffer: List[float] = []
        self.anomaly_buffer: List[float] = []
        
        logger.info(f"DataGenerator initialized:")
        logger.info(f"  Equipment: {n_equipment}")
        logger.info(f"  Time steps per equipment: {n_time_steps}")
        logger.info(f"  Samples per file: {samples_per_file}")
        logger.info(f"  Random seed: {random_seed}")
    
    def generate_vibrations_data(self, t: int) -> Dict:
        """
        Generate data for Vibrations process type.
        
        Args:
            t: Time step
        
        Returns:
            Dictionary with generated values
        """
        # Generate vibration
        vib = np.sin(t / 5) + np.random.normal(0, 0.5)
        
        # Temperature correlated with vibration
        temp = 20 + 2 * vib + np.random.normal(0, 0.5)
        
        # Pressure as quadratic function of vibration
        pres = 30 + 3 * (vib ** 2) + np.random.normal(0, 1)
        
        # Failure calculation
        fail = int((0.3 * vib + 0.2 * temp - 0.1 * pres + np.random.normal(0, 0.5)) > 1)
        
        return {
            'vibration': vib,
            'temperature': temp,
            'pressure': pres,
            'oil_quality': np.nan,
            'contaminant_level': np.nan,
            'acidity': np.nan,
            'hours_operated': np.nan,
            'maintenance_history': np.nan,
            'load': np.nan,
            'failure': fail,
            'sensor_value': temp  # Primary sensor value
        }
    
    def generate_oil_analysis_data(self, t: int) -> Dict:
        """
        Generate data for Oil Analysis process type.
        
        Args:
            t: Time step
        
        Returns:
            Dictionary with generated values
        """
        # Oil quality with temporal trend
        oil_q = np.random.uniform(0, 100) + t * 0.1
        
        # Contaminant level correlated
        cont_level = 50 + 0.5 * oil_q + np.random.normal(0, 5)
        
        # Acidity as power function
        acid = 10 + 0.3 * (oil_q ** 1.5) + np.random.normal(0, 2)
        
        # Failure calculation
        fail = int((0.2 * oil_q - 0.1 * cont_level + 0.05 * acid + np.random.normal(0, 1)) > 5)
        
        return {
            'vibration': np.nan,
            'temperature': np.nan,
            'pressure': np.nan,
            'oil_quality': oil_q,
            'contaminant_level': cont_level,
            'acidity': acid,
            'hours_operated': np.nan,
            'maintenance_history': np.nan,
            'load': np.nan,
            'failure': fail,
            'sensor_value': oil_q  # Primary sensor value
        }
    
    def generate_hours_operated_data(self, t: int) -> Dict:
        """
        Generate data for Hours Operated process type.
        
        Args:
            t: Time step
        
        Returns:
            Dictionary with generated values
        """
        # Hours operated with exponential distribution and trend
        hours_op = np.random.exponential(scale=50) + t * 0.5
        
        # Maintenance history (Poisson)
        maint_hist = np.random.poisson(lam=2)
        
        # Load with temporal trend
        ld = 100 + 0.1 * t + np.random.normal(0, 10)
        
        # Failure calculation
        fail = int((0.05 * hours_op + 0.1 * maint_hist - 0.02 * ld + np.random.normal(0, 1)) > 3)
        
        return {
            'vibration': np.nan,
            'temperature': np.nan,
            'pressure': np.nan,
            'oil_quality': np.nan,
            'contaminant_level': np.nan,
            'acidity': np.nan,
            'hours_operated': hours_op,
            'maintenance_history': maint_hist,
            'load': ld,
            'failure': fail,
            'sensor_value': ld  # Primary sensor value
        }
    
    def introduce_anomaly(self, data: Dict, process_type: str) -> Tuple[Dict, int]:
        """
        Introduce random anomaly with 2% probability.
        
        Args:
            data: Dictionary with sensor data
            process_type: Type of process
        
        Returns:
            Tuple of (modified_data, anomaly_flag)
        """
        if np.random.rand() < 0.02:  # 2% probability
            anomaly = 1
            
            if process_type == 'Vibrations':
                data['vibration'] += np.random.normal(10, 5)
                # Recalculate temperature if vibration changed
                if 'temperature' in data and not np.isnan(data.get('temperature', np.nan)):
                    data['temperature'] = 20 + 2 * data['vibration'] + np.random.normal(0, 0.5)
                    data['sensor_value'] = data['temperature']
            
            elif process_type == 'Oil Analysis':
                data['oil_quality'] += np.random.uniform(50, 100)
                data['sensor_value'] = data['oil_quality']
            
            elif process_type == 'Hours Operated':
                data['load'] += np.random.uniform(50, 100)
                data['sensor_value'] = data['load']
        else:
            anomaly = 0
        
        return data, anomaly
    
    def save_sample_file(self, samples: List[float], is_normal: bool, equipment_id: int) -> None:
        """
        Save a sample file with exactly samples_per_file samples.
        
        Args:
            samples: List of sensor values
            is_normal: Whether this is a normal operation sample
            equipment_id: Equipment identifier
        """
        # Ensure exactly samples_per_file samples
        if len(samples) < self.samples_per_file:
            # Pad with last value
            samples.extend([samples[-1]] * (self.samples_per_file - len(samples)))
        elif len(samples) > self.samples_per_file:
            # Truncate
            samples = samples[:self.samples_per_file]
        
        # Determine filename
        if is_normal:
            filename = self.output_dir_normal / f"equipment_{equipment_id:03d}_sample_{self.normal_sample_count:04d}.csv"
            self.normal_sample_count += 1
        else:
            filename = self.output_dir_anomaly / f"equipment_{equipment_id:03d}_anomaly_{self.anomaly_sample_count:04d}.csv"
            self.anomaly_sample_count += 1
        
        # Save as single column CSV
        pd.DataFrame(samples, columns=['sensor_value']).to_csv(
            filename,
            index=False,
            header=False
        )
    
    def flush_buffers(self) -> None:
        """Flush remaining samples in buffers to files."""
        # Flush normal buffer
        if len(self.normal_buffer) > 0:
            # Find equipment ID from last sample (we'll use a default)
            self.save_sample_file(self.normal_buffer.copy(), is_normal=True, equipment_id=0)
            self.normal_buffer.clear()
        
        # Flush anomaly buffer
        if len(self.anomaly_buffer) > 0:
            self.save_sample_file(self.anomaly_buffer.copy(), is_normal=False, equipment_id=0)
            self.anomaly_buffer.clear()
    
    def add_sample_to_buffer(self, sensor_value: float, is_normal: bool, equipment_id: int) -> None:
        """
        Add sample to buffer and save file when buffer is full.
        
        Args:
            sensor_value: Sensor reading value
            is_normal: Whether this is normal operation
            equipment_id: Equipment identifier
        """
        if is_normal:
            self.normal_buffer.append(sensor_value)
            if len(self.normal_buffer) >= self.samples_per_file:
                self.save_sample_file(self.normal_buffer.copy(), is_normal=True, equipment_id=equipment_id)
                self.normal_buffer.clear()
        else:
            self.anomaly_buffer.append(sensor_value)
            if len(self.anomaly_buffer) >= self.samples_per_file:
                self.save_sample_file(self.anomaly_buffer.copy(), is_normal=False, equipment_id=equipment_id)
                self.anomaly_buffer.clear()
    
    def generate_all_data(self) -> pd.DataFrame:
        """
        Generate all equipment data.
        
        Returns:
            DataFrame with all generated records
        """
        all_records = []
        
        logger.info("Starting data generation...")
        
        for equipment in range(1, self.n_equipment + 1):
            # Select process type
            process_type = np.random.choice(['Vibrations', 'Oil Analysis', 'Hours Operated'])
            
            logger.debug(f"Equipment {equipment}: {process_type}")
            
            for t in range(1, self.n_time_steps + 1):
                # Generate data according to process type
                if process_type == 'Vibrations':
                    data = self.generate_vibrations_data(t)
                elif process_type == 'Oil Analysis':
                    data = self.generate_oil_analysis_data(t)
                elif process_type == 'Hours Operated':
                    data = self.generate_hours_operated_data(t)
                else:
                    raise ValueError(f"Unknown process type: {process_type}")
                
                # Introduce anomaly
                data, anomaly = self.introduce_anomaly(data, process_type)
                
                # Create full record
                record = {
                    'equipment_id': equipment,
                    'time_step': t,
                    'process_type': process_type,
                    'vibration': data.get('vibration', np.nan),
                    'temperature': data.get('temperature', np.nan),
                    'pressure': data.get('pressure', np.nan),
                    'oil_quality': data.get('oil_quality', np.nan),
                    'contaminant_level': data.get('contaminant_level', np.nan),
                    'acidity': data.get('acidity', np.nan),
                    'hours_operated': data.get('hours_operated', np.nan),
                    'maintenance_history': data.get('maintenance_history', np.nan),
                    'load': data.get('load', np.nan),
                    'failure': data.get('failure', 0),
                    'anomaly': anomaly
                }
                
                all_records.append(record)
                
                # Add to buffer for edge device format
                sensor_value = data.get('sensor_value', np.nan)
                if not np.isnan(sensor_value):
                    is_normal = (anomaly == 0 and data.get('failure', 0) == 0)
                    self.add_sample_to_buffer(sensor_value, is_normal, equipment)
            
            # Log progress
            if equipment % 10 == 0:
                logger.info(f"Processed {equipment}/{self.n_equipment} equipment")
        
        # Flush remaining buffers
        self.flush_buffers()
        
        # Create DataFrame
        df = pd.DataFrame(all_records)
        
        # Fill NaN values with column means
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        logger.info(f"Data generation completed:")
        logger.info(f"  Total records: {len(df)}")
        logger.info(f"  Normal samples: {self.normal_sample_count}")
        logger.info(f"  Anomaly samples: {self.anomaly_sample_count}")
        
        return df
    
    def save_complete_dataset(self, df: pd.DataFrame, filename: str = "emulated_data.csv") -> None:
        """
        Save complete dataset for predictive maintenance.
        
        Args:
            df: DataFrame with all records
            filename: Output filename
        """
        output_path = Path(filename)
        df.to_csv(output_path, index=False)
        logger.info(f"Complete dataset saved to: {output_path}")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate simulated sensor data for anomaly detection')
    parser.add_argument(
        '--equipment',
        type=int,
        default=100,
        help='Number of equipment units (default: 100)'
    )
    parser.add_argument(
        '--time-steps',
        type=int,
        default=40,
        help='Number of time steps per equipment (default: 40)'
    )
    parser.add_argument(
        '--samples-per-file',
        type=int,
        default=128,
        help='Number of samples per CSV file (default: 128)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output-normal',
        type=str,
        default='datasets/normal_operation',
        help='Output directory for normal operation samples'
    )
    parser.add_argument(
        '--output-anomaly',
        type=str,
        default='datasets/anomaly_operation',
        help='Output directory for anomaly operation samples'
    )
    parser.add_argument(
        '--save-complete',
        type=str,
        default=None,
        help='Save complete dataset to this file (default: None)'
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = DataGenerator(
        n_equipment=args.equipment,
        n_time_steps=args.time_steps,
        samples_per_file=args.samples_per_file,
        random_seed=args.seed,
        output_dir_normal=args.output_normal,
        output_dir_anomaly=args.output_anomaly
    )
    
    # Generate data
    df = generator.generate_all_data()
    
    # Save complete dataset if requested
    if args.save_complete:
        generator.save_complete_dataset(df, args.save_complete)
    
    logger.info("✅ Data generation completed successfully!")


if __name__ == "__main__":
    main()


