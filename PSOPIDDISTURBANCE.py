import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class PSO_PID_Comparator:
    """
    Comprehensive class to compare PSO-PID performance with and without disturbances
    """
    
    def __init__(self):
        self.results_dir = "comparison_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Test configurations
        self.desired_combinations = np.array([
            [1.0,  0.0,   0.0,    0.0],
            [1.5,  0.1,  -0.1,    0.0],
            [2.0, -0.2,   0.2,    0.0],
            [1.0,  0.0,   0.0,    np.pi/4],
            [0.5, -0.1,  -0.1,   -np.pi/6],
            [2.5,  0.15, -0.15,   np.pi/8],
            [0.8, -0.05,  0.05,  -np.pi/12]
        ])
        
        # PSO parameters
        self.nVar = 12
        self.VarMin = np.array([2.0, 0.01, 0.1,  0.1, 0.001, 0.1,  0.1, 0.001, 0.1,  0.1, 0.001, 0.1])
        self.VarMax = np.array([15,  2.0,  5.0, 10,  0.1,   2.0, 10,  0.1,   2.0, 10,  0.1,   2.0])
        self.MaxIter = 80  # Reduced for faster comparison
        self.nPop = 40     # Reduced for faster comparison
        self.num_tests = 20  # Reduced for faster comparison
        
        # Physical parameters
        self.m = 1.0
        self.g = 9.81
        self.Ix = 0.1
        self.Iy = 0.1
        self.Iz = 0.2
        
    def run_comprehensive_comparison(self):
        """
        Run comprehensive comparison between PSO-PID with and without disturbances
        """
        print("=" * 60)
        print("COMPREHENSIVE PSO-PID COMPARISON")
        print("With vs Without Disturbances")
        print("=" * 60)
        
        all_results = []
        
        for i, combo in enumerate(tqdm(self.desired_combinations, desc="Overall Progress")):
            z_des, phi_des, theta_des, psi_des = combo
            
            print(f'\n\n{"="*50}')
            print(f'TEST CASE {i+1}/{len(self.desired_combinations)}')
            print(f'Desired: z={z_des}, φ={phi_des}, θ={theta_des}, ψ={psi_des}')
            print(f'{"="*50}')
            
            # Run both versions
            result_no_dist = self.run_single_test(
                z_des, phi_des, theta_des, psi_des, disturbance=False, test_id=i+1
            )
            
            result_with_dist = self.run_single_test(
                z_des, phi_des, theta_des, psi_des, disturbance=True, test_id=i+1
            )
            
            # Store comparison results
            comparison_result = {
                'Test_ID': i + 1,
                'z_des': z_des,
                'phi_des': phi_des,
                'theta_des': theta_des,
                'psi_des': psi_des,
                'NoDist_Fitness': result_no_dist['best_fitness'],
                'WithDist_Fitness': result_with_dist['best_fitness'],
                'NoDist_SettlingTime': result_no_dist['best_metrics']['t_settle'],
                'WithDist_SettlingTime': result_with_dist['best_metrics']['t_settle'],
                'NoDist_Overshoot': result_no_dist['best_metrics']['overshoot'],
                'WithDist_Overshoot': result_with_dist['best_metrics']['overshoot'],
                'NoDist_ITSE': result_no_dist['best_metrics']['ITSE'],
                'WithDist_ITSE': result_with_dist['best_metrics']['ITSE'],
                'NoDist_IAE': result_no_dist['best_metrics']['IAE'],
                'WithDist_IAE': result_with_dist['best_metrics']['IAE'],
                'NoDist_RMSE': result_no_dist['best_metrics']['RMSE'],
                'WithDist_RMSE': result_with_dist['best_metrics']['RMSE'],
                'Fitness_Improvement': result_no_dist['best_fitness'] - result_with_dist['best_fitness'],
                'Performance_Ratio': result_with_dist['best_fitness'] / result_no_dist['best_fitness'] if result_no_dist['best_fitness'] != 0 else 0
            }
            
            all_results.append(comparison_result)
            
            # Plot individual comparison
            self.plot_individual_comparison(
                result_no_dist, result_with_dist, z_des, i+1
            )
        
        # Save comprehensive results
        self.save_comprehensive_results(all_results)
        
        # Generate summary statistics
        self.generate_summary_statistics(all_results)
        
        print(f"\nComparison completed! Results saved in '{self.results_dir}' directory")
    
    def run_single_test(self, z_des, phi_des, theta_des, psi_des, disturbance=False, test_id=1):
        """
        Run single PSO-PID optimization test
        """
        version = "With_Disturbance" if disturbance else "No_Disturbance"
        print(f'\nRunning {version} - Test {test_id}')
        
        results = []
        best_fitness_over_time = []
        best_global = {'fitness': float('inf'), 'position': None}
        best_metrics = None
        best_t = None
        best_z = None
        
        # Progress bar for individual tests
        pbar = tqdm(total=self.num_tests, desc=f'{version} Tests')
        
        for test in range(self.num_tests):
            if disturbance:
                global_best, metrics, convergence, t_best, z_best = self.optimize_pid_with_disturbance(
                    z_des, phi_des, theta_des, psi_des
                )
            else:
                global_best, metrics, convergence, t_best, z_best = self.optimize_pid_without_disturbance(
                    z_des, phi_des, theta_des, psi_des
                )
            
            best_fitness_over_time.append(convergence)
            
            # Store results
            result = {
                'Test': test + 1,
                'Fitness': global_best['fitness'],
                'SettlingTime': metrics['t_settle'],
                'Overshoot': metrics['overshoot'],
                'RiseTime': metrics['t_rise'],
                'SteadyError': metrics['steady_error'],
                'ITSE': metrics['ITSE'],
                'IAE': metrics['IAE'],
                'RMSE': metrics['RMSE'],
                'Kp_z': global_best['position'][0],
                'Ki_z': global_best['position'][1],
                'Kd_z': global_best['position'][2],
                'Kp_phi': global_best['position'][3],
                'Ki_phi': global_best['position'][4],
                'Kd_phi': global_best['position'][5],
                'Kp_theta': global_best['position'][6],
                'Ki_theta': global_best['position'][7],
                'Kd_theta': global_best['position'][8],
                'Kp_psi': global_best['position'][9],
                'Ki_psi': global_best['position'][10],
                'Kd_psi': global_best['position'][11]
            }
            results.append(result)
            
            # Update best overall
            if global_best['fitness'] < best_global['fitness']:
                best_global = global_best
                best_metrics = metrics
                best_t = t_best
                best_z = z_best
            
            pbar.update(1)
            pbar.set_postfix({'Best Fitness': f"{best_global['fitness']:.4f}"})
        
        pbar.close()
        
        # Save individual test results
        self.save_individual_results(results, z_des, phi_des, theta_des, psi_des, disturbance, test_id)
        
        return {
            'best_fitness': best_global['fitness'],
            'best_position': best_global['position'],
            'best_metrics': best_metrics,
            'best_t': best_t,
            'best_z': best_z,
            'convergence_data': best_fitness_over_time,
            'all_results': results
        }
    
    def optimize_pid_without_disturbance(self, z_des, phi_des, theta_des, psi_des):
        """PSO optimization without disturbances"""
        return self._optimize_pid(z_des, phi_des, theta_des, psi_des, disturbance=False)
    
    def optimize_pid_with_disturbance(self, z_des, phi_des, theta_des, psi_des):
        """PSO optimization with disturbances"""
        return self._optimize_pid(z_des, phi_des, theta_des, psi_des, disturbance=True)
    
    def _optimize_pid(self, z_des, phi_des, theta_des, psi_des, disturbance=False):
        """Generic PSO optimization function"""
        # PSO parameters
        w = 0.7
        w_damp = 0.97
        c1 = 1.7
        c2 = 1.7
        
        # Initialize
        particles = []
        global_best = {'position': None, 'fitness': float('inf')}
        convergence = np.zeros(self.MaxIter)
        best_metrics = None
        best_t = None
        best_z = None
        
        # Initialize particles
        for i in range(self.nPop):
            position = np.random.uniform(self.VarMin, self.VarMax)
            velocity = np.zeros(self.nVar)
            
            if disturbance:
                fitness, metrics, t, z = self.evaluate_pid_with_disturbance(
                    position, z_des, phi_des, theta_des, psi_des)
            else:
                fitness, metrics, t, z = self.evaluate_pid_without_disturbance(
                    position, z_des, phi_des, theta_des, psi_des)
            
            particle = {
                'position': position,
                'velocity': velocity,
                'fitness': fitness,
                'best_position': position.copy(),
                'best_fitness': fitness
            }
            
            particles.append(particle)
            
            if fitness < global_best['fitness']:
                global_best = {'position': position.copy(), 'fitness': fitness}
                best_metrics = metrics
                best_t = t
                best_z = z
        
        # PSO main loop
        for iter in range(self.MaxIter):
            for i in range(self.nPop):
                # Update velocity
                r1, r2 = np.random.rand(self.nVar), np.random.rand(self.nVar)
                cognitive = c1 * r1 * (particles[i]['best_position'] - particles[i]['position'])
                social = c2 * r2 * (global_best['position'] - particles[i]['position'])
                particles[i]['velocity'] = w * particles[i]['velocity'] + cognitive + social
                
                # Update position
                particles[i]['position'] = np.clip(
                    particles[i]['position'] + particles[i]['velocity'], 
                    self.VarMin, self.VarMax
                )
                
                # Evaluate
                if disturbance:
                    fitness, metrics, t, z = self.evaluate_pid_with_disturbance(
                        particles[i]['position'], z_des, phi_des, theta_des, psi_des)
                else:
                    fitness, metrics, t, z = self.evaluate_pid_without_disturbance(
                        particles[i]['position'], z_des, phi_des, theta_des, psi_des)
                
                particles[i]['fitness'] = fitness
                
                # Update personal best
                if fitness < particles[i]['best_fitness']:
                    particles[i]['best_position'] = particles[i]['position'].copy()
                    particles[i]['best_fitness'] = fitness
                    
                    # Update global best
                    if fitness < global_best['fitness']:
                        global_best = {'position': particles[i]['position'].copy(), 'fitness': fitness}
                        best_metrics = metrics
                        best_t = t
                        best_z = z
            
            convergence[iter] = global_best['fitness']
            w *= w_damp  # Damping inertia weight
        
        return global_best, best_metrics, convergence, best_t, best_z
    
    def evaluate_pid_without_disturbance(self, gains, z_des, phi_des, theta_des, psi_des):
        """Evaluate PID without disturbances"""
        return self._evaluate_pid(gains, z_des, phi_des, theta_des, psi_des, disturbance=False)
    
    def evaluate_pid_with_disturbance(self, gains, z_des, phi_des, theta_des, psi_des):
        """Evaluate PID with disturbances"""
        return self._evaluate_pid(gains, z_des, phi_des, theta_des, psi_des, disturbance=True)
    
    def _evaluate_pid(self, gains, z_des, phi_des, theta_des, psi_des, disturbance=False):
        """Generic PID evaluation function"""
        # Reset integral terms
        self._reset_integrals()
        
        # Initial conditions
        x0 = np.zeros(6)
        xdot0 = np.zeros(6)
        X0 = np.concatenate((x0, xdot0))
        t_span = (0, 10)
        
        try:
            if disturbance:
                sol = solve_ivp(
                    lambda t, X: self.quadrotor_dynamics_with_disturbance(
                        t, X, gains, z_des, phi_des, theta_des, psi_des),
                    t_span, X0, t_eval=np.linspace(0, 10, 1000), method='RK45'
                )
            else:
                sol = solve_ivp(
                    lambda t, X: self.quadrotor_dynamics_without_disturbance(
                        t, X, gains, z_des, phi_des, theta_des, psi_des),
                    t_span, X0, t_eval=np.linspace(0, 10, 1000), method='RK45'
                )
            
            t = sol.t
            X = sol.y
            z = X[2, :]
            
            metrics = self.calculate_metrics(t, z, z_des)
            fitness = self.calculate_fitness(metrics, disturbance)
            
            return fitness, metrics, t, z
            
        except:
            # Return high fitness if simulation fails
            metrics = self.get_default_metrics()
            t = np.linspace(0, 10, 100)
            z = np.zeros_like(t)
            fitness = 1000 if not disturbance else 1500
            
            return fitness, metrics, t, z
    
    def quadrotor_dynamics_without_disturbance(self, t, X, gains, z_des, phi_des, theta_des, psi_des):
        """Quadrotor dynamics without disturbances"""
        return self._quadrotor_dynamics(t, X, gains, z_des, phi_des, theta_des, psi_des, disturbance=False)
    
    def quadrotor_dynamics_with_disturbance(self, t, X, gains, z_des, phi_des, theta_des, psi_des):
        """Quadrotor dynamics with multiple disturbances"""
        return self._quadrotor_dynamics(t, X, gains, z_des, phi_des, theta_des, psi_des, disturbance=True)
    
    def _quadrotor_dynamics(self, t, X, gains, z_des, phi_des, theta_des, psi_des, disturbance=False):
        """Generic quadrotor dynamics function"""
        pos = X[:6]
        vel = X[6:]
        
        # Extract gains
        Kp_z, Ki_z, Kd_z = gains[0], gains[1], gains[2]
        Kp_phi, Ki_phi, Kd_phi = gains[3], gains[4], gains[5]
        Kp_theta, Ki_theta, Kd_theta = gains[6], gains[7], gains[8]
        Kp_psi, Ki_psi, Kd_psi = gains[9], gains[10], gains[11]
        
        # Calculate errors
        err = np.array([
            z_des - pos[2],
            phi_des - pos[3],
            theta_des - pos[4],
            psi_des - pos[5]
        ])
        
        # Update integral terms with anti-windup
        max_int = 10
        if not hasattr(self, 'integrals'):
            self.integrals = np.zeros(4)
        
        self.integrals = np.clip(self.integrals + err, -max_int, max_int)
        
        # PID control
        U1 = Kp_z * err[0] + Ki_z * self.integrals[0] + Kd_z * (-vel[2])
        U2 = Kp_phi * err[1] + Ki_phi * self.integrals[1] + Kd_phi * (-vel[3])
        U3 = Kp_theta * err[2] + Ki_theta * self.integrals[2] + Kd_theta * (-vel[4])
        U4 = Kp_psi * err[3] + Ki_psi * self.integrals[3] + Kd_psi * (-vel[5])
        
        # Disturbances
        F_d = np.zeros(3)
        if disturbance:
            # Multiple disturbance types
            F_d = self._calculate_disturbances(t)
        
        # Dynamics
        acc_lin = np.array([
            (np.cos(pos[3]) * np.sin(pos[4]) * np.cos(pos[5]) + np.sin(pos[3]) * np.sin(pos[5])) * U1 / self.m + F_d[0]/self.m,
            (np.cos(pos[3]) * np.sin(pos[4]) * np.sin(pos[5]) - np.sin(pos[3]) * np.cos(pos[5])) * U1 / self.m + F_d[1]/self.m,
            (np.cos(pos[3]) * np.cos(pos[4]) * U1 / self.m) - self.g + F_d[2]/self.m
        ])
        
        acc_ang = np.array([
            (U2 + (self.Iy - self.Iz) * vel[4] * vel[5]) / self.Ix,
            (U3 + (self.Iz - self.Ix) * vel[3] * vel[5]) / self.Iy,
            (U4 + (self.Ix - self.Iy) * vel[3] * vel[4]) / self.Iz
        ])
        
        return np.concatenate((vel, acc_lin, acc_ang))
    
    def _calculate_disturbances(self, t):
        """Calculate multiple disturbance types"""
        # 1. Wind gusts (sinusoidal)
        wind_gust = np.array([
            0.5 * np.sin(0.5 * t),
            0.5 * np.cos(0.5 * t),
            0.2 * np.sin(0.3 * t)
        ])
        
        # 2. Turbulence (random component)
        turbulence = np.array([
            0.1 * np.random.randn(),
            0.1 * np.random.randn(),
            0.05 * np.random.randn()
        ])
        
        # 3. Step disturbance (appears at t=3s)
        step_dist = np.array([0, 0, 0])
        if 3 <= t <= 7:
            step_dist = np.array([0.3, -0.2, 0.1])
        
        # 4. Impulse disturbance (at t=5s)
        impulse_dist = np.array([0, 0, 0])
        if 4.9 <= t <= 5.1:
            impulse_dist = np.array([0.5, 0.3, -0.2])
        
        total_disturbance = wind_gust + turbulence + step_dist + impulse_dist
        return total_disturbance
    
    def _reset_integrals(self):
        """Reset integral terms"""
        self.integrals = np.zeros(4)
    
    def calculate_metrics(self, t, z, z_des):
        """Calculate performance metrics"""
        error_z = z_des - z
        
        # Settling time (within 2% of desired)
        tol = 0.02 * z_des
        idx_settle = np.where(np.abs(error_z) > tol)[0]
        t_settle = t[idx_settle[-1]] if len(idx_settle) > 0 else 0
        
        # Overshoot
        overshoot = max(0, (np.max(z) - z_des) / z_des * 100) if z_des != 0 else 0
        
        # Rise time (10% to 90%)
        try:
            rise_start_idx = np.where(z >= 0.1 * z_des)[0][0]
            rise_end_idx = np.where(z >= 0.9 * z_des)[0][0]
            t_rise = t[rise_end_idx] - t[rise_start_idx]
        except:
            t_rise = np.nan
        
        # Steady-state error
        steady_error = np.mean(np.abs(error_z[int(0.8 * len(error_z)):])) if len(error_z) > 0 else np.nan
        
        # Integral metrics
        ITSE = np.trapz(t * error_z**2, t) if len(t) > 1 else np.nan
        IAE = np.trapz(np.abs(error_z), t) if len(t) > 1 else np.nan
        RMSE = np.sqrt(np.mean(error_z**2)) if len(error_z) > 0 else np.nan
        
        return {
            't_settle': t_settle,
            'overshoot': overshoot,
            't_rise': t_rise,
            'steady_error': steady_error,
            'ITSE': ITSE,
            'IAE': IAE,
            'RMSE': RMSE
        }
    
    def calculate_fitness(self, metrics, disturbance=False):
        """Calculate fitness function"""
        weights = [0.25, 0.25, 0.25, 0.25] if not disturbance else [0.2, 0.2, 0.3, 0.3]
        
        fitness = (
            weights[0] * min(metrics['t_settle'] / 10, 1) +
            weights[1] * min(metrics['overshoot'] / 100, 1) +
            weights[2] * min(metrics['ITSE'] / 50, 1) +
            weights[3] * min(metrics['IAE'] / 20, 1)
        )
        
        return fitness
    
    def get_default_metrics(self):
        """Return default metrics for failed simulations"""
        return {
            't_settle': 10,
            'overshoot': 100,
            't_rise': 5,
            'steady_error': 1,
            'ITSE': 50,
            'IAE': 20,
            'RMSE': 10
        }
    
    def plot_individual_comparison(self, result_no_dist, result_with_dist, z_des, test_id):
        """Plot comparison for individual test case"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Response comparison
        ax1.plot(result_no_dist['best_t'], result_no_dist['best_z'], 'b-', linewidth=2, label='No Disturbance')
        ax1.plot(result_with_dist['best_t'], result_with_dist['best_z'], 'r-', linewidth=2, label='With Disturbance')
        ax1.axhline(y=z_des, color='k', linestyle='--', label='Desired')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Height z (m)')
        ax1.set_title(f'Response Comparison - Test {test_id}')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Convergence comparison
        avg_no_dist = np.mean(result_no_dist['convergence_data'], axis=0)
        avg_with_dist = np.mean(result_with_dist['convergence_data'], axis=0)
        ax2.plot(avg_no_dist, 'b-', linewidth=2, label='No Disturbance')
        ax2.plot(avg_with_dist, 'r-', linewidth=2, label='With Disturbance')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Fitness')
        ax2.set_title('PSO Convergence Comparison')
        ax2.legend()
        ax2.grid(True)
        ax2.set_yscale('log')
        
        # Plot 3: Metrics comparison (bar chart)
        metrics_names = ['Settling Time', 'Overshoot', 'ITSE', 'IAE']
        no_dist_metrics = [
            result_no_dist['best_metrics']['t_settle'],
            result_no_dist['best_metrics']['overshoot'],
            result_no_dist['best_metrics']['ITSE'],
            result_no_dist['best_metrics']['IAE']
        ]
        with_dist_metrics = [
            result_with_dist['best_metrics']['t_settle'],
            result_with_dist['best_metrics']['overshoot'],
            result_with_dist['best_metrics']['ITSE'],
            result_with_dist['best_metrics']['IAE']
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        ax3.bar(x - width/2, no_dist_metrics, width, label='No Disturbance', alpha=0.7)
        ax3.bar(x + width/2, with_dist_metrics, width, label='With Disturbance', alpha=0.7)
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Values')
        ax3.set_title('Performance Metrics Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics_names, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Fitness distribution
        no_dist_fitness = [r['Fitness'] for r in result_no_dist['all_results']]
        with_dist_fitness = [r['Fitness'] for r in result_with_dist['all_results']]
        
        ax4.boxplot([no_dist_fitness, with_dist_fitness], labels=['No Disturbance', 'With Disturbance'])
        ax4.set_ylabel('Fitness')
        ax4.set_title('Fitness Distribution Comparison')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'Comparison_Test_{test_id}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_individual_results(self, results, z_des, phi_des, theta_des, psi_des, disturbance, test_id):
        """Save individual test results to Excel"""
        version = "With_Disturbance" if disturbance else "No_Disturbance"
        filename = f'PSO_PID_{version}_Test_{test_id}.xlsx'
        
        df = pd.DataFrame(results)
        df.to_excel(os.path.join(self.results_dir, filename), index=False)
    
    def save_comprehensive_results(self, all_results):
        """Save comprehensive comparison results"""
        df = pd.DataFrame(all_results)
        
        # Save detailed results
        df.to_excel(os.path.join(self.results_dir, 'Comprehensive_Comparison_Results.xlsx'), index=False)
        
        # Save summary results
        summary = df[['Test_ID', 'NoDist_Fitness', 'WithDist_Fitness', 'Performance_Ratio']].copy()
        summary['Fitness_Difference'] = summary['NoDist_Fitness'] - summary['WithDist_Fitness']
        summary.to_excel(os.path.join(self.results_dir, 'Summary_Results.xlsx'), index=False)
    
    def generate_summary_statistics(self, all_results):
        """Generate and display summary statistics"""
        df = pd.DataFrame(all_results)
        
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        print(f"\nTotal Test Cases: {len(df)}")
        print(f"Average Fitness (No Disturbance): {df['NoDist_Fitness'].mean():.4f} ± {df['NoDist_Fitness'].std():.4f}")
        print(f"Average Fitness (With Disturbance): {df['WithDist_Fitness'].mean():.4f} ± {df['WithDist_Fitness'].std():.4f}")
        print(f"Average Performance Ratio: {df['Performance_Ratio'].mean():.4f} ± {df['Performance_Ratio'].std():.4f}")
        
        # Count improvements
        improvements = df[df['Fitness_Improvement'] > 0]
        print(f"\nDisturbance-optimized performed better in {len(improvements)}/{len(df)} test cases ({len(improvements)/len(df)*100:.1f}%)")
        
        # Best and worst cases
        best_improvement = df.loc[df['Fitness_Improvement'].idxmax()]
        worst_improvement = df.loc[df['Fitness_Improvement'].idxmin()]
        
        print(f"\nBest improvement: Test {int(best_improvement['Test_ID'])} - Fitness improvement: {best_improvement['Fitness_Improvement']:.4f}")
        print(f"Worst case: Test {int(worst_improvement['Test_ID'])} - Fitness change: {worst_improvement['Fitness_Improvement']:.4f}")
        
        # Create summary plot
        plt.figure(figsize=(12, 8))
        
        # Fitness comparison
        plt.subplot(2, 2, 1)
        plt.bar(df['Test_ID'] - 0.2, df['NoDist_Fitness'], width=0.4, label='No Disturbance', alpha=0.7)
        plt.bar(df['Test_ID'] + 0.2, df['WithDist_Fitness'], width=0.4, label='With Disturbance', alpha=0.7)
        plt.xlabel('Test Case')
        plt.ylabel('Fitness')
        plt.title('Fitness Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Performance ratio
        plt.subplot(2, 2, 2)
        plt.bar(df['Test_ID'], df['Performance_Ratio'], alpha=0.7)
        plt.axhline(y=1, color='r', linestyle='--', label='Equal Performance')
        plt.xlabel('Test Case')
        plt.ylabel('Performance Ratio')
        plt.title('Performance Ratio (With/No Disturbance)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Improvement distribution
        plt.subplot(2, 2, 3)
        plt.hist(df['Fitness_Improvement'], bins=10, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='--', label='No Improvement')
        plt.xlabel('Fitness Improvement')
        plt.ylabel('Frequency')
        plt.title('Fitness Improvement Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Metrics comparison
        plt.subplot(2, 2, 4)
        metrics_to_compare = ['SettlingTime', 'Overshoot', 'ITSE', 'IAE']
        no_dist_avg = [df[f'NoDist_{metric}'].mean() for metric in metrics_to_compare]
        with_dist_avg = [df[f'WithDist_{metric}'].mean() for metric in metrics_to_compare]
        
        x = np.arange(len(metrics_to_compare))
        width = 0.35
        plt.bar(x - width/2, no_dist_avg, width, label='No Disturbance', alpha=0.7)
        plt.bar(x + width/2, with_dist_avg, width, label='With Disturbance', alpha=0.7)
        plt.xlabel('Metrics')
        plt.ylabel('Average Values')
        plt.title('Average Metrics Comparison')
        plt.xticks(x, ['Settle\nTime', 'Overshoot\n%', 'ITSE', 'IAE'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'Summary_Statistics.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to run the comparison"""
    comparator = PSO_PID_Comparator()
    comparator.run_comprehensive_comparison()

if __name__ == "__main__":
    main()