import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from scipy.integrate import solve_ivp
import os

# ==========================================================
# === Animation Function for z, φ, θ, ψ responses ==========
# ==========================================================
def animate_quadrotor_response(t, X, z_des, phi_des, theta_des, psi_des):
    """
    Animate the evolution of z (altitude), phi (roll), theta (pitch), and psi (yaw).
    """
    z = X[2, :]
    phi = X[3, :]
    theta = X[4, :]
    psi = X[5, :]

    fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    labels = ['Altitude (z, m)', 'Roll (φ, rad)', 'Pitch (θ, rad)', 'Yaw (ψ, rad)']
    desired_values = [z_des, phi_des, theta_des, psi_des]
    data = [z, phi, theta, psi]
    lines = []

    for ax, label, d in zip(axs, labels, desired_values):
        line, = ax.plot([], [], lw=2)
        ax.axhline(y=d, color='k', linestyle='--', label='Desired')
        ax.set_ylabel(label)
        ax.legend(loc='upper right')
        ax.grid(True)
        lines.append(line)

    axs[-1].set_xlabel('Time (s)')
    fig.suptitle('Quadrotor Responses (z, φ, θ, ψ)', fontsize=14)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        for i, line in enumerate(lines):
            line.set_data(t[:frame], data[i][:frame])
        return lines

    ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init,
                                  interval=20, blit=True)
    plt.tight_layout()
    plt.show()

# ==========================================================
# === Plotting Functions ===================================
# ==========================================================
def plot_all_responses(t, X, z_des, phi_des, theta_des, psi_des, filename):
    """
    Plot all responses (z, phi, theta, psi) in separate subplots
    """
    z = X[2, :]
    phi = X[3, :]
    theta = X[4, :]
    psi = X[5, :]
    
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # Altitude
    axs[0].plot(t, z, 'b-', linewidth=2)
    axs[0].axhline(y=z_des, color='r', linestyle='--', label='Desired')
    axs[0].set_ylabel('Altitude z (m)')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title('Altitude Response')
    
    # Roll
    axs[1].plot(t, phi, 'g-', linewidth=2)
    axs[1].axhline(y=phi_des, color='r', linestyle='--', label='Desired')
    axs[1].set_ylabel('Roll φ (rad)')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_title('Roll Response')
    
    # Pitch
    axs[2].plot(t, theta, 'm-', linewidth=2)
    axs[2].axhline(y=theta_des, color='r', linestyle='--', label='Desired')
    axs[2].set_ylabel('Pitch θ (rad)')
    axs[2].legend()
    axs[2].grid(True)
    axs[2].set_title('Pitch Response')
    
    # Yaw
    axs[3].plot(t, psi, 'c-', linewidth=2)
    axs[3].axhline(y=psi_des, color='r', linestyle='--', label='Desired')
    axs[3].set_ylabel('Yaw ψ (rad)')
    axs[3].set_xlabel('Time (s)')
    axs[3].legend()
    axs[3].grid(True)
    axs[3].set_title('Yaw Response')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_top5_responses(mejores_resultados, z_des, phi_des, theta_des, psi_des, filename):
    """
    Plot the top 5 trajectories for all variables
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    variables = ['z', 'phi', 'theta', 'psi']
    desired_values = [z_des, phi_des, theta_des, psi_des]
    titles = ['Altitude z', 'Roll φ', 'Pitch θ', 'Yaw ψ']
    colors = ['b', 'g', 'r', 'c']
    
    for i, (var, des, title, color) in enumerate(zip(variables, desired_values, titles, colors)):
        for j, result in enumerate(mejores_resultados[:5]):
            if var == 'z':
                data = result['z']
            elif var == 'phi':
                data = result['phi']
            elif var == 'theta':
                data = result['theta']
            else:  # psi
                data = result['psi']
                
            axs[i].plot(result['t'], data, color=color, alpha=0.7, linewidth=1.5, 
                       label=result['label'] if i == 0 else "")
        
        axs[i].axhline(y=des, color='k', linestyle='--', linewidth=2, label='Desired' if i == 0 else "")
        axs[i].set_ylabel(title)
        axs[i].set_xlabel('Time (s)')
        axs[i].grid(True)
        axs[i].set_title(f'{title} Response')
    
    # Add legend only to the first subplot to avoid repetition
    axs[0].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ==========================================================
# === Main PSO-PID Functions ===============================
# ==========================================================
def pso_pid_multiple_tests():
    desired_combinations = np.array([
        [1.0,  0.0,   0.0,    0.0],
        [1.5,  0.1,  -0.1,    0.0],
        [2.0, -0.2,   0.2,    0.0],
        [1.0,  0.0,   0.0,    np.pi/4],
        [0.5, -0.1,  -0.1,   -np.pi/6]
    ])

    for i in range(len(desired_combinations)):
        print(f'\n============ Test {i+1} ============')
        z_des, phi_des, theta_des, psi_des = desired_combinations[i]
        excel_filename = f'Results_PSO_PID_Test_{i+1}.xlsx'
        fitness_figure_name = f'Convergence_Test_{i+1}.png'
        response_figure_name = f'All_Responses_Test_{i+1}.png'

        pso_pid_optimization_with_metrics(
            z_des, phi_des, theta_des, psi_des,
            excel_filename, fitness_figure_name, response_figure_name
        )


def pso_pid_optimization_with_metrics(z_des, phi_des, theta_des, psi_des, 
                                     excel_filename, fitness_figure_name, response_figure_name):

    num_tests = 5   # Reduced for testing, change to 30 for final run
    results = []
    best_fitness_over_time = []
    best_global_overall = {'fitness': float('inf')}
    mejores_resultados = []
    
    os.makedirs('results', exist_ok=True)
    
    for test in range(num_tests):
        global_best, metrics, convergence_fitness, t_best, X_best = (
            optimize_pid_with_pso_and_metrics(z_des, phi_des, theta_des, psi_des)
        )
        
        best_fitness_over_time.append(convergence_fitness)
        
        # Extract all responses
        z_best = X_best[2, :]
        phi_best = X_best[3, :]
        theta_best = X_best[4, :]
        psi_best = X_best[5, :]
        
        result = {
            'Test': test + 1,
            'Fitness': global_best['fitness'],
            'SettlingTime_z': metrics['t_settle_z'],
            'SettlingTime_phi': metrics['t_settle_phi'],
            'SettlingTime_theta': metrics['t_settle_theta'],
            'SettlingTime_psi': metrics['t_settle_psi'],
            'Overshoot_z': metrics['overshoot_z'],
            'Overshoot_phi': metrics['overshoot_phi'],
            'Overshoot_theta': metrics['overshoot_theta'],
            'Overshoot_psi': metrics['overshoot_psi'],
            'RiseTime_z': metrics['t_rise_z'],
            'RiseTime_phi': metrics['t_rise_phi'],
            'RiseTime_theta': metrics['t_rise_theta'],
            'RiseTime_psi': metrics['t_rise_psi'],
            'SteadyError_z': metrics['steady_error_z'],
            'SteadyError_phi': metrics['steady_error_phi'],
            'SteadyError_theta': metrics['steady_error_theta'],
            'SteadyError_psi': metrics['steady_error_psi'],
            'ITSE_z': metrics['ITSE_z'],
            'ITSE_phi': metrics['ITSE_phi'],
            'ITSE_theta': metrics['ITSE_theta'],
            'ITSE_psi': metrics['ITSE_psi'],
            'IAE_z': metrics['IAE_z'],
            'IAE_phi': metrics['IAE_phi'],
            'IAE_theta': metrics['IAE_theta'],
            'IAE_psi': metrics['IAE_psi'],
            'RMSE_z': metrics['RMSE_z'],
            'RMSE_phi': metrics['RMSE_phi'],
            'RMSE_theta': metrics['RMSE_theta'],
            'RMSE_psi': metrics['RMSE_psi'],
            # PID Gains
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
        
        # Store for top 5 plots
        mejores_resultados.append({
            't': t_best,
            'z': z_best,
            'phi': phi_best,
            'theta': theta_best,
            'psi': psi_best,
            'label': f'Test {test+1}',
            'fitness': global_best['fitness']
        })
        
        if global_best['fitness'] < best_global_overall['fitness']:
            best_global_overall = global_best
            t_global_best = t_best
            X_global_best = X_best
        
        print(f"{test+1}\tFitness: {global_best['fitness']:.4f}\t"
              f"Settle_z: {metrics['t_settle_z']:.4f}\tSettle_phi: {metrics['t_settle_phi']:.4f}")
    
    df = pd.DataFrame(results)
    df.to_excel(os.path.join('results', excel_filename), index=False)
    print(f'\nFile saved: {excel_filename}')
    
    # Average convergence plot
    avg_convergence = np.mean(best_fitness_over_time, axis=0)
    plt.figure()
    plt.plot(avg_convergence, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('Average PSO Convergence')
    plt.grid(True)
    plt.savefig(os.path.join('results', fitness_figure_name))
    plt.close()
    
    # Plot all responses for the best solution
    plot_all_responses(t_global_best, X_global_best, z_des, phi_des, theta_des, psi_des,
                      os.path.join('results', response_figure_name))
    
    # Plot top 5 responses
    mejores_resultados.sort(key=lambda x: x['fitness'])
    plot_top5_responses(mejores_resultados, z_des, phi_des, theta_des, psi_des,
                       os.path.join('results', response_figure_name.replace('.png', '_Top5.png')))
    
    # Animate final best response
    print("\nAnimating best test response...")
    animate_quadrotor_response(t_global_best, X_global_best, z_des, phi_des, theta_des, psi_des)


def optimize_pid_with_pso_and_metrics(z_des, phi_des, theta_des, psi_des):
    nVar = 12
    VarMin = np.array([2.0, 0.01, 0.1,  0.1, 0.001, 0.1,  0.1, 0.001, 0.1,  0.1, 0.001, 0.1])
    VarMax = np.array([15,  2.0,  5.0, 10,  0.1,   2.0, 10,  0.1,   2.0, 10,  0.1,   2.0])
    MaxIter = 30    # Reduced for testing, change to 100 for final run
    nPop = 20       # Reduced for testing, change to 50 for final run
    w = 0.7
    d = 0.97
    c1 = 1.7
    c2 = 1.7
    
    particles = []
    global_best = {'position': None, 'fitness': float('inf')}
    B = np.zeros(MaxIter)
    t_best = []
    X_best = []
    metrics = None
    
    for i in range(nPop):
        particle = {
            'position': np.random.uniform(VarMin, VarMax),
            'velocity': np.zeros(nVar),
            'fitness': float('inf'),
            'best': {'position': None, 'fitness': float('inf')}
        }
        # CORREGIDO: La función evaluate_pid retorna 4 valores, no 5
        fitness, metrics_temp, t_temp, X_temp = evaluate_pid(particle['position'], z_des, phi_des, theta_des, psi_des)
        particle['fitness'] = fitness
        particle['best'] = {'position': particle['position'].copy(), 'fitness': particle['fitness']}
        if particle['fitness'] < global_best['fitness']:
            global_best = {'position': particle['position'].copy(), 'fitness': particle['fitness']}
        particles.append(particle)
    
    for iter in range(MaxIter):
        for i in range(nPop):
            r1, r2 = np.random.rand(nVar), np.random.rand(nVar)
            particles[i]['velocity'] = (w * particles[i]['velocity'] + 
                c1 * r1 * (particles[i]['best']['position'] - particles[i]['position']) + 
                c2 * r2 * (global_best['position'] - particles[i]['position']))
            
            # CORREGIDO: Error de sintaxis en esta línea - corregir los corchetes
            particles[i]['position'] = np.clip(
                particles[i]['position'] + particles[i]['velocity'], VarMin, VarMax)
            
            # CORREGIDO: Usar 4 valores de retorno
            fitness, temp_metrics, t, X = evaluate_pid(
                particles[i]['position'], z_des, phi_des, theta_des, psi_des)
            particles[i]['fitness'] = fitness
            
            if fitness < particles[i]['best']['fitness']:
                particles[i]['best']['position'] = particles[i]['position'].copy()
                particles[i]['best']['fitness'] = fitness
                if fitness < global_best['fitness']:
                    global_best = {'position': particles[i]['position'].copy(), 'fitness': fitness}
                    metrics = temp_metrics
                    t_best = t
                    X_best = X
        
        w = max(w * d, 0.4)
        B[iter] = global_best['fitness']
    
    return global_best, metrics, B, t_best, X_best


def evaluate_pid(gains, z_des, phi_des, theta_des, psi_des):
    m = 1.0
    g = 9.81
    Ix, Iy, Iz = 0.1, 0.1, 0.2
    x0 = np.zeros(6)
    xdot0 = np.zeros(6)
    X0 = np.concatenate((x0, xdot0))
    t_span = (0, 10)
    
    (Kp_z, Ki_z, Kd_z,
     Kp_phi, Ki_phi, Kd_phi,
     Kp_theta, Ki_theta, Kd_theta,
     Kp_psi, Ki_psi, Kd_psi) = gains
    
    # Initialize metrics for all variables
    metrics = {
        't_settle_z': np.nan, 'overshoot_z': np.nan, 't_rise_z': np.nan, 'steady_error_z': np.nan,
        't_settle_phi': np.nan, 'overshoot_phi': np.nan, 't_rise_phi': np.nan, 'steady_error_phi': np.nan,
        't_settle_theta': np.nan, 'overshoot_theta': np.nan, 't_rise_theta': np.nan, 'steady_error_theta': np.nan,
        't_settle_psi': np.nan, 'overshoot_psi': np.nan, 't_rise_psi': np.nan, 'steady_error_psi': np.nan,
        'ITSE_z': np.nan, 'IAE_z': np.nan, 'RMSE_z': np.nan,
        'ITSE_phi': np.nan, 'IAE_phi': np.nan, 'RMSE_phi': np.nan,
        'ITSE_theta': np.nan, 'IAE_theta': np.nan, 'RMSE_theta': np.nan,
        'ITSE_psi': np.nan, 'IAE_psi': np.nan, 'RMSE_psi': np.nan
    }
    
    try:
        sol = solve_ivp(
            lambda t, X: quadrotor_dynamics(
                t, X, m, g, Ix, Iy, Iz,
                Kp_z, Ki_z, Kd_z, Kp_phi, Ki_phi, Kd_phi,
                Kp_theta, Ki_theta, Kd_theta, Kp_psi, Ki_psi, Kd_psi,
                z_des, phi_des, theta_des, psi_des),
            t_span, X0, t_eval=np.linspace(0, 10, 500)
        )
        t, X = sol.t, sol.y
        
        # Extract all responses
        z = X[2, :]
        phi = X[3, :]
        theta = X[4, :]
        psi = X[5, :]
        
        # Calculate metrics for each variable
        variables = [
            (z, z_des, 'z'),
            (phi, phi_des, 'phi'),
            (theta, theta_des, 'theta'),
            (psi, psi_des, 'psi')
        ]
        
        for response, desired, name in variables:
            error = desired - response
            tol = 0.02 * abs(desired) if abs(desired) > 0 else 0.02
            
            # Settling time
            idx_settle = np.where(np.abs(error) > tol)[0]
            metrics[f't_settle_{name}'] = t[idx_settle[-1]] if len(idx_settle) > 0 else 0
            
            # Overshoot
            if desired != 0:
                metrics[f'overshoot_{name}'] = max(0, (np.max(response) - desired) / abs(desired) * 100)
            else:
                metrics[f'overshoot_{name}'] = np.max(np.abs(response)) * 100
            
            # Rise time
            if desired != 0:
                rise_start, rise_end = desired * 0.1, desired * 0.9
                try:
                    t_rise_start = t[np.where(response >= rise_start)[0][0]]
                    t_rise_end = t[np.where(response >= rise_end)[0][0]]
                    metrics[f't_rise_{name}'] = t_rise_end - t_rise_start
                except:
                    metrics[f't_rise_{name}'] = np.nan
            else:
                metrics[f't_rise_{name}'] = np.nan
            
            # Steady state error
            metrics[f'steady_error_{name}'] = np.mean(np.abs(error[int(0.9*len(error)):]))
            
            # Performance indices
            metrics[f'ITSE_{name}'] = np.trapezoid(t * error**2, t)
            metrics[f'IAE_{name}'] = np.trapezoid(np.abs(error), t)
            metrics[f'RMSE_{name}'] = np.sqrt(np.mean(error**2))

        # Combined fitness function considering all variables
        fitness = (
            0.25 * min(metrics['t_settle_z']/10, 1) + 
            0.25 * min(metrics['overshoot_z']/100, 1) + 
            0.10 * min(metrics['ITSE_z']/50, 1) + 
            0.10 * min(metrics['IAE_z']/20, 1) +
            0.08 * min(metrics['t_settle_phi']/5, 1) + 
            0.08 * min(metrics['overshoot_phi']/100, 1) +
            0.08 * min(metrics['t_settle_theta']/5, 1) + 
            0.08 * min(metrics['overshoot_theta']/100, 1) +
            0.04 * min(metrics['t_settle_psi']/5, 1) + 
            0.04 * min(metrics['overshoot_psi']/100, 1)
        )
        
    except Exception as e:
        print(f"Error in evaluate_pid: {e}")
        fitness = 1000
        # Set all metrics to high values in case of failure
        for name in ['z', 'phi', 'theta', 'psi']:
            metrics[f't_settle_{name}'] = 100
            metrics[f'overshoot_{name}'] = 1000
            metrics[f't_rise_{name}'] = 10
            metrics[f'steady_error_{name}'] = 1
            metrics[f'ITSE_{name}'] = 50
            metrics[f'IAE_{name}'] = 20
            metrics[f'RMSE_{name}'] = 50
        t = np.linspace(0, 10, 100)
        X = np.zeros((12, len(t)))
    
    # CORREGIDO: Retornar solo 4 valores
    return fitness, metrics, t, X


def quadrotor_dynamics(t, X, m, g, Ix, Iy, Iz,
                      Kp_z, Ki_z, Kd_z, Kp_phi, Ki_phi, Kd_phi,
                      Kp_theta, Ki_theta, Kd_theta, Kp_psi, Ki_psi, Kd_psi,
                      z_des, phi_des, theta_des, psi_des):

    if not hasattr(quadrotor_dynamics, 'iz'):
        quadrotor_dynamics.iz = quadrotor_dynamics.ip = 0
        quadrotor_dynamics.it = quadrotor_dynamics.ipsi = 0

    pos = X[:6]
    vel = X[6:]
    err = np.array([z_des - pos[2], phi_des - pos[3], theta_des - pos[4], psi_des - pos[5]])

    max_int = 10
    quadrotor_dynamics.iz = np.clip(quadrotor_dynamics.iz + err[0], -max_int, max_int)
    quadrotor_dynamics.ip = np.clip(quadrotor_dynamics.ip + err[1], -max_int, max_int)
    quadrotor_dynamics.it = np.clip(quadrotor_dynamics.it + err[2], -max_int, max_int)
    quadrotor_dynamics.ipsi = np.clip(quadrotor_dynamics.ipsi + err[3], -max_int, max_int)

    U1 = Kp_z * err[0] + Ki_z * quadrotor_dynamics.iz + Kd_z * (-vel[2])
    U2 = Kp_phi * err[1] + Ki_phi * quadrotor_dynamics.ip + Kd_phi * (-vel[3])
    U3 = Kp_theta * err[2] + Ki_theta * quadrotor_dynamics.it + Kd_theta * (-vel[4])
    U4 = Kp_psi * err[3] + Ki_psi * quadrotor_dynamics.ipsi + Kd_psi * (-vel[5])

    acc_lin = np.array([
        (np.cos(pos[3]) * np.sin(pos[4]) * np.cos(pos[5]) + np.sin(pos[3]) * np.sin(pos[5])) * U1 / m,
        (np.cos(pos[3]) * np.sin(pos[4]) * np.sin(pos[5]) - np.sin(pos[3]) * np.cos(pos[5])) * U1 / m,
        (np.cos(pos[3]) * np.cos(pos[4]) * U1 / m) - g
    ])
    acc_ang = np.array([
        (U2 + (Iy - Iz) * vel[4] * vel[5]) / Ix,
        (U3 + (Iz - Ix) * vel[3] * vel[5]) / Iy,
        (U4 + (Ix - Iy) * vel[3] * vel[4]) / Iz
    ])
    return np.concatenate((vel, acc_lin, acc_ang))


# ==========================================================
# === Run All Tests ========================================
# ==========================================================
if __name__ == "__main__":
    pso_pid_multiple_tests()