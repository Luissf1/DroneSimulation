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
        z_figure_name = f'ResponseZ_Test_{i+1}.png'

        pso_pid_optimization_with_metrics(
            z_des, phi_des, theta_des, psi_des,
            excel_filename, fitness_figure_name, z_figure_name
        )


def pso_pid_optimization_with_metrics(z_des, phi_des, theta_des, psi_des, 
                                     excel_filename, fitness_figure_name, z_figure_name):

    num_tests = 5   # reduce for faster animation tests
    results = []
    best_fitness_over_time = []
    best_global_overall = {'fitness': float('inf')}
    best_results = []
    
    os.makedirs('results', exist_ok=True)
    
    for test in range(num_tests):
        global_best, metrics, convergence_fitness, t_best, X_best = (
            optimize_pid_with_pso_and_metrics(z_des, phi_des, theta_des, psi_des)
        )
        
        best_fitness_over_time.append(convergence_fitness)
        
        result = {
            'Test': test + 1,
            'Fitness': global_best['fitness'],
            'SettlingTime': metrics['t_settle'],
            'Overshoot': metrics['overshoot'],
            'RiseTime': metrics['t_rise'],
            'SteadyError': metrics['steady_error'],
            'ITSE': metrics['ITSE'],
            'IAE': metrics['IAE'],
            'RMSE': metrics['RMSE']
        }
        results.append(result)
        
        if global_best['fitness'] < best_global_overall['fitness']:
            best_global_overall = global_best
            t_global_best = t_best
            X_global_best = X_best
        
        print(f"{test+1}\tFitness: {global_best['fitness']:.4f}")
    
    df = pd.DataFrame(results)
    df.to_excel(os.path.join('results', excel_filename), index=False)
    print(f'\nFile saved: {excel_filename}')
    
    avg_convergence = np.mean(best_fitness_over_time, axis=0)
    plt.figure()
    plt.plot(avg_convergence, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('Average PSO Convergence')
    plt.grid(True)
    plt.savefig(os.path.join('results', fitness_figure_name))
    plt.close()
    
    # === Animate final best response ===
    print("\nAnimating best test response...")
    animate_quadrotor_response(t_global_best, X_global_best, z_des, phi_des, theta_des, psi_des)


def optimize_pid_with_pso_and_metrics(z_des, phi_des, theta_des, psi_des):
    nVar = 12
    VarMin = np.array([2.0, 0.01, 0.1,  0.1, 0.001, 0.1,  0.1, 0.001, 0.1,  0.1, 0.001, 0.1])
    VarMax = np.array([15,  2.0,  5.0, 10,  0.1,   2.0, 10,  0.1,   2.0, 10,  0.1,   2.0])
    MaxIter = 30
    nPop = 20
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
        particle['fitness'], _, _, _, _ = evaluate_pid(particle['position'], z_des, phi_des, theta_des, psi_des)
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
            particles[i]['position'] = np.clip(
                particles[i]['position'] + particles[i]['velocity'], VarMin, VarMax)
            
            fitness, temp_metrics, t, z, X = evaluate_pid(
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
    
    metrics = {'t_settle': np.nan, 'overshoot': np.nan, 't_rise': np.nan,
               'steady_error': np.nan, 'ITSE': np.nan, 'IAE': np.nan, 'RMSE': np.nan}
    
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
        z = X[2, :]
        error_z = z_des - z
        tol = 0.02 * z_des
        idx_settle = np.where(np.abs(error_z) > tol)[0]
        metrics['t_settle'] = t[idx_settle[-1]] if len(idx_settle) > 0 else 0
        metrics['overshoot'] = max(0, (np.max(z) - z_des) / z_des * 100)
        rise_start, rise_end = z_des * 0.1, z_des * 0.9
        try:
            t_rise_start = t[np.where(z >= rise_start)[0][0]]
            t_rise_end = t[np.where(z >= rise_end)[0][0]]
            metrics['t_rise'] = t_rise_end - t_rise_start
        except:
            metrics['t_rise'] = np.nan
        metrics['steady_error'] = np.mean(np.abs(error_z[int(0.9*len(error_z)):]))

        metrics['ITSE'] = np.trapezoid(t * error_z**2, t)
        metrics['IAE'] = np.trapezoid(np.abs(error_z), t)
        metrics['RMSE'] = np.sqrt(np.mean(error_z**2))

        fitness = (0.3 * min(metrics['t_settle']/10, 1) + 
                   0.3 * min(metrics['overshoot']/100, 1) + 
                   0.2 * min(metrics['ITSE']/50, 1) + 
                   0.2 * min(metrics['IAE']/20, 1))
    except:
        fitness = 1000
        metrics = {'t_settle': 100, 'overshoot': 1000, 't_rise': 10,
                   'steady_error': 1, 'ITSE': 50, 'IAE': 20, 'RMSE': 50}
        t = np.linspace(0, 10, 100)
        X = np.zeros((12, len(t)))
    
    return fitness, metrics, t, X[2, :], X


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
