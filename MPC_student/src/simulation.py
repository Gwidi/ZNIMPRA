import matplotlib
import mujoco
import mujoco_viewer
import numpy as np
from matplotlib import pyplot as plt
import csv
import casadi as ca
from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, ACADOS_INFTY
import scipy.linalg

class Pendulum():
    def __init__(self) -> None:
        self.dt = 0.05
        self.b = 0.05
        self.m = 1.
        self.g = 9.81
        self.model, self.data = self.create_mujoco_model()
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

    def create_mujoco_model(self):
        pendulum = f"""
        <mujoco>
        <option timestep="{self.dt}" integrator="RK4">
            <flag energy="enable"/>
        </option>

        <default>
            <joint type="hinge" axis="0 -1 0"/>
            <geom type="capsule" size=".02"/>
        </default>

        <worldbody>
            <light pos="0 -.4 1"/>
            <camera name="fixed" pos="0 -2 0.2" xyaxes="1 0 0 0 0 1"/>
            <body name="pole" pos="0 0 0">
            <joint axis="0 1 0" name="hinge" pos="0 0 0" type="hinge"/>
            <geom fromto="0 0 0 0 0 -0.3" name="cpole" rgba="0 0.7 0.7 1" size="0.045 0.15" type="capsule" mass="{self.m}"/>
            <site name="tip" pos="0 0 -0.3" size="0.01 0.01"/>
            </body>
        </worldbody>

        <actuator>
            <motor name="my_motor" joint="hinge" gear="1"/>
        </actuator>
        </mujoco>
        """
        
        model = mujoco.MjModel.from_xml_string(pendulum)
        data = mujoco.MjData(model)
        return model, data

    def init_episode(self):
        mujoco.mj_resetData(self.model, self.data)

    def step_sim(self, u, render=False):
        self.data.ctrl[0] = u
        mujoco.mj_step(self.model, self.data)
        if render and self.viewer.is_alive:
            self.viewer.render()

    def set_state(self, q, q_dot):
        self.data.joint('hinge').qpos[0] = q
        self.data.joint('hinge').qvel[0] = q_dot
        mujoco.mj_forward(self.model, self.data)

    def get_state(self):
        return self.data.joint('hinge').qpos[0], self.data.joint('hinge').qvel[0]
    
def simulate():
    env = Pendulum()
    env.init_episode()
    env.set_state(q=0.5, q_dot=0.0)
    for _ in range(1000):
        env.step_sim(u=0.0, render=True)

def identify_parameters():
    env = Pendulum()
    env.init_episode()
    env.set_state(q=1.0, q_dot=0.0)
    
    # Analityczna postać równania ruchu wahadła:
    #   m*l^2*ddq + b*dq + m*g*l*sin(q) = u

    data_q = []    # pozycje
    data_dq = []   # prędkości
    data_ddq = []  # przyspieszenia (można odczytać z data.qacc lub różniczkować)
    data_tau = []  # momenty (u)

    # Zbieranie danych
    for _ in range(2000):
        # Podajemy losowe sterowanie, aby "pobudzić" system
        u = np.random.uniform(-1, 1) 
        env.step_sim(u=u, render=False)
        
        q, q_dot, q_ddot = env.get_state()
        
        data_q.append(q)
        data_dq.append(q_dot)
        data_ddq.append(q_ddot)
        data_tau.append(u)

    Phi = np.column_stack([
    np.array(data_ddq), 
    np.array(data_dq), 
    np.sin(np.array(data_q))
    ])

    y = np.array(data_tau)

    # Obliczenie optymalnego p (Moore-Penrose Pseudo-Inverse)
    p = np.linalg.pinv(Phi) @ y

    I_hat, b_hat, mgl_hat = p
    print(f"Zidentyfikowane parametry: I={I_hat:.4f}, b={b_hat:.4f}, mgl={mgl_hat:.4f}")

def get_pendulum_model():
    J = 0.0339
    b = 0.0
    mgl = 1.4715
    

    theta = ca.SX.sym('theta')
    dtheta = ca.SX.sym('dtheta')
    u = ca.SX.sym('u') # moment sterujący 


    x = ca.vertcat(theta, dtheta)
    xdot = ca.SX.sym('xdot', x.shape)

    gravity_torque = mgl * ca.sin(theta)
    friction_torque = b * dtheta

    ddtheta = (u - gravity_torque - friction_torque) / J

    f_expl_expr = ca.vertcat(dtheta, ddtheta)

    
    model = AcadosModel()
    model.name = 'inverted_pendulum'
    model.x = x
    model.xdot = xdot
    model.u = u
    model.f_expl_expr = f_expl_expr
    model.f_impl_expr = xdot -f_expl_expr

    return model

def setup_ocp_solver(x0, umax, dt_0, N_horizon, Tf,
                     RTI=False, timeout_max_time=0.0, heuristic="ZERO",
                     with_anderson_acceleration=False,
                     nlp_solver_max_iter = 20, tol = 1e-6, with_abs_cost=False,
                     hessian_approx = 'GAUSS_NEWTON', regularize_method = 'NO_REGULARIZE',
                     anderson_activation_threshold=ACADOS_INFTY) -> AcadosOcpSolver:
    ocp = AcadosOcp()

    model = get_pendulum_model()
    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.solver_options.N_horizon = N_horizon

    # set cost module
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    Q_mat = np.diag([50., 1.])
    R_mat = np.diag([1e-1])

    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.W_e = Q_mat

    ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
    ocp.model.cost_y_expr_e = model.x

    yref = np.array([np.pi, 0.0, 0.0])
    yref_e = np.array([np.pi, 0.0])
    ocp.cost.yref = yref
    ocp.cost.yref_e = yref_e

    # set constraints
    ocp.constraints.lbu = np.array([-umax])
    ocp.constraints.ubu = np.array([+umax])
    ocp.constraints.idxbu = np.array([0])

    if with_abs_cost:
        val = 1.4
        # add cost term abs(x[0]-val) via slacks
        # ocp.constraints.idxbx_e = np.array([0])
        # ocp.constraints.lbx_e = np.array([val])
        # ocp.constraints.ubx_e = np.array([val])
        # ocp.constraints.idxsbx_e = np.array([0])

        # ocp.cost.zl_e = 1e2 * np.array([1.0])
        # ocp.cost.zu_e = 1e2 * np.array([1.0])
        # ocp.cost.Zl_e = np.array([10.0])
        # ocp.cost.Zu_e = np.array([10.0])

        ocp.constraints.idxbx = np.array([0])
        ocp.constraints.lbx = np.array([val])
        ocp.constraints.ubx = np.array([val])
        ocp.constraints.idxsbx = np.array([0])

        ocp.cost.zl = 1e3 * np.array([1.0])
        ocp.cost.zu = 1e3 * np.array([1.0])
        ocp.cost.Zl = 0.0 * np.array([1.0])
        ocp.cost.Zu = 0.0 * np.array([1.0])


    ocp.constraints.x0 = x0

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = hessian_approx
    ocp.solver_options.regularize_method = regularize_method
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.reg_epsilon = 5e-2

    # NOTE we use a nonuniform grid!
    ocp.solver_options.time_steps = np.array([dt_0] + [(Tf-dt_0)/(N_horizon-1)]*(N_horizon-1))
    ocp.solver_options.sim_method_num_steps = np.array([1] + [2]*(N_horizon-1))
    ocp.solver_options.levenberg_marquardt = 1e-6
    ocp.solver_options.nlp_solver_max_iter = nlp_solver_max_iter
    ocp.solver_options.with_anderson_acceleration = with_anderson_acceleration
    ocp.solver_options.anderson_activation_threshold = anderson_activation_threshold

    ocp.solver_options.nlp_solver_type = 'SQP_RTI' if RTI else 'SQP'
    ocp.solver_options.qp_solver_cond_N = N_horizon
    ocp.solver_options.tol = tol

    ocp.solver_options.tf = Tf

    # timeout
    ocp.solver_options.timeout_max_time = timeout_max_time
    ocp.solver_options.timeout_heuristic = heuristic

    solver_json = 'acados_ocp_' + model.name + '.json'
    ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json, verbose=False)

    return ocp_solver

def mpc_control():
    env = Pendulum()
    env.init_episode()
    env.set_state(q=1.0, q_dot=0.0)

    ocp_solver = setup_ocp_solver([1.0 , 0.0], umax=2.0, dt_0=env.dt, N_horizon=20, Tf=1.0)

    status = ocp_solver.solve()
    if status != 0:
        print(f"Solver failed with status {status}")

    for _ in range(2000):
        q, q_dot = env.get_state()
        x_current = np.array([q, q_dot])

        ocp_solver.set(0, "lbx", x_current)
        ocp_solver.set(0, "ubx", x_current)
        ocp_solver.set(0, "x", x_current)
        status = ocp_solver.solve()
        if status != 0:
            print(f"Solver failed with status {status}")
        u_optimal = ocp_solver.get(0, "u")
        env.step_sim(u_optimal[0], render=True)



        

    

if __name__ == "__main__":
    #identify_parameters()
    mpc_control()