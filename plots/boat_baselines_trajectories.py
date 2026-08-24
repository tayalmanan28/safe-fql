#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ensure repo root is on sys.path if running from results dir like original script
sys.path.append('.')

# Import environment drawing helper (same interface as reference script)
try:
    from env.boat_robot import BoatRobot
except Exception as e:
    print('Warning: failed to import BoatRobot from env.boat_robot:', e)
    BoatRobot = None


# Re-use a compact draw helper similar to the reference implementation
def draw_env_map(ax, env):
    if env is None:
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Agent trajectories (no env info)')
        return
    # sensible limits - try to use env bounds if available
    a = np.linspace(-3, 2, 10)
    b = np.linspace(-2, 2, 15)
    X, Y = np.meshgrid(a, b)

    # Define the vector field
    U = 2 - 0.5 * Y**2  # dx/dt
    V = np.zeros_like(U)  # dy/dt

    # Create the figure and axis
    ax.quiver(X, Y, U, V, color='blue', alpha=0.3, scale=35, width=0.005)
    hazard_size = [0.4, 0.4]
    # hazards
    for i, hazard_pos in enumerate(env.hazard_position_list):
        circle = plt.Circle((hazard_pos[0], hazard_pos[1]), hazard_size[i], facecolor='#6B6B6B', linewidth=1.2, alpha=1.0)
        ax.add_patch(circle)
    
    # goal
    ax.add_patch(plt.Circle((env.goal_position[0], env.goal_position[1]), env.goal_size*10, facecolor='green', alpha=1.0))
    

    # set limits
    try:
        ax.set_xlim(env.xlim)
        ax.set_ylim(env.ylim)
    except Exception:
        ax.set_xlim([-3, 2])
        ax.set_ylim([-2, 2])

    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # ax.set_title('Agent trajectories')


# load single CSV -> Nx2 numpy array (first two columns)
def load_xy_from_csv(p: Path):
    try:
        df = pd.read_csv(p)
        if df.shape[1] < 2:
            raise ValueError('CSV must have at least two columns for x,y')
        arr = np.asarray(df.iloc[:, :2].values, dtype=float)
        return arr
    except Exception as e:
        # fallback: try numpy loadtxt
        try:
            arr = np.loadtxt(str(p), delimiter=',')
            if arr.ndim == 1:
                arr = arr.reshape(-1, arr.shape[0])
            return arr[:, :2]
        except Exception as e2:
            warnings.warn(f'Failed to load {p}: {e}; fallback failed: {e2}')
            return None


# Group trajectories by method name inferred from filename
# expected filename pattern: <method>_trajectory_<n>.csv
# but inference is robust: it will split at '_trajectory' token

def collect_trajectories(traj_dir: Path):
    traj_dir = Path(traj_dir)
    if not traj_dir.exists():
        raise FileNotFoundError(f'Trajectory directory not found: {traj_dir}')
    files = sorted(traj_dir.glob('*_trajectory_*.csv'))
    grouped = {}
    for p in files:
        name = p.name
        if '_trajectory' in name:
            method = name.split('_trajectory')[0]
        else:
            # take leading token before first underscore
            method = name.split('_')[0]
        arr = load_xy_from_csv(p)
        if arr is None or arr.size == 0:
            continue
        grouped.setdefault(method, []).append({'path': p, 'xy': arr})
    return grouped


def plot_grouped_trajectories(grouped, out: Path, env=None, methods_order=None, figsize=(10.5,9)):
    fig, ax = plt.subplots(figsize=figsize)
    draw_env_map(ax, env)

    # default order and palette
    default_order = ['bearl', 'bcql', 'cpq', 'coptidice', 'cdt', 'fisor', 'ours']
    if methods_order is None:
        methods_order = default_order

    # get a color for each method using tab10 palette
    # cmap = plt.cm.get_cmap('tab10')
    method_list = [m for m in methods_order if m in grouped]
    colors = {
    "bcql": "#000000",          # BCQ-Lag
    "bearl": "#b3b3b3",         # BEAR-Lag
    "cpq": "#cdb3ff",           # CPQ
    "coptidice": "#0091ff",     # CoptiDICE
    "c2iql": "#740cad",         # C2IQL
    "fisor": "#850f67",         # FISOR
    "ours": "#ff9500",          # Ours
    "cdt": "#EC0A0A",           # CDT
    }

    # if any other methods present, append them alphabetically
    other_methods = sorted([m for m in grouped.keys() if m not in method_list])
    method_list += other_methods

    for i, method in enumerate(method_list):
        entries = grouped[method]
        color = colors[method] if method in colors else print(f'No color defined for method {method}, using default color.')
        # plot each trajectory file for that method
        for j, entry in enumerate(entries):
            xy = entry['xy']
            # uppercase the method name for legend
            method_cap = method.upper()
            label = method_cap if j == 0 else None  # only label first trajectory for legend
            ax.plot(xy[:,0], xy[:,1], '-', linewidth=4.0, alpha=0.9, color=color, label=label)
            # start and end markers for each trajectory
            ax.scatter(xy[0,0], xy[0,1], marker='o', s=30, color=color, edgecolors='k')
            ax.scatter(xy[-1,0], xy[-1,1], marker='X', s=36, color=color, edgecolors='k')

    # place legend outside
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=len(method_list), fontsize='large')
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f'Saved overlay trajectory plot to: {out}')


# CLI
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_dir', type=str, default='trajectories', help='Directory containing *_trajectory_*.csv files')
    parser.add_argument('--out', type=str, default='all_baselines_trajectories.pdf', help='Output PNG filename')
    parser.add_argument('--env', type=str, default='boat', choices=['boat', 'none'], help='Which env map to draw (boat or none)')
    parser.add_argument('--methods', type=str, default=None, help='Comma separated method order to prefer, e.g. bearl,bcql,cpq')
    args = parser.parse_args()

    traj_dir = Path(args.traj_dir)
    out = Path(args.out)

    grouped = collect_trajectories(traj_dir)
    if not grouped:
        print('No trajectory files found in', traj_dir)
        raise SystemExit(1)

    # optionally create env
    env_obj = None
    if args.env == 'boat':
        if BoatRobot is None:
            print('BoatRobot not available; drawing without env overlay')
        else:
            try:
                env_obj = BoatRobot(seed=0)
            except Exception as e:
                print('Failed to instantiate BoatRobot:', e)
                env_obj = None

    methods_order = args.methods.split(',') if args.methods else None

    plot_grouped_trajectories(grouped, out, env=env_obj, methods_order=methods_order)
